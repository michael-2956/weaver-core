''' Particle Transformer (ParT)

Paper: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/2202.03772
'''
import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

from weaver.utils.logger import _logger


@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part


def to_pt_rap_phi_m(x, return_mass=True, eps=1e-8, for_onnx=False):
    # NOTE: (+) rename function, split with (1, 1, 1, 1, 1)
    # and do px, py, pz, energy, preds = ...
    # then return return torch.cat((pt, rapidity, phi, preds), dim=1)
    # or                 torch.cat((pt, rapidity, phi, preds, m), dim=1)

    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = (atan2 if for_onnx else torch.atan2)(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps)**(-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)


def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8, for_onnx=False):
    # NOTE: pti, rapi, phii, predsi = ...
    #       ptj, rapj, phij, predsj = ...

    pti, rapi, phii = to_pt_rap_phi_m(xi, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_pt_rap_phi_m(xj, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)

    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin = ((pti <= ptj) * pti + (pti > ptj) * ptj) if for_onnx else torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:
        xij = xi + xj
        lnm2 = torch.log(to_m2(xij, eps=eps))
        outputs.append(lnm2)

    # NOTE: add here a new num_outputs > 4
    # shift all next indices by 1
    # should be outputs.append(predsij)
    # predsij is formed with predsi, predsj

    if num_outputs > 4:
        lnds2 = torch.log(torch.clamp(-to_m2(xi - xj, eps=None), min=eps))
        outputs.append(lnds2)

    # the following features are not symmetric for (i, j)
    if num_outputs > 5:
        xj_boost = boost(xj, xij)
        costheta = (p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps)).sum(dim=1, keepdim=True)
        outputs.append(costheta)

    if num_outputs > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        outputs += [deltarap, deltaphi]

    assert (len(outputs) == num_outputs)
    return torch.cat(outputs, dim=1)


def build_sparse_tensor(uu, idx, seq_len):
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs)
    # return: (N, C, seq_len, seq_len)
    batch_size, num_fts, num_pairs = uu.size()
    idx = torch.min(idx, torch.ones_like(idx) * seq_len)
    i = torch.cat((
        torch.arange(0, batch_size, device=uu.device).repeat_interleave(num_fts * num_pairs).unsqueeze(0),
        torch.arange(0, num_fts, device=uu.device).repeat_interleave(num_pairs).repeat(batch_size).unsqueeze(0),
        idx[:, :1, :].expand_as(uu).flatten().unsqueeze(0),
        idx[:, 1:, :].expand_as(uu).flatten().unsqueeze(0),
    ), dim=0)
    return torch.sparse_coo_tensor(
        i, uu.flatten(),
        size=(batch_size, num_fts, seq_len + 1, seq_len + 1),
        device=uu.device).to_dense()[:, :, :seq_len, :seq_len]


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # From https://github.com/rwightman/pytorch-image-models/blob/18ec173f95aa220af753358bf860b16b6691edb2/timm/layers/weight_init.py#L8
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class SequenceTrimmer(nn.Module):

    def __init__(
            self,
            enabled=False,
            target=(0.9, 1.02),
            warmup_steps=5,
            trim_in_test=False,
            fixed_length=None,
            shuffle_before_cut=True,
            **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0
        self.warmup_steps = warmup_steps
        self.trim_in_test = trim_in_test
        self.fixed_length = fixed_length
        self.shuffle_before_cut = shuffle_before_cut

    def forward(self, x, v=None, mask=None, uu=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # uu: (N, C', P, P)
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()

        if self.enabled:
            if self._counter < self.warmup_steps:
                self._counter += 1
            else:
                if self.training or self.trim_in_test:
                    if self.fixed_length:
                        maxlen = self.fixed_length
                    else:
                        q = min(1, random.uniform(*self.target))
                        maxlen = torch.quantile(mask.type_as(x).sum(dim=-1), q).long()
                    if self.shuffle_before_cut:
                        rand = torch.rand_like(mask.type_as(x))
                        rand.masked_fill_(~mask, -1)
                        perm = rand.argsort(dim=-1, descending=True)  # (N, 1, P)
                        mask = torch.gather(mask, -1, perm)
                        x = torch.gather(x, -1, perm.expand_as(x))
                        if v is not None:
                            v = torch.gather(v, -1, perm.expand_as(v))
                        if uu is not None:
                            uu = torch.gather(uu, -2, perm.unsqueeze(-1).expand_as(uu))
                            uu = torch.gather(uu, -1, perm.unsqueeze(-2).expand_as(uu))
                else:
                    maxlen = mask.sum(dim=-1).max()
                maxlen = max(maxlen, 1)
                if maxlen < mask.size(-1):
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]
                    if v is not None:
                        v = v[:, :, :maxlen]
                    if uu is not None:
                        uu = uu[:, :, :maxlen, :maxlen]

        return x, v, mask, uu


class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation='gelu'):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend([
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x)
            x = x.permute(2, 0, 1).contiguous()
        # x: (seq_len, batch, embed_dim)
        return self.embed(x)


def pass_prev_attn_w(multiple_pair_embed, multiple_pair_embed_mode):
    """
        Whether mode needs previous attention weight \\
        to be passed as uu
    """
    return multiple_pair_embed and multiple_pair_embed_mode in [
        "independent_add_prev_attn_wts", "independent_add_prev_attn_wts_two_embeds"
    ]

def pemb_transforms_attn_w_logits(multiple_pair_embed, multiple_pair_embed_mode):
    """
        Whether mode transforms current attn weights using lv inputs
    """
    return multiple_pair_embed and multiple_pair_embed_mode in [
        "independent_transform_attn_wt_logits_one_embed"
    ]

def one_embed(multiple_pair_embed, multiple_pair_embed_mode):
    """
        Whether mode has only single embedding weights but applied
        multiple times still due to different inputs
    """
    return multiple_pair_embed and multiple_pair_embed_mode in [
        "independent_transform_attn_wt_logits_one_embed"
    ]

def two_embeds(multiple_pair_embed, multiple_pair_embed_mode):
    """
        Whether mode has different embedding weights for \\
        first vs. rest (True) or different for all (False)
    """
    return multiple_pair_embed and multiple_pair_embed_mode in [
        "independent_add_prev_attn_wts_two_embeds"
    ]


class Residual1x1Block(nn.Module):
    def __init__(self, in_dim, out_dim, activation='gelu', with_residual=False):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU() if activation=='gelu' else nn.ReLU()
        
        self.with_residual = with_residual
        if with_residual:
            if in_dim != out_dim:
                layers = [nn.Conv1d(in_dim, out_dim, kernel_size=1)]
                layers.append(nn.BatchNorm1d(out_dim))
                self.skip = nn.Sequential(*layers)
            else:
                self.skip = nn.Identity()

    def forward(self, x):
        if self.with_residual:
            identity = self.skip(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        if self.with_residual:
            return out + identity
        else:
            return out


def create_conv_sequence(
        input_dim, dims, n_embeds, normalize_input=True,
        activation='gelu', use_pre_activation_pair=False,
        with_residual=False,
    ) -> nn.ModuleList:
    embed_modules = []
    for _ in range(n_embeds):
        modules = []
        if normalize_input:
            modules.append(nn.BatchNorm1d(input_dim))
        prev_dim = input_dim
        for dim in dims:
            modules.append(Residual1x1Block(prev_dim, dim, activation=activation, with_residual=with_residual))
            prev_dim = dim
        if use_pre_activation_pair:
            last = modules[-1]
            if isinstance(last, Residual1x1Block):
                last.act = nn.Identity()
        embed_modules.append(nn.Sequential(*modules))
    return nn.ModuleList(embed_modules)


class PairEmbed(nn.Module):
    def __init__(
            self, pairwise_lv_dim, pairwise_input_dim, dims,
            remove_self_pair=False, use_pre_activation_pair=True, mode='sum',
            normalize_input=True, activation='gelu', eps=1e-8,
            for_onnx=False, multiple_pair_embed=False,
            multiple_pair_embed_mode="independent", num_layers=None,
            with_residual=False
        ):
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim
        self.pairwise_input_dim = pairwise_input_dim
        # NOTE: change here so that still treated as symmetric
        self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
        self.remove_self_pair = remove_self_pair
        self.mode = mode
        self.for_onnx = for_onnx
        self.pairwise_lv_fts = partial(pairwise_lv_fts, num_outputs=pairwise_lv_dim, eps=eps, for_onnx=for_onnx)
        self.out_dim = dims[-1]

        self.multiple_pair_embed = multiple_pair_embed
        self.multiple_pair_embed_mode = multiple_pair_embed_mode
        self.pass_prev_attn_w = pass_prev_attn_w(multiple_pair_embed, multiple_pair_embed_mode)
        if multiple_pair_embed:
            assert num_layers is not None
            assert multiple_pair_embed_mode in [
                # just run different convolutions from the same lv input
                "independent",
                # run convolutions from the same lv input, but add prev. attn weights
                "independent_add_prev_attn_wts",
                # run convolutions from the same lv input, add prev. attn weights,
                # but only have separate Us for 1st and rest of the layers
                "independent_add_prev_attn_wts_two_embeds",
                # run convolutions from the same lv input concat cur. attn weights,
                # return new attn weights. One embedding, but ran multiple times
                # with different inputs
                "independent_transform_attn_wt_logits_one_embed",
            ]

        # we're in a multiple mode but need only two embeds
        self.multiple_one_embed = one_embed(multiple_pair_embed, multiple_pair_embed_mode)
        self.multiple_two_embeds = two_embeds(multiple_pair_embed, multiple_pair_embed_mode)
        if self.multiple_one_embed:
            self.n_embeds = 1
        elif self.multiple_two_embeds:
            self.n_embeds = 2
        elif multiple_pair_embed:
            self.n_embeds = num_layers  # need num_layers embeds
        else:
            self.n_embeds = 1

        create_conv_sequence_p = partial(
            create_conv_sequence,
            normalize_input=normalize_input,
            activation=activation,
            use_pre_activation_pair=use_pre_activation_pair,
            with_residual=with_residual,
        )

        if self.mode == 'concat':
            if self.pass_prev_attn_w:
                # no prev input the first time
                self.embed = create_conv_sequence_p(pairwise_lv_dim, dims, 1)
            else:
                self.embed = create_conv_sequence_p(pairwise_lv_dim + pairwise_input_dim, dims, 1)
            # add the rest
            self.embed.extend(create_conv_sequence_p(pairwise_lv_dim + pairwise_input_dim, dims, self.n_embeds - 1))
        elif self.mode == 'sum':
            if pairwise_lv_dim > 0:
                self.embed = create_conv_sequence_p(pairwise_lv_dim, dims, self.n_embeds)
            if pairwise_input_dim > 0:
                if self.pass_prev_attn_w:
                    # no prev input the first time
                    self.fts_embed = nn.ModuleList([nn.Module()])
                else:
                    self.fts_embed = create_conv_sequence_p(pairwise_input_dim, dims, 1)
                # add the rest
                self.fts_embed.extend(create_conv_sequence_p(pairwise_input_dim, dims, self.n_embeds - 1))
        else:
            raise RuntimeError('`mode` can only be `sum` or `concat`')

    def forward(self, x, uu=None, block_index=0):
        # x: (batch, v_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)
        assert (x is not None or uu is not None)
        with torch.no_grad():
            if x is not None:
                batch_size, _, seq_len = x.size()
            else:
                batch_size, _, seq_len, _ = uu.size()
            if self.is_symmetric and not self.for_onnx:
                i, j = torch.tril_indices(seq_len, seq_len, offset=-1 if self.remove_self_pair else 0,
                                          device=(x if x is not None else uu).device)
                if x is not None:
                    x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
                    xi = x[:, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2)
                    xj = x[:, :, j, i]
                    x = self.pairwise_lv_fts(xi, xj)
                if uu is not None:
                    # (batch, dim, seq_len*(seq_len+1)/2)
                    uu = uu[:, :, i, j]
            else:
                if x is not None:
                    x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                    if self.remove_self_pair:
                        i = torch.arange(0, seq_len, device=x.device)
                        x[:, :, i, i] = 0
                    x = x.view(-1, self.pairwise_lv_dim, seq_len * seq_len)
                if uu is not None:
                    uu = uu.view(-1, self.pairwise_input_dim, seq_len * seq_len)
            if self.mode == 'concat':
                if x is None:
                    pair_fts = uu
                elif uu is None:
                    pair_fts = x
                else:
                    pair_fts = torch.cat((x, uu), dim=1)

        if self.multiple_one_embed:
            block_index = min(block_index, 0)  # 0 at max
        if self.multiple_two_embeds:
            block_index = min(block_index, 1)  # 1 at max
        assert block_index < self.n_embeds
        if block_index > 0:
            assert self.multiple_pair_embed

        if self.mode == 'concat':
            elements = self.embed[block_index](pair_fts)  # (batch, embed_dim, num_elements)
        elif self.mode == 'sum':
            if x is None:
                elements = self.fts_embed[block_index](uu)
            elif uu is None:
                elements = self.embed[block_index](x)
            else:
                elements = self.embed[block_index](x) + self.fts_embed[block_index](uu)

        if self.is_symmetric and not self.for_onnx:
            y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=elements.device)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)

        return y


class Block(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, ffn_ratio=4,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation='gelu',
                 scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
        )
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.c_attn = nn.Parameter(torch.ones(num_heads), requires_grad=True) if scale_heads else None
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            x_cls (Tensor, optional): class token input to the layer of shape `(1, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        if x_cls is not None:
            with torch.no_grad():
                # prepend one element for x_cls: -> (batch, 1+seq_len)
                padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
            # class attention: https://arxiv.org/pdf/2103.17239.pdf
            residual = x_cls
            u = torch.cat((x_cls, x), dim=0)  # (seq_len+1, batch, embed_dim)
            u = self.pre_attn_norm(u)
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask)[0]  # (1, batch, embed_dim)
        else:
            residual = x
            x = self.pre_attn_norm(x)
            padding_mask = padding_mask.float() * -1e9  # avoid deprecation
            x = self.attn(x, x, x, key_padding_mask=padding_mask,
                          attn_mask=attn_mask)[0]  # (seq_len, batch, embed_dim)

        if self.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            x = torch.einsum('tbhd,h->tbdh', x, self.c_attn)
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual

        return x


from .new_arch_modules import AlteredBlock


class InteractionTransformer(nn.Module):
    def __init__(self,
                 input_seq_len,
                 interactions_dim=4,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[8, 16, 8],
                 pair_embed_dims=[8, 8, 8],
                 num_heads=2,
                 num_layers=6,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[],
                 identical_attn_weights=False,
                 activation='gelu',
                 attention='linformer',
                 lin_proj_dim=128,
                 return_P_bars=False,
                 # misc
                 trim=True,
                 trim_mode="random_cutoff_in_train",
                 trim_mode_fixed_length=None,
                 trim_random_cutoff_range=(0.9, 1.02),
                 for_inference=False,
                 use_amp=False,
                 use_xla=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        if trim_mode in ["random_cutoff_in_train", "random_cutoff_always"]:
            self.trimmer = SequenceTrimmer(
                enabled=trim and not for_inference,
                target=trim_random_cutoff_range,
                trim_in_test=(trim_mode == "random_cutoff_always")
            )
        elif trim_mode in ["fixed_shuffle_always", "fixed_noshuffle_always"]:
            assert trim_mode_fixed_length is not None
            self.trimmer = SequenceTrimmer(
                enabled=trim,
                fixed_length=trim_mode_fixed_length,
                warmup_steps=0,
                trim_in_test=True,
                shuffle_before_cut=(trim_mode == "fixed_shuffle_always")
            )
            input_seq_len = min(input_seq_len, trim_mode_fixed_length)
        else:
            raise ValueError(f"trim_mode {trim_mode} not supported")

        self.for_inference = for_inference
        self.use_amp = use_amp
        self.use_xla = use_xla
        self.identical_attn_weights = identical_attn_weights
        self.return_P_bars = return_P_bars
        self.input_seq_len = input_seq_len

        self.interactions_dim = interactions_dim

        input_seq_len_2d = input_seq_len**2

        print(f"{input_seq_len_2d = }")

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else interactions_dim
        default_cfg = dict(
            embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
            dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
            add_bias_kv=False, activation=activation,
            attention=attention,
            input_seq_len=input_seq_len_2d,
            lin_proj_dim=lin_proj_dim,
            scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True
        )

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)
        _logger.info('cfg_block: %s' % str(cfg_block))

        cfg_cls_block = copy.deepcopy(default_cfg)
        # add class token
        cfg_cls_block.update(dict(input_seq_len=input_seq_len_2d+1))
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)
        _logger.info('cfg_cls_block: %s' % str(cfg_cls_block))

        self.pair_extra_dim = pair_extra_dim
        assert pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_extra_dim, pair_embed_dims + [interactions_dim],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference, multiple_pair_embed=False, num_layers=num_layers,
            activation=activation,
        )

        self.embed = Embed(interactions_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity()

        self.num_layers = num_layers
        self.num_cls_layers = num_cls_layers
        if identical_attn_weights:
            self.blocks = nn.ModuleList([AlteredBlock(**cfg_block) for _ in range(1)])
            self.cls_blocks = nn.ModuleList([AlteredBlock(**cfg_cls_block) for _ in range(1)])
        else:
            self.blocks = nn.ModuleList([AlteredBlock(**cfg_block) for _ in range(num_layers)])
            self.cls_blocks = nn.ModuleList([AlteredBlock(**cfg_cls_block) for _ in range(num_cls_layers)])

        self.norm = nn.LayerNorm(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs)
        # for onnx: uu (N, C', P, P), uu_idx=None

        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            _, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)  # (N, P)
            padding_mask = padding_mask.unsqueeze(2) & padding_mask.unsqueeze(1)  # (N, P, P)
            padding_mask = padding_mask.reshape((padding_mask.shape[0], -1))  # (N, P * P)

        with torch.autocast('xla' if self.use_xla else 'cuda', enabled=self.use_amp):

            v = v.masked_fill(~mask, 0)  # mask out particles that are padded
            x_pair = self.pair_embed(v, uu)  # (N, interactions_dim, P, P)
            x_pair = x_pair.reshape((x_pair.shape[0], self.interactions_dim, -1))  # (N, interactions_dim, P * P)

            # input embedding
            x = self.embed(x_pair)  # (P*P, N, interactions_dim)

            if self.return_P_bars:
                P_bars = []
            
            for i in range(self.num_layers):
                bi = 0 if self.identical_attn_weights else i
                if self.return_P_bars:
                    P_bars.append(
                        self.blocks[bi](x, x_cls=None, padding_mask=padding_mask, attn_mask=None, return_qk_attn_weight_logits=True)
                    )
                x = self.blocks[bi](x, x_cls=None, padding_mask=padding_mask, attn_mask=None)

            # extract class token
            cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # (1, N, C)
            for i in range(self.num_cls_layers):
                bi = 0 if self.identical_attn_weights else i
                cls_tokens = self.cls_blocks[bi](x, x_cls=cls_tokens, padding_mask=padding_mask, attn_mask=None)

            x_cls = self.norm(cls_tokens).squeeze(0)

            # fc
            if self.fc is None:
                return x_cls
            output = self.fc(x_cls)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            # print('output:\n', output)
            if self.return_P_bars:
                return output, P_bars
            else:
                return output


class ParticleTransformer(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128, 512, 128],
                 pair_embed_dims=[64, 64, 64],
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[],
                 activation='gelu',
                 multiple_pair_embed=False,
                 multiple_pair_embed_mode="independent",
                 identical_attn_weights=False,
                 pair_embed_with_residual=False,
                 return_qk_final_U_attn_weights=False,
                 add_sink_token=False,
                 uniformly_add_nblocks=None,
                 add_QK_U_alpha_in_every_block=False,
                 QK_U_alpha_mode="static", # accepts static, decode
                 # uses cls_block_params & num_cls_layers & identical_attn_weights
                 weighted_decode_every_layer=False,
                 weighted_decode_softmax_mode="softmax",  # accepts softmax, gumbel_softmax, gumbel_softmax_sample, sigmoid_every, gumbel_sigmoid_every
                 weighted_decode_normalize_sigmoids=True,
                 weighted_decode_warmup_steps=None,
                 weighted_decode_mode="ensemble",  # accepts ensemble, aggregate_x
                 # misc
                 trim=True,
                 trim_mode="random_cutoff_in_train",
                 trim_mode_fixed_length=None,
                 trim_random_cutoff_range=(0.9, 1.02),
                 for_inference=False,
                 use_amp=False,
                 use_xla=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        if trim_mode in ["random_cutoff_in_train", "random_cutoff_always"]:
            self.trimmer = SequenceTrimmer(
                enabled=trim and not for_inference,
                target=trim_random_cutoff_range,
                trim_in_test=(trim_mode == "random_cutoff_always")
            )
        elif trim_mode in ["fixed_shuffle_always", "fixed_noshuffle_always"]:
            assert trim_mode_fixed_length is not None
            self.trimmer = SequenceTrimmer(
                enabled=trim,
                fixed_length=trim_mode_fixed_length,
                warmup_steps=0,
                trim_in_test=True,
                shuffle_before_cut=(trim_mode == "fixed_shuffle_always")
            )
        else:
            raise ValueError(f"trim_mode {trim_mode} not supported")

        self.for_inference = for_inference
        self.use_amp = use_amp
        self.use_xla = use_xla
        self.identical_attn_weights = identical_attn_weights
        self.num_layers = num_layers
        self.num_cls_layers = num_cls_layers
        self.return_qk_final_U_attn_weights = return_qk_final_U_attn_weights
        self.uniformly_add_nblocks = uniformly_add_nblocks
        self.weighted_decode_every_layer = weighted_decode_every_layer
        self.weighted_decode_normalize_sigmoids = weighted_decode_normalize_sigmoids
        self.weighted_decode_warmup_steps = weighted_decode_warmup_steps
        self.weighted_decode_warmup_steps_done = 0
        self.add_QK_U_alpha_in_every_block = add_QK_U_alpha_in_every_block

        if weighted_decode_warmup_steps is not None:
            assert isinstance(weighted_decode_warmup_steps, int)

        if weighted_decode_every_layer:
            assert weighted_decode_softmax_mode in ["softmax", "gumbel_softmax", "gumbel_softmax_sample", "sigmoid_every", "gumbel_sigmoid_every"]
            self.weighted_decode_softmax_mode = weighted_decode_softmax_mode
            assert weighted_decode_mode in ["ensemble", "aggregate_x"]
            self.weighted_decode_mode = weighted_decode_mode

        if add_QK_U_alpha_in_every_block:
            assert QK_U_alpha_mode in ["static", "decode"]
            self.QK_U_alpha_mode = QK_U_alpha_mode

        if uniformly_add_nblocks is not None:
            assert identical_attn_weights  # other can be implemented if need be
            assert isinstance(uniformly_add_nblocks, int)

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=activation,
                           scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True)

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)
        _logger.info('cfg_block: %s' % str(cfg_block))

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)
        _logger.info('cfg_cls_block: %s' % str(cfg_cls_block))

        self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity()
        
        self.pair_extra_dim = pair_extra_dim
        self.multiple_pair_embed = multiple_pair_embed
        self.multiple_pair_embed_mode = multiple_pair_embed_mode
        self.pemb_needs_prev_attn_w = pass_prev_attn_w(multiple_pair_embed, multiple_pair_embed_mode)
        self.pemb_transforms_attn_w_logits = pemb_transforms_attn_w_logits(multiple_pair_embed, multiple_pair_embed_mode)
        if self.pemb_needs_prev_attn_w or self.pemb_transforms_attn_w_logits:
            assert pair_extra_dim == 0, f"{pair_extra_dim = } is not supported with {multiple_pair_embed_mode = }"
            pair_extra_dim = cfg_block['num_heads']
        self.pair_embed = PairEmbed(
                pair_input_dim, pair_extra_dim, pair_embed_dims + [cfg_block['num_heads']],
                remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
                for_onnx=for_inference, activation=activation,
                multiple_pair_embed=multiple_pair_embed,
                multiple_pair_embed_mode=multiple_pair_embed_mode,
                num_layers=num_layers,
                mode="concat" if self.pemb_needs_prev_attn_w or self.pemb_transforms_attn_w_logits else "sum",
                with_residual=pair_embed_with_residual
            ) if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None
        
        if add_QK_U_alpha_in_every_block:
            if QK_U_alpha_mode == "static":
                additional_layers = 0 if uniformly_add_nblocks is None else uniformly_add_nblocks
                self.qk_u_encoder_alphas = nn.Parameter(torch.zeros(num_layers + additional_layers, cfg_block['num_heads']), requires_grad=True)
                trunc_normal_(self.qk_u_encoder_alphas, std=.02)
            elif QK_U_alpha_mode == "decode":
                # blocks
                if identical_attn_weights:
                    self.QK_U_alpha_blocks = nn.ModuleList([AlteredBlock(**cfg_cls_block) for _ in range(1)])
                else:
                    self.QK_U_alpha_blocks = nn.ModuleList([AlteredBlock(**cfg_cls_block) for _ in range(num_cls_layers)])
                # norm
                self.QK_U_alpha_norm = nn.LayerNorm(embed_dim)
                # decoding token
                self.QK_U_alpha_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
                trunc_normal_(self.QK_U_alpha_token, std=.02)
                # FC to get alpha value for each head
                assert fc_params is not None  # can be implemented if need be
                fcs = []
                prev_dim = embed_dim
                for dim, drop_rate in fc_params:
                    fcs.append(nn.Sequential(nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(drop_rate)))
                    prev_dim = dim
                fcs.append(nn.Linear(prev_dim, cfg_block['num_heads']))
                self.QK_U_alpha_fc = nn.Sequential(*fcs)
            else:
                raise ValueError(f"Unknown {QK_U_alpha_mode = }")

        if identical_attn_weights:
            self.blocks = nn.ModuleList([AlteredBlock(**cfg_block) for _ in range(1)])
            self.cls_blocks = nn.ModuleList([AlteredBlock(**cfg_cls_block) for _ in range(1)])
            if weighted_decode_every_layer:
                self.weighting_blocks = nn.ModuleList([AlteredBlock(**cfg_cls_block) for _ in range(1)])
        else:
            self.blocks = nn.ModuleList([AlteredBlock(**cfg_block) for _ in range(num_layers)])
            self.cls_blocks = nn.ModuleList([AlteredBlock(**cfg_cls_block) for _ in range(num_cls_layers)])
            if weighted_decode_every_layer:
                self.weighting_blocks = nn.ModuleList([AlteredBlock(**cfg_cls_block) for _ in range(num_cls_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        if weighted_decode_every_layer:
            self.weighting_norm = nn.LayerNorm(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        if weighted_decode_every_layer:
            assert fc_params is not None  # can be implemented if need be
            fcs = []
            prev_dim = embed_dim
            for dim, drop_rate in fc_params:
                fcs.append(nn.Sequential(nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(drop_rate)))
                prev_dim = dim
            fcs.append(nn.Linear(prev_dim, 1))
            self.weighting_fc = nn.Sequential(*fcs)

        # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=.02)
        if weighted_decode_every_layer:
            self.weighting_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
            trunc_normal_(self.weighting_token, std=.02)

        self.sink_token = None
        if add_sink_token:
            self.sink_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
            trunc_normal_(self.sink_token, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', 'sink_token', 'qk_u_encoder_alphas', }

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs)
        # for onnx: uu (N, C', P, P), uu_idx=None

        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)  # (N, P)

        with torch.autocast('xla' if self.use_xla else 'cuda', enabled=self.use_amp):
            # input embedding
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)

            # add sink token if needed
            if self.sink_token is not None:
                with torch.no_grad():
                    padding_mask = torch.cat((  # sink is not padding, add zeros
                        torch.zeros_like(padding_mask[:, :1]), padding_mask
                    ), dim=1)
                # duplicate sink token and add it
                sink_tokens = self.sink_token.expand(1, x.size(1), -1)  # (1, N, C)
                x = torch.cat((sink_tokens, x), dim=0)  # (P + 1, N, C)
                # prepend zeros to v to accomovate the sink token
                sink_tokens_v = torch.zeros_like(v[:, :, :1])  # (N, 4, 1)
                v = torch.cat([sink_tokens_v, v], dim=2)  # (N, 4, P + 1)

            if self.return_qk_final_U_attn_weights:
                attn_weights_list = []
                qk_attn_weights_list = []

            add_attn_mask = (v is not None or uu is not None) and self.pair_embed is not None
            pair_embeds, prev_attn_weight = None, None
            num_blocks = self.num_layers
            if self.uniformly_add_nblocks is not None:
                num_blocks += torch.randint(self.uniformly_add_nblocks + 1, size=(1,))[0].item()

            def decode(x_inp, token, blocks, norm):
                # extract class token
                cls_tokens = token.expand(1, x_inp.size(1), -1)  # (1, N, C)
                for cbi in range(self.num_cls_layers):
                    if self.identical_attn_weights:
                        cls_block = blocks[0]
                    else:
                        cls_block = blocks[cbi]
                    cls_tokens = cls_block(x_inp, x_cls=cls_tokens, padding_mask=padding_mask)
                x_cls = norm(cls_tokens).squeeze(0)
                return x_cls
            
            if self.weighted_decode_every_layer:
                x_weights = []
                outputs = []
            
            for bi in range(num_blocks):

                if self.identical_attn_weights:
                    block = self.blocks[0]
                else:
                    block = self.blocks[bi]

                qk_u_alpha = None
                if self.add_QK_U_alpha_in_every_block:
                    if self.QK_U_alpha_mode == "static":
                        qk_u_alpha = self.qk_u_encoder_alphas[bi].unsqueeze(0)  # (1, num_heads)
                    elif self.QK_U_alpha_mode == "decode":
                        x_QK_U_alpha = decode(x, self.QK_U_alpha_token, self.QK_U_alpha_blocks, self.QK_U_alpha_norm)
                        x_QK_U_alpha = self.QK_U_alpha_fc(x_QK_U_alpha)  # (B, num_heads)
                    else:
                        raise ValueError(f"Unknown {self.QK_U_alpha_mode = }")

                if self.pemb_transforms_attn_w_logits:
                    assert not self.return_qk_final_U_attn_weights # can be implemented if need be
                    # completely different logic here
                    qk_attn_weight_logits = block(x, x_cls=None, padding_mask=None, attn_mask=None, return_qk_attn_weight_logits=True)
                    assert uu is None  # not supported here
                    pair_embeds = self.pair_embed(v, qk_attn_weight_logits, block_index=bi)
                    x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=None, use_qk_attn_weight_logits=pair_embeds, qk_u_alpha=qk_u_alpha)
                    continue

                attn_mask = None
                if add_attn_mask:
                    if pair_embeds is None or self.multiple_pair_embed:
                        # recalculate pair_embeds if:
                        #     - self.multiple_pair_embed is True
                        #     - it was not yet calculated (for single U case)
                        block_index = bi if self.multiple_pair_embed else 0

                        uu_aug = uu
                        if self.pemb_needs_prev_attn_w:
                            assert uu_aug is None
                            uu_aug = prev_attn_weight  # None for the first iteration

                        pair_embeds = self.pair_embed(v, uu_aug, block_index=block_index)
                    attn_mask = pair_embeds.view(-1, v.size(-1), v.size(-1))  # (N*num_heads, P, P)

                if self.return_qk_final_U_attn_weights:
                    qk_attn_weight_logits = block(x, x_cls=None, padding_mask=None, attn_mask=None, return_qk_attn_weight_logits=True)
                    qk_attn_weights_list.append(qk_attn_weight_logits.detach().cpu())
                
                if self.pemb_needs_prev_attn_w:
                    x, prev_attn_weight = block(
                        x, x_cls=None, padding_mask=padding_mask,
                        attn_mask=attn_mask, return_final_attn_weight=True,
                        qk_u_alpha=qk_u_alpha
                    )
                    if self.return_qk_final_U_attn_weights:
                        attn_weights_list.append(attn_weights.detach().cpu())
                else:
                    if self.return_qk_final_U_attn_weights:
                        x, attn_weights = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask, return_final_attn_weight=True, qk_u_alpha=qk_u_alpha)
                        attn_weights_list.append(attn_weights.detach().cpu())
                    else:
                        x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask, qk_u_alpha=qk_u_alpha)
                
                if self.weighted_decode_every_layer:
                    x_weight = decode(x, self.weighting_token, self.weighting_blocks, self.weighting_norm)
                    x_weight = self.weighting_fc(x_weight)  # (B, 1)
                    x_weights.append(x_weight)
                    if self.weighted_decode_mode == "ensemble":
                        x_cls = decode(x, self.cls_token, self.cls_blocks, self.norm)
                        output = self.fc(x_cls)  # (B, num_classes)
                        outputs.append(output)
                    elif self.weighted_decode_mode == "aggregate_x":
                        outputs.append(x)  # (n_particles, B, ch_size)

            if self.weighted_decode_every_layer:
                x_weights = torch.cat(x_weights, dim=1)  # (B, num_blocks)
                if self.return_qk_final_U_attn_weights:
                    x_weights_unnorm = x_weights
                if self.weighted_decode_softmax_mode == "softmax":
                    x_weights = torch.softmax(x_weights, dim=1)  # (B, num_blocks)
                elif self.weighted_decode_softmax_mode == "sigmoid_every":
                    x_weights = torch.sigmoid(x_weights)                            # (B, num_blocks)
                    if self.weighted_decode_normalize_sigmoids:
                        x_weights = x_weights / x_weights.sum(dim=1, keepdim=True)  # (B, num_blocks)
                elif self.weighted_decode_softmax_mode == "gumbel_sigmoid_every":
                    sig = torch.sigmoid(x_weights)                # (B, num_blocks)
                    if self.weighted_decode_normalize_sigmoids:
                        sig = sig / sig.sum(dim=1, keepdim=True)  # (B, num_blocks)
                    hard = (sig > 0.5).float()                    # (B, num_blocks) (0 or 1)
                    x_weights = sig + (hard - sig).detach()       # (B, num_blocks)
                elif self.weighted_decode_softmax_mode in ["gumbel_softmax", "gumbel_softmax_sample"]:
                    x_weights_soft = F.softmax(x_weights, dim=1)  # (B, num_blocks)
                    if self.weighted_decode_softmax_mode == "gumbel_softmax_sample" and self.training:
                        idx = torch.multinomial(x_weights_soft, num_samples=1).squeeze(1)
                    else:
                        _, idx = x_weights_soft.max(dim=1)
                    x_weights_hard = F.one_hot(idx, num_blocks).float()  # (B, num_blocks)
                    # this way loss is calculated based on x_hard,
                    # but backward-prop uses gradient from x_soft
                    x_weights = (x_weights_hard - x_weights_soft).detach() + x_weights_soft
                if self.weighted_decode_warmup_steps is not None:
                    warmup_flag = torch.as_tensor(
                        self.weighted_decode_warmup_steps_done < self.weighted_decode_warmup_steps and self.training,
                        dtype=x_weights.dtype, device=x_weights.device,
                    )  # 1.0 during warmup, 0.0 after
                    if self.training:
                        self.weighted_decode_warmup_steps_done += 1
                    x_weights_hard_last = torch.zeros_like(x_weights)
                    x_weights_hard_last[:, -1] = 1
                    x_weights = x_weights * (1 - warmup_flag) + x_weights_hard_last * warmup_flag
                if self.weighted_decode_mode == "ensemble":
                    outputs = torch.stack(outputs)  # (num_blocks, B, num_classes)
                    output = torch.einsum('bn,nbc->bc', x_weights, outputs)  # (B, num_classes)
                elif self.weighted_decode_mode == "aggregate_x":
                    outputs = torch.stack(outputs)  # (num_blocks, n_particles, B, ch_size)
                    x_agg = torch.einsum('bn,npbc->pbc', x_weights, outputs)  # (n_particles, B, ch_size)
                    x_cls = decode(x_agg, self.cls_token, self.cls_blocks, self.norm)
                    output = self.fc(x_cls)  # (B, num_classes)
            else:
                x_cls = decode(x, self.cls_token, self.cls_blocks, self.norm)
                # fc
                if self.fc is None:
                    return x_cls
                output = self.fc(x_cls)

            if self.for_inference:
                output = torch.softmax(output, dim=1)

            if self.return_qk_final_U_attn_weights:
                if self.weighted_decode_every_layer:
                    return output, qk_attn_weights_list, attn_weights_list, attn_mask, x_weights_unnorm, x_weights, outputs
                else:
                    return output, qk_attn_weights_list, attn_weights_list, attn_mask
            return output


class ParticleTransformerMultipleRuns(nn.Module):
    def __init__(self,
                 input_dim,
                 num_runs=10,
                 **kwargs) -> None:
        super().__init__()

        self.num_runs = num_runs

        self.pt = ParticleTransformer(input_dim, **kwargs)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return self.pt.no_weight_decay()

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
        scores_list = []
        for k in range(self.num_runs):
            scores = self.pt(
                x, v=v, mask=mask, uu=uu, uu_idx=uu_idx
            )
            scores_list.append(scores)
        return torch.stack(scores_list).mean(axis=0)


class ParticleTransformerWithInverter(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128, 512, 128],
                 pair_embed_dims=[64, 64, 64],
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[],
                 activation='gelu',
                 # misc
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 use_xla=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.use_amp = use_amp
        self.use_xla = use_xla
        self.inverter_alpha = 10

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=activation,
                           scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True)

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)
        _logger.info('cfg_block: %s' % str(cfg_block))

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)
        _logger.info('cfg_cls_block: %s' % str(cfg_cls_block))

        self.pair_extra_dim = pair_extra_dim
        self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity()
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_extra_dim, pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference) if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None
        self.blocks = nn.ModuleList([AlteredBlock(**cfg_block) for _ in range(num_layers)])
        self.cls_blocks = nn.ModuleList([AlteredBlock(**cfg_cls_block) for _ in range(num_cls_layers)])
        self.cls_inverter_blocks = nn.ModuleList([AlteredBlock(**cfg_cls_block) for _ in range(num_cls_layers)])
        self.norm = nn.LayerNorm(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)

            fcs_inverter = []
            in_dim = embed_dim
            for out_dim, drop_rate in fc_params:
                fcs_inverter.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs_inverter.append(nn.Linear(in_dim, 1))  # only one output
            self.fc_inverter = nn.Sequential(*fcs)
        else:
            self.fc = None
            self.fc_inverter = None

        # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=.02)

        self.cls_inverter_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_inverter_token, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs)
        # for onnx: uu (N, C', P, P), uu_idx=None

        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)  # (N, P)

        with torch.autocast('xla' if self.use_xla else 'cuda', enabled=self.use_amp):
            # input embedding
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)
            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, uu).view(-1, v.size(-1), v.size(-1))  # (N*num_heads, P, P)

            # transform
            for block in self.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

            # extract class token
            cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # (1, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)

            x_cls = self.norm(cls_tokens).squeeze(0)

            # fc
            if self.fc is None:
                return x_cls
            
            output = self.fc(x_cls)
            
            # extract class token for inverter
            cls_inverter_tokens = self.cls_inverter_token.expand(1, x.size(1), -1)  # (1, N, C)
            for block in self.cls_inverter_blocks:
                cls_inverter_tokens = block(x, x_cls=cls_inverter_tokens, padding_mask=padding_mask)

            x_inverter_cls = self.norm(cls_inverter_tokens).squeeze(0)

            inverter_output = self.fc_inverter(x_inverter_cls)
            inverter_output = 2. * torch.sigmoid(self.inverter_alpha * inverter_output) - 1  # (B, 1)

            # (B, 2) * (B, 1) = (B, 2)
            output = output * inverter_output
            
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            # print('output:\n', output)
            return output


class ParticleTransformerTagger(nn.Module):

    def __init__(self,
                 pf_input_dim,
                 sv_input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128, 512, 128],
                 pair_embed_dims=[64, 64, 64],
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[],
                 activation='gelu',
                 # misc
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 use_xla=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.use_amp = use_amp
        self.use_xla = use_xla

        self.pf_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.sv_trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        self.pf_embed = Embed(pf_input_dim, embed_dims, activation=activation)
        self.sv_embed = Embed(sv_input_dim, embed_dims, activation=activation)

        self.part = ParticleTransformer(input_dim=embed_dims[-1],
                                        num_classes=num_classes,
                                        # network configurations
                                        pair_input_dim=pair_input_dim,
                                        pair_extra_dim=pair_extra_dim,
                                        remove_self_pair=remove_self_pair,
                                        use_pre_activation_pair=use_pre_activation_pair,
                                        embed_dims=[],
                                        pair_embed_dims=pair_embed_dims,
                                        num_heads=num_heads,
                                        num_layers=num_layers,
                                        num_cls_layers=num_cls_layers,
                                        block_params=block_params,
                                        cls_block_params=cls_block_params,
                                        fc_params=fc_params,
                                        activation=activation,
                                        # misc
                                        trim=False,
                                        for_inference=for_inference,
                                        use_amp=use_amp,
                                        use_xla=use_xla)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'part.cls_token', }

    def forward(self, pf_x, pf_v=None, pf_mask=None, sv_x=None, sv_v=None, sv_mask=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0

        with torch.no_grad():
            pf_x, pf_v, pf_mask, _ = self.pf_trimmer(pf_x, pf_v, pf_mask)
            sv_x, sv_v, sv_mask, _ = self.sv_trimmer(sv_x, sv_v, sv_mask)
            v = torch.cat([pf_v, sv_v], dim=2)
            mask = torch.cat([pf_mask, sv_mask], dim=2)

        with torch.autocast('xla' if self.use_xla else 'cuda', enabled=self.use_amp):
            pf_x = self.pf_embed(pf_x)  # after embed: (seq_len, batch, embed_dim)
            sv_x = self.sv_embed(sv_x)
            x = torch.cat([pf_x, sv_x], dim=0)

            return self.part(x, v, mask)


class ParticleTransformerTaggerWithExtraPairFeatures(nn.Module):

    def __init__(self,
                 pf_input_dim,
                 sv_input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128, 512, 128],
                 pair_embed_dims=[64, 64, 64],
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[],
                 activation='gelu',
                 # misc
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 use_xla=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.use_amp = use_amp
        self.use_xla = use_xla
        self.for_inference = for_inference

        self.pf_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.sv_trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        self.pf_embed = Embed(pf_input_dim, embed_dims, activation=activation)
        self.sv_embed = Embed(sv_input_dim, embed_dims, activation=activation)

        self.part = ParticleTransformer(input_dim=embed_dims[-1],
                                        num_classes=num_classes,
                                        # network configurations
                                        pair_input_dim=pair_input_dim,
                                        pair_extra_dim=pair_extra_dim,
                                        remove_self_pair=remove_self_pair,
                                        use_pre_activation_pair=use_pre_activation_pair,
                                        embed_dims=[],
                                        pair_embed_dims=pair_embed_dims,
                                        num_heads=num_heads,
                                        num_layers=num_layers,
                                        num_cls_layers=num_cls_layers,
                                        block_params=block_params,
                                        cls_block_params=cls_block_params,
                                        fc_params=fc_params,
                                        activation=activation,
                                        # misc
                                        trim=False,
                                        for_inference=for_inference,
                                        use_amp=use_amp,
                                        use_xla=use_xla)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'part.cls_token', }

    def forward(self, pf_x, pf_v=None, pf_mask=None, sv_x=None, sv_v=None, sv_mask=None, pf_uu=None, pf_uu_idx=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0

        with torch.no_grad():
            if not self.for_inference:
                if pf_uu_idx is not None:
                    pf_uu = build_sparse_tensor(pf_uu, pf_uu_idx, pf_x.size(-1))

            pf_x, pf_v, pf_mask, pf_uu = self.pf_trimmer(pf_x, pf_v, pf_mask, pf_uu)
            sv_x, sv_v, sv_mask, _ = self.sv_trimmer(sv_x, sv_v, sv_mask)
            v = torch.cat([pf_v, sv_v], dim=2)
            mask = torch.cat([pf_mask, sv_mask], dim=2)
            uu = torch.zeros(v.size(0), pf_uu.size(1), v.size(2), v.size(2), dtype=v.dtype, device=v.device)
            uu[:, :, :pf_x.size(2), :pf_x.size(2)] = pf_uu

        with torch.autocast('xla' if self.use_xla else 'cuda', enabled=self.use_amp):
            pf_x = self.pf_embed(pf_x)  # after embed: (seq_len, batch, embed_dim)
            sv_x = self.sv_embed(sv_x)
            x = torch.cat([pf_x, sv_x], dim=0)

            return self.part(x, v, mask, uu)
