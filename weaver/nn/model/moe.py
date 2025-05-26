# https://github.com/lucidrains/st-moe-pytorch

from __future__ import annotations

from functools import partial
from collections import namedtuple
from typing import Tuple, List

import torch
from torch.nn import Module, ModuleList
from torch import nn, einsum
import torch.nn.functional as F

from beartype import beartype

import einx
from einops import rearrange, repeat, reduce, pack, unpack

from colt5_attention import topk as maybe_differentiable_topk

# constants

MIN_EXPERT_CAPACITY = 4

MixtureOfExpertsReturn = namedtuple('MixtureOfExpertsReturn', [
    'outputs',
    'total_aux_loss'
])


# ==================================== HELPER FUNCTIONS ====================================
def exists(val):
    return val is not None


def default(val, default):
    if exists(val):
        return val
    return default() if callable(default) else default


def divisible_by(num, den):
    return (num % den) == 0


def chunk_num(num, chunks):
    """
    Get sizes of each chunk
    chunk_num(10, 3) -> [4, 3, 3]
    """
    num_per_chunk, remainder = divmod(num, chunks)
    out = []
    for i in range(chunks):
        n = num_per_chunk
        out.append(n + int(i < remainder))
    return out


def pack_one(t, pattern):
    """
    Convenience wrapper around `einops.pack` for a single tensor.
    Args:
        t (Tensor): input tensor to be packed.
        pattern (str): einops pattern describing the packing transformation.
    Returns:
        Tuple[Tensor, Any]: the packed tensor and metadata for unpacking.
    """
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    """
    Convenience wrapper around `einops.unpack` for a single tensor.
    """
    return unpack(t, ps, pattern)[0]


def cast_tuple(el, len=1):
    """
    Check if 'el' is a tuple, and if not, cast it to a tuple of length 'len' containing 'el'
    """
    return el if isinstance(el, tuple) else ((el,) * len)


def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))


# ============================= TENSOR-RELATED HELPER FUNCTIONS =======================
def cumsum_exclusive(t, dim=-3):
    """
    Get cumulative sum of all elements BEFORE the current one. Works for more than 1 dim
    """
    assert dim < 0
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return F.pad(t, (*pre_padding, 1, -1)).cumsum(dim=dim)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def safe_one_hot(indexes, max_length):
    """
    PyTorch one hot throws an error if there are out of bound indices.
    Tensorflow, in contrast, does not
    """
    max_index = indexes.max() + 1
    one_hot_classes = max(max_index + 1, max_length)
    return F.one_hot(indexes, one_hot_classes)[..., :max_length]


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.gamma * self.scale


class GEGLU(Module):
    """
    GEGLU works better than GELU, according to the authors
    """

    def __init__(
            self,
            dim,
            mult_bias=True
    ):
        super().__init__()
        self.mult_bias = nn.Parameter(torch.ones(dim)) if mult_bias else 1.

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x * self.mult_bias


class Expert(Module):
    """
    An 'Expert' feedforward network
    """

    def __init__(self,
                 embed_dim,
                 ffn_dim,
                 activation='gelu',
                 activation_dropout=0.1,
                 prenorm=False,
                 scale_fc=True):
        super().__init__()

        self.pre_fc_norm = RMSNorm(embed_dim) if prenorm else None
        self.fc1 = nn.Linear(embed_dim, ffn_dim * 2)
        self.act = GEGLU(ffn_dim) if activation == 'gelu' else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

        self.apply(self.init_)

    def init_(self, module):
        """
        Uniform init values
        """
        if isinstance(module, nn.Linear):
            dim = module.weight.shape[0]
            std = dim ** -0.5

            module.weight.data.uniform_(-std, std)
            module.bias.data.uniform_(-std, std)

    def forward(self, x):
        if self.pre_fc_norm is not None:
            x = self.pre_fc_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        return x


class Experts(Module):
    def __init__(
            self,
            experts,
            allow_var_seq_len=False  # whether to handle variable sequence length
    ):
        super().__init__()
        self.num_experts = len(experts)
        self.experts = ModuleList(experts)

        self.allow_var_seq_len = allow_var_seq_len

    def forward(self, x):
        """
        einops notation:
        b - batch
        e - experts
        n - sequence (number of tokens per expert)
        d - feature dimension
        """

        shape, num_experts = x.shape, self.num_experts
        # set the expert dimension first, so torch can fuse calls-per-expert
        x = rearrange(x, 'b e n d -> e b n d')
        outs = []

        # call each expert sequentially
        for expert, expert_input in zip(self.experts, x):
            out = expert(expert_input)
            outs.append(out)

        # gather outputs from all experts
        if len(outs) > 0:
            outs = torch.stack(outs)
        else:
            outs = torch.empty_like(x, requires_grad=self.training)

        # split the batch dimension back first
        outs = rearrange(outs, 'e b n d -> b e n d')
        assert outs.shape == shape
        return outs


# the below code is almost all transcribed from the official tensorflow version, from which the papers are written
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/moe.py

# gating network

class TopNGating(Module):
    @beartype
    def __init__(
            self,
            dim,
            num_gates,
            eps=1e-9,
            top_n=2,
            threshold_train: float | Tuple[float, ...] = 0.2,
            threshold_eval: float | Tuple[float, ...] = 0.2,
            capacity_factor_train=1.25,
            capacity_factor_eval=2.,
            straight_through_dispatch_tensor=True,
            differentiable_topk=False,
            differentiable_topk_fused=True
    ):
        super().__init__()
        self.eps = eps
        self.num_gates = num_gates
        self.to_gates = nn.Linear(dim, num_gates, bias=False)

        self.differentiable_topk = differentiable_topk

        self.topk = partial(
            maybe_differentiable_topk,
            non_differentiable=not differentiable_topk,
            fused=differentiable_topk_fused  # use triton fused coordinate descent if possible by default
        )

        assert top_n >= 2, 'must be 2 or more experts'
        self.top_n = top_n
        top_n_minus_1 = top_n - 1

        threshold_train = cast_tuple(threshold_train, top_n_minus_1)
        threshold_eval = cast_tuple(threshold_eval, top_n_minus_1)

        assert len(threshold_train) == len(threshold_eval) == top_n_minus_1

        self.register_buffer('threshold_train', torch.tensor([eps, *threshold_train]))
        self.register_buffer('threshold_eval', torch.tensor([eps, *threshold_eval]))

        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

        self.straight_through_dispatch_tensor = straight_through_dispatch_tensor
        self.register_buffer('zero', torch.zeros((1,)), persistent=False)

    def forward(
            self,
            x,
            noise_gates=False,
            noise_mult=1.
    ):
        """
        einstein notation:

        b - batch
        n - sequence
        e - experts
        k - top-n experts
        c - capacity
        """

        # x arrives as [*, b, n, dim]
        *_, b, group_size, dim, dtype, top_n, num_gates, eps = *x.shape, x.dtype, self.top_n, self.num_gates, self.eps

        # threshold, capacity depending on training or eval
        suffix = 'train' if self.training else 'eval'
        threshold = getattr(self, f'threshold_{suffix}')
        capacity_factor = getattr(self, f'capacity_factor_{suffix}')

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes
        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        # gate logits and gates
        gate_logits = self.to_gates(x)
        maybe_noised_gate_logits = gate_logits
        if noise_gates:
            noise = gumbel_noise(maybe_noised_gate_logits)
            maybe_noised_gate_logits = maybe_noised_gate_logits + noise * noise_mult

        # find top N experts per token
        raw_gates = maybe_noised_gate_logits.softmax(dim=-1)  # [b, n, e]
        topk_return = self.topk(raw_gates, k=top_n)  # [b, n, k]
        gate_indices = topk_return.indices
        if self.differentiable_topk:
            # allow for differentiable topk using coordinate descent
            # used successfully for routing from CoLT5 paper https://github.com/lucidrains/CoLT5-attention
            gates = topk_return.coor_descent_values
        else:
            gates = topk_return.values

        # move the top-n dimension to be first, so experts are easier to batch
        gates = rearrange(gates, '... k -> k ...')  # [k, b, n]
        gate_indices = rearrange(gate_indices, '... k -> k ...')  # [k, b, n]

        # masks
        one_hot_gate_indices = F.one_hot(gate_indices,
                                         num_gates)  # [k, b, n, e], where k - top-Kth expert, e - 1-hot vector with '1' at the index of top-Kth expert and 0s elsewhere
        mask = one_hot_gate_indices.float()
        mask_1 = mask[0]  # needed for balancing loss

        # re-normalize top-k (because some values were masked)
        denom = reduce(gates, 'k ... -> 1 ...', 'sum').clamp(min=eps)
        gates = gates / denom

        # best performing policy was to route to the second expert, with probability of min(1., score / threshold), where score = gate2 / (gate1 + gate2)
        # optimal threshold was ~ 0.2
        # generalized to more than 2 experts
        probs = torch.zeros_like(gates).uniform_(0., 1.)
        should_route = probs < einx.divide('k b n, k -> k b n', gates, threshold.clamp(min=eps))

        # tokens should always be routed to first expert
        # threshold for first expert already set to very small number, but just in case
        should_route[0, ...] = True
        mask *= rearrange(should_route.float(), '... -> ... 1')
        mask_cumsum = cumsum_exclusive(mask, dim=-2)  # along sequence dimension

        # compute assignment to experts - (batch, seq, experts)
        # This is the position within the expert's mini-batch for this sequence
        positions = []
        prev_expert_count = 0.

        # Iterate over top_n positions
        for n in range(self.top_n):
            # Find out top-Nth experts for each token position
            position_in_expert = (mask_cumsum[n] + prev_expert_count) * mask[n]

            # Remove the elements that don't fit (never applicable for the top-1 expert)
            mask[n] *= (position_in_expert < expert_capacity_f).float()

            # Find out how many tokens so far have come to each expert
            prev_expert_count = reduce(mask[n], '... n e -> ... 1 e', 'sum') + prev_expert_count

            # compute how many tokens have already been routed to every expert
            position_in_expert = reduce(position_in_expert, '... n e -> ... n', 'sum')  # [b, n]
            positions.append(position_in_expert)

        positions = torch.stack(positions)

        # mostly ones, but zeros where something didn't fit
        mask_flat = reduce(mask, '... n e -> ... n', 'sum')  # [k, b, n]

        # (batch, sequence, experts, expert_capacity)
        combine_tensor = einx.multiply(
            'k b n, k b n, k b n e, k b n c -> k b n e c',
            gates,
            mask_flat,
            one_hot_gate_indices,
            safe_one_hot(positions.long(), expert_capacity)
        )

        combine_tensor = reduce(combine_tensor, 'k b n e c -> b n e c', 'sum')
        dispatch_tensor = combine_tensor.bool().type(dtype)

        if self.straight_through_dispatch_tensor:
            dispatch_tensor = dispatch_tensor + combine_tensor - combine_tensor.detach()

        # balance losses - (batch, experts)
        # We want to equalize the fraction of the batch assigned to each expert

        if self.training:
            density_1 = reduce(mask_1, '... n e -> ... e', 'mean')
            density_1_proxy = reduce(raw_gates, '... n e -> ... e',
                                     'mean')  # Something continuous that is correlated with what we want to equalize.

            balance_loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)
        else:
            balance_loss = self.zero

        # calculate the router z-loss proposed in paper

        if self.training:
            router_z_loss = torch.logsumexp(gate_logits, dim=-1)
            router_z_loss = torch.square(router_z_loss)
            router_z_loss = router_z_loss.mean()
        else:
            router_z_loss = self.zero

        return dispatch_tensor, combine_tensor, balance_loss, router_z_loss


# plain mixture of experts

class MoE(Module):
    @beartype
    def __init__(self,
                 embed_dim,
                 ffn_dim,
                 num_experts=16,
                 threshold_train=0.2,
                 threshold_eval=0.2,
                 capacity_factor_train=1.25,
                 capacity_factor_eval=2.,
                 gating_top_n=2,
                 balance_loss_coef=1e-2,
                 router_z_loss_coef=1e-3,
                 m=2,
                 num_shared_experts=1,
                 straight_through_dispatch_tensor=True,
                 differentiable_topk=False,
                 differentiable_topk_fused=True,
                 allow_var_seq_len=False,
                 expert_scale_fc=False,
                 expert_activation_dropout=0.1
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim // m
        self.num_experts = num_experts * m - num_shared_experts
        self.top_n = gating_top_n * m - num_shared_experts

        self.gate = TopNGating(
            embed_dim,
            top_n=self.top_n,
            num_gates=self.num_experts,
            straight_through_dispatch_tensor=straight_through_dispatch_tensor,
            differentiable_topk=differentiable_topk,
            differentiable_topk_fused=differentiable_topk_fused,
            threshold_train=threshold_train,
            threshold_eval=threshold_eval,
            capacity_factor_train=capacity_factor_train,
            capacity_factor_eval=capacity_factor_eval
        )

        experts = [Expert(embed_dim=self.embed_dim,
                          ffn_dim=self.ffn_dim,
                          scale_fc=expert_scale_fc,
                          activation_dropout=expert_activation_dropout) for _ in range(self.num_experts)]

        self.experts = Experts(
            experts,
            allow_var_seq_len=allow_var_seq_len
        )

        self.register_buffer('expert_output_multi',
                             torch.tensor(self.num_experts / (self.num_experts + num_shared_experts)).float())
        self.register_buffer('shared_expert_output_multi',
                             torch.tensor(num_shared_experts / (self.num_experts + num_shared_experts)).float())

        shared_experts = [Expert(embed_dim=self.embed_dim,
                                 ffn_dim=self.ffn_dim,
                                 scale_fc=expert_scale_fc,
                                 activation_dropout=expert_activation_dropout) for _ in range(num_shared_experts)]

        self.shared_experts = ModuleList(shared_experts)

        self.balance_loss_coef = balance_loss_coef
        self.router_z_loss_coef = router_z_loss_coef

    def forward(
            self,
            x,  # [n, b, d]
            noise_gates=False,
            noise_mult=1.
    ):
        print(f'size_out: {x.size()}')
        x_reshaped = x.permute(1, 0, 2)  # [n, b, d] -> [b, n, d]
        dispatch_tensor, combine_tensor, balance_loss, router_z_loss = self.gate(x_reshaped, noise_gates=noise_gates,
                                                                                 noise_mult=noise_mult)

        # dispatch
        expert_inputs = einsum('b n d, b n e c -> b e c d', x_reshaped, dispatch_tensor)

        # feed the expert inputs through the experts.
        expert_outputs = self.experts(expert_inputs)

        # combine
        experts_output_tensor = einsum('b e c d, b n e c -> b n d', expert_outputs, combine_tensor)  # [b, n, d]
        expert_output_multi = getattr(self, 'expert_output_multi')

        shared_experts_output = []
        for expert in self.shared_experts:
            shared_experts_output.append(expert(x_reshaped))

        shared_experts_output_tensor = torch.stack(shared_experts_output, dim=-1).sum(dim=-1)
        shared_expert_output_multi = getattr(self, 'shared_expert_output_multi')

        combined_output = shared_experts_output_tensor * shared_expert_output_multi \
                          + experts_output_tensor * expert_output_multi
        print(f'size_out: {combined_output.size()}')

        # losses
        weighted_balance_loss = balance_loss * self.balance_loss_coef
        weighted_router_z_loss = router_z_loss * self.router_z_loss_coef

        # combine the losses
        total_aux_loss = weighted_balance_loss + weighted_router_z_loss

        return MixtureOfExpertsReturn(combined_output, total_aux_loss.squeeze())


# sparse moe block
# in particular, they found that adding a feedforward before or after greatly stabilized the training and improved results
class SparseMoEBlock(Module):
    @beartype
    def __init__(
            self,
            moe: MoE,
            *,
            add_ff_before=False,
            add_ff_after=True
    ):
        super().__init__()
        embed_dim = moe.embed_dim
        ffn_dim = moe.ffn_dim

        self.moe = moe
        self.moe_prenorm = RMSNorm(embed_dim)

        self.ff_before = Expert(embed_dim, ffn_dim=ffn_dim, prenorm=True) if add_ff_before else None
        self.ff_after = Expert(embed_dim, ffn_dim=ffn_dim, prenorm=True) if add_ff_after else None

    def forward(
            self,
            x,
            noise_gates=False,
            noise_mult=1.
    ):

        # feedforward before
        if exists(self.ff_before):
            x = self.ff_before(x) + x

        # mixture of experts layer
        residual = x
        moe_out, total_aux_loss = self.moe(self.moe_prenorm(x), noise_gates=noise_gates, noise_mult=noise_mult)
        x = moe_out + residual

        # feedforward after
        if exists(self.ff_after):
            x = self.ff_after(x) + x

        return MixtureOfExpertsReturn(x, total_aux_loss)
