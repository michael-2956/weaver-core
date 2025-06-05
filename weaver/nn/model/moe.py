# https://github.com/lucidrains/st-moe-pytorch
# https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/main/modeling_deepseek.py

from __future__ import annotations

import math
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


class DeepseekRMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        """
        DeepseekRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


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


class MoEGate(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_experts_per_tok=3,
                 n_routed_experts=31,
                 scoring_func='softmax',
                 aux_loss_alpha=0.001,
                 seq_aux=True,
                 norm_topk_prob=True):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = n_routed_experts

        self.scoring_func = scoring_func
        self.alpha = aux_loss_alpha
        self.seq_aux = seq_aux

        # topk selection algorithm
        self.norm_topk_prob = norm_topk_prob
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.embed_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        seq_len, bsz, embed_dim = x.shape
        ### compute gating score
        logits = F.linear(x.view(-1, embed_dim), self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=x.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=x.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


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

        self.pre_fc_norm = DeepseekRMSNorm(embed_dim) if prenorm else None
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

# plain mixture of experts
class MoE(Module):
    @beartype
    def __init__(self,
                 embed_dim,
                 ffn_dim,
                 num_experts=16,
                 num_experts_per_tok=2,
                 num_shared_experts=1,
                 m=2,
                 expert_scale_fc=False,
                 expert_activation_dropout=0.1,
                 seq_aux=True,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim // m
        self.num_experts = num_experts * m - num_shared_experts
        self.top_n = num_experts_per_tok * m - num_shared_experts
        self.n_shared_experts = num_shared_experts if num_shared_experts != 0 else None

        self.gate = MoEGate(embed_dim,
                            num_experts_per_tok=self.top_n,
                            n_routed_experts=self.num_experts,
                            scoring_func='softmax',
                            aux_loss_alpha=0.001,
                            seq_aux=seq_aux,
                            norm_topk_prob=True)

        self.experts = nn.ModuleList([Expert(embed_dim=self.embed_dim,
                                      ffn_dim=self.ffn_dim,
                                      scale_fc=expert_scale_fc,
                                      activation_dropout=expert_activation_dropout)
                                  for _ in range(self.num_experts)])

        if self.n_shared_experts is not None:
            batched_ffn_dim = ffn_dim * self.n_shared_experts
            self.shared_experts = Expert(embed_dim=self.embed_dim,
                                         ffn_dim=batched_ffn_dim,
                                         scale_fc=expert_scale_fc,
                                         activation_dropout=expert_activation_dropout)

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)
        print(topk_idx.size())
        print(topk_idx)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.top_n, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x, dtype=torch.float16)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.top_n
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache


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
        self.moe_prenorm = DeepseekRMSNorm(embed_dim)

        self.ff_before = Expert(embed_dim, ffn_dim=ffn_dim, prenorm=True) if add_ff_before else None
        self.ff_after = Expert(embed_dim, ffn_dim=ffn_dim, prenorm=True) if add_ff_after else None

    def forward(
            self,
            x,
    ):

        # feedforward before
        if exists(self.ff_before):
            x = self.ff_before(x) + x

        # mixture of experts layer
        residual = x
        moe_out = self.moe(self.moe_prenorm(x))
        x = moe_out + residual

        # feedforward after
        if exists(self.ff_after):
            x = self.ff_after(x) + x

        return x
