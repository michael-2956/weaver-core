import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from linformer_pytorch import LinearAttentionHead, get_EF

class FFNBlockSection(nn.Module):
    def __init__(self,
                 embed_dim,
                 ffn_dim,
                 activation='gelu',
                 activation_dropout=0.1,
                 scale_fc=True):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        return x

class MoEFFN(nn.Module):
    def __init__(self,
                 # MoE params
                 embed_dim,
                 ffn_dim,
                 N=16,
                 k_shared=1,
                 m=2,
                 top_k=2,
                 device_count=1,
                 expert_balance_alpha=0.01,
                 device_balance_alpha=0.1,
                 # Expert params
                 activation='gelu',
                 activation_dropout=0.1,
                 scale_fc=True):
        """
        Mixture-of-Experts block replacing a standard FFN.
        - N: total number of experts (part of m * N in DeepSeekMoE).
        - k_shared: number of shared experts always activated.
        - m: fine-graining scalar, by which we 'split' every expert, and multiply their total number
        - top_k: total number of experts activated per token (K, including shared).
        - device_count: number of device groups for experts (for device-level balancing).
        - expert_balance_alpha: α₁ coefficient for expert-level load balance loss.
        - device_balance_alpha: α₂ coefficient for device-level load balance loss.
        """
        super().__init__()
        assert 0 <= k_shared < N, "k_shared must be less than total experts"
        assert top_k > k_shared, "top_k must be greater than k_shared (at least one routed expert)"
        assert m >= 1, "m must be greater than 1"
        assert int(m) == m, "m must be an int"
        self.num_experts = m * N
        self.expert_dim = math.ceil(ffn_dim / m)
        self.k_shared = k_shared
        self.route_experts = self.num_experts - self.k_shared # total experts that can be routed
        self.top_k = m * top_k                      # total experts used per token (shared + routed)
        self.k_route = self.top_k - self.k_shared             # number of experts chosen by gating (excluding shared)
        self.device_count = device_count
        self.expert_balance_alpha = expert_balance_alpha
        self.device_balance_alpha = device_balance_alpha

        # Gating network: a linear layer that outputs a score for each expert.
        # (Bias is allowed; it can learn an offset for each expert's logit.)
        self.gate = nn.Linear(embed_dim, self.num_experts)

        # Expert networks: Each expert is a two-layer feed-forward (FFN) with GELU.
        self.experts = nn.ModuleList([
            FFNBlockSection(embed_dim, self.expert_dim, activation, activation_dropout, scale_fc)
        ])

        # Define grouping of experts into devices for device-level loss calculation.
        # We'll assign routed experts (indices k_shared ... num_experts-1) evenly to `device_count` groups.
        self.expert_group = []
        if device_count > 1:
            base = self.route_experts // device_count
            extra = self.route_experts % device_count
            start_idx = k_shared
            for d in range(device_count):
                group_size = base + (1 if d < extra else 0)
                group_experts = list(range(start_idx, start_idx + group_size))
                self.expert_group.append(group_experts)
                start_idx += group_size
        else:
            # All experts (or all routed experts) on one device group
            self.expert_group = [list(range(k_shared, self.num_experts))]

    def forward(self, x):
        """
        x: (seq_len, batch, embed_dim)
        
        Forward pass of MoE block.
        Returns a tuple: (output, expert_balance_loss, device_balance_loss)
        """
        seq_len, batch_size, embed_dim = x.size()
        # Flatten batch and sequence into one dimension for routing
        x_flat = x.reshape(batch_size * seq_len, embed_dim)  # shape [T, d] where T = batch_size * seq_len

        # Compute gating scores for all experts
        gate_scores = self.gate(x_flat)  # shape [T, num_experts]
        # Exclude shared experts from routing selection by masking their scores (they will be added deterministically)
        if self.k_shared > 0:
            gate_scores[:, :self.k_shared] = -1e9  # a very large negative value to effectively zero out softmax for shared idx
        # Compute softmax probabilities for gating (over all experts, shared ones effectively ~0 after masking)
        gate_probs = torch.softmax(gate_scores, dim=-1)  # shape [T, num_experts]

        # Select top-k_route experts for each token (these are indices >= k_shared due to masking)
        # torch.topk returns the top values and indices along the last dimension
        topk_vals, topk_idx = torch.topk(gate_probs, self.k_route, dim=-1)  # shapes [T, k_route]

        # Initialize output contributions (on flattened tokens)
        output_flat = torch.zeros_like(x_flat)  # [T, d]

        # Always-on shared experts: compute their output for all tokens and add.
        if self.k_shared > 0:
            # For each shared expert, apply it to all tokens and accumulate
            for j in range(self.k_shared):
                output_flat += self.experts[j](x_flat)  # every token goes through expert j (shared)

        # Routed experts: for each token, we have selected expert indices in topk_idx
        # We will gather tokens per expert and apply the expert.
        T = x_flat.size(0)
        # Flatten token indices and corresponding expert selections
        token_indices = torch.arange(T, device=x.device).unsqueeze(1).expand(-1, self.k_route)  # [T, k_route]
        flat_tokens = token_indices.reshape(-1)   # [T * k_route]
        flat_experts = topk_idx.reshape(-1)       # [T * k_route]
        flat_gates = topk_vals.reshape(-1)        # [T * k_route]

        # Sort the selections by expert index to process tokens expert-by-expert
        sorted_idx = torch.argsort(flat_experts)
        flat_experts_sorted = flat_experts[sorted_idx]   # sorted expert indices
        flat_tokens_sorted = flat_tokens[sorted_idx]     # corresponding token indices
        flat_gates_sorted = flat_gates[sorted_idx]       # corresponding gate values

        # Iterate through sorted lists and batch tokens for each expert
        idx = 0
        n = flat_experts_sorted.numel()
        while idx < n:
            exp_id = int(flat_experts_sorted[idx].item())
            # Gather all tokens for this expert exp_id
            same_exp_indices = []
            same_exp_tokens = []
            while idx < n and int(flat_experts_sorted[idx].item()) == exp_id:
                same_exp_indices.append(idx)
                same_exp_tokens.append(int(flat_tokens_sorted[idx].item()))
                idx += 1
            # Convert to tensor
            token_batch = torch.tensor(same_exp_tokens, device=x.device, dtype=torch.long)
            gate_batch = flat_gates_sorted[same_exp_indices].unsqueeze(1)  # shape [num_tokens_for_exp, 1]
            # Run the expert FFN on all these tokens at once
            expert_out = self.experts[exp_id](x_flat[token_batch])  # shape [num_tokens_for_exp, d]
            # Multiply outputs by their respective gate weights
            expert_out *= gate_batch  # broadcast multiply each vector by the scalar weight
            # Add the weighted outputs to the respective token positions in output_flat
            output_flat.index_add_(0, token_batch, expert_out)

        output = output_flat.view(seq_len, batch_size, embed_dim)

        # **Compute Load-Balancing Losses** (expert-level and device-level):
        with torch.no_grad():
            if self.training:
                # Calculate f_i (fraction of tokens) and p_i (average gate probability) for each routed expert.
                # Count how many times each expert index appears in flat_experts (i.e., selected for some token)
                expert_selection_counts = torch.bincount(flat_experts, minlength=self.num_experts)
                # We only care about routed experts (shared experts usage is deterministic, exclude them from balance loss)
                expert_selection_counts = expert_selection_counts[self.k_shared:]  # length N'
                # Fraction of tokens for expert i: f_i = (N' / (K' * T)) * (#tokens routed to i)
                K_prime = self.k_route  # number of experts chosen per token (excluding shared)
                T_float = float(T)
                f = (self.route_experts / (K_prime * T_float)) * expert_selection_counts.float()  # shape [N']
                # Average gating probability for expert i: p_i = (1/T) * sum_{t} s_{i,t}
                # Sum gating probs for each expert over all tokens:
                total_probs_per_expert = gate_probs.sum(dim=0)  # shape [num_experts]
                total_probs_per_expert = total_probs_per_expert[self.k_shared:]  # only routed experts
                p = total_probs_per_expert / T_float  # shape [N']

                # Expert-level balance loss: α1 * sum_i f_i * p_i
                expert_balance_loss = self.expert_balance_alpha * (f * p).sum()

                # Device-level balance loss: α2 * sum_{device=1..D} f'_d * p'_d
                if self.device_count > 1:
                    # Compute f'_d and p'_d for each device group
                    f_group = torch.zeros(self.device_count, device=x.device)
                    p_group = torch.zeros(self.device_count, device=x.device)
                    # Sum f and p for experts in each group
                    for d_idx, exp_indices in enumerate(self.expert_group):
                        if len(exp_indices) == 0:
                            continue
                        # Convert global expert indices to indices in f/p array (subtract k_shared)
                        routed_indices = [j - self.k_shared for j in exp_indices if j >= self.k_shared]
                        if len(routed_indices) == 0:
                            continue
                        f_group[d_idx] = f[routed_indices].mean()   # average f for group d
                        p_group[d_idx] = p[routed_indices].sum()    # total p for group d
                    device_balance_loss = self.device_balance_alpha * (f_group * p_group).sum()
                else:
                    device_balance_loss = torch.tensor(0.0, device=x.device)
            else:
                expert_balance_loss = torch.tensor(0.0, device=x.device)
                device_balance_loss = torch.tensor(0.0, device=x.device)

        return output, expert_balance_loss, device_balance_loss

class EfficientAttention(nn.Module):
    """
    A drop-in replacement for nn.MultiheadAttention that uses
    F.scaled_dot_product_attention under the hood when attention_mode='classic'.
    
    Otherwise, it uses linear attention. The arguments input_seq_len and
    lin_proj_dim are used to create relevant projection matrices.

    Expects the input q/k/v embeddings to be of embed_dim length.
    """
    def __init__(
            self,
            embed_dim,
            num_heads,
            attn_dropout=0.1,
            add_bias_kv=False,
            attention_mode='classic',
            input_seq_len=None,
            lin_proj_dim=None,
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # the actual per-head q/k/v embedding size
        self.head_dim = embed_dim // num_heads
        self.attn_dropout = attn_dropout
        self.add_bias_kv = add_bias_kv

        # Linear projections for Q, K, V, plus final output projection
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # If add_bias_kv=True, create bias_k and bias_v (shape [1, 1, embed_dim])
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        assert attention_mode in ['linformer', 'classic']
        self.attention_mode = attention_mode
        if attention_mode == 'linformer':
            assert input_seq_len is not None and lin_proj_dim is not None
            self.heads = nn.ModuleList()
            if self.add_bias_kv:
                input_seq_len += 1
            for _ in range(num_heads):
                E_proj = get_EF(input_seq_len, lin_proj_dim)
                self.heads.append(LinearAttentionHead(
                    dim=embed_dim, dropout=attn_dropout,
                    E_proj=E_proj, F_proj=E_proj, # becaue self attention
                    causal_mask=None,
                ))

    def forward(
            self,
            query,                            # (Lq, B, embed_dim)
            key,                              # (Lkv, B, embed_dim)
            value,                            # (Lkv, B, embed_dim)
            key_padding_mask=None,            # (B, Lkv)
            attn_mask=None,                   # (B*num_heads, Lq, Lkv)
            return_final_attn_weight=False,
            return_qk_attn_weight_logits=False,
            use_qk_attn_weight_logits=None,   # (B, num_heads, Lq, Lkv)
        ):
        """
        - `return_final_attn_weight` makes the function \\
          return `(B, num_heads, Lq, Lkv)` final softmax attention weights, \\
          but makes the computation a bit slower
        - `return_qk_attn_weight_logits` makes the function \\
          return `(B, num_heads, Lq, Lkv)` initial (no masks, no softmax) \\
          attention weight logits and makes the function not compute output
        - `use_qk_attn_weight_logits` should be the transformed attn weight logits \\
          of shape `(B, num_heads, Lq, Lkv)`
        """
        # number of queries, batch size
        Lq, B, _ = query.shape
        # number of keys/values
        Lkv, _, _ = key.shape

        # project
        q = self.q_proj(query)   # (Lq, B, embed_dim)
        k = self.k_proj(key)     # (Lkv, B, embed_dim)
        v = self.v_proj(value)   # (Lkv, B, embed_dim)

        # Add bias_k, bias_v to keys and values.
        if self.add_bias_kv:
            # Repeat bias B times along batch dimension and concatenate to batch keys/values
            k = torch.cat([k, self.bias_k.repeat(1, B, 1)], dim=0)  # shape: (Lkv+1, B, embed_dim)
            v = torch.cat([v, self.bias_v.repeat(1, B, 1)], dim=0)  # shape: (Lkv+1, B, embed_dim)

            # do not mask out the original tokens (value = 0 by default)
            if key_padding_mask is not None:
                # adds padding on the right hand side of last dim
                key_padding_mask = F.pad(key_padding_mask, (0, 1))  # (B, Lkv+1)

            # pad with zeroes
            if attn_mask is not None:
                # adds padding on the right hand side of last dim
                attn_mask = F.pad(attn_mask, (0, 1))  # (B*num_heads, Lq, Lkv+1)

            # increase length
            Lkv = Lkv + 1

        q = q.transpose(0, 1)  # shape: (B, Lq, E)
        k = k.transpose(0, 1)  # shape: (B, Lkv, E)
        v = v.transpose(0, 1)  # shape: (B, Lkv, E)

        # eplit embeddings into per-head embeddings
        q = q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)  # shape: (B, num_heads, Lq, head_dim)
        k = k.view(B, Lkv, self.num_heads, self.head_dim).transpose(1, 2) # shape: (B, num_heads, Lkv, head_dim)
        v = v.view(B, Lkv, self.num_heads, self.head_dim).transpose(1, 2) # shape: (B, num_heads, Lkv, head_dim)

        if return_qk_attn_weight_logits:
            assert self.attention_mode == 'classic'  # linformer may be supported if need be
            return q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))

        if self.attention_mode == 'classic':
            
            if key_padding_mask is not None:
                # only accept boolean key_padding_mask
                assert key_padding_mask.dtype == torch.bool
                # make it additive
                addidive_key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).float() * -1e9  # (B, 1, 1, Lkv)
                if attn_mask is None:
                    attn_mask = addidive_key_padding_mask
                else:
                    # add them together
                    assert attn_mask.shape[0] == B * self.num_heads
                    attn_mask = attn_mask.view(B, self.num_heads, Lq, Lkv)  # (B, num_heads, Lq, Lkv)
                    attn_mask = attn_mask + addidive_key_padding_mask       # add across all heads and queries
            else:
                assert attn_mask.shape[0] == B * self.num_heads
                attn_mask = attn_mask.view(B, self.num_heads, Lq, Lkv)  # (B, num_heads, Lq, Lkv)
            
            if return_final_attn_weight:
                if use_qk_attn_weight_logits is None:
                    # (B, num_heads, Lq, head_dim) @ (B, num_heads, head_dim, Lkv) = (B, num_heads, Lq, Lkv)
                    attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
                else:
                    attn_weight = use_qk_attn_weight_logits
                if attn_mask is not None:
                    attn_weight += attn_mask
                attn_weight = torch.softmax(attn_weight, dim=-1)
                attn_weight = torch.dropout(attn_weight, (self.attn_dropout if self.training else 0.0), train=True)
                attn_output = attn_weight @ v
            else:
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]):
                    attn_output = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attn_mask,
                        # ensure dropout does not apply in eval mode
                        dropout_p=(self.attn_dropout if self.training else 0.0)
                    )  # shape: (B, num_heads, Lq, head_dim)
        
        elif self.attention_mode == 'linformer':
            
            assert not return_final_attn_weight  # may be supported if need be
            assert use_qk_attn_weight_logits is None  # linformer would need something different here
            assert key_padding_mask.dtype == torch.bool
            assert attn_mask is None
            
            input_mask = None
            if key_padding_mask is not None:
                input_mask = ~key_padding_mask  # (B, Lkv)
            head_outputs = []
            
            for hdi, head in enumerate(self.heads):
                q_head = q[:, hdi]  # shape: (B, Lq, head_dim)
                k_head = k[:, hdi]  # shape: (B, Lkv, head_dim)
                v_head = v[:, hdi]  # shape: (B, Lkv, head_dim)
                out_h = head(q_head, k_head, v_head, input_mask=input_mask)  # (B, Lq, head_dim)
                head_outputs.append(out_h.unsqueeze(1))  # (B, 1, Lq, head_dim)
            
            attn_output = torch.cat(head_outputs, dim=1)  # (B, num_heads, Lq, head_dim)

        # shape: (B, Lq, embed_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, Lq, self.embed_dim)
        # shape: (Lq, B, embed_dim)
        attn_output = attn_output.transpose(0, 1)

        # Final projection of the results
        # shape: (Lq, B, embed_dim)
        output = self.out_proj(attn_output)

        if return_final_attn_weight:
            return output, attn_weight
        else:
            return output


class AlteredBlock(nn.Module):
    def __init__(
            self,
            embed_dim=128,
            num_heads=8,
            ffn_ratio=4,
            dropout=0.1,
            attn_dropout=0.1,
            activation_dropout=0.1,
            add_bias_kv=False,
            activation='gelu',
            attention='classic',
            input_seq_len=None,
            lin_proj_dim=None,
            scale_fc=True,
            scale_attn=True,
            scale_heads=True,
            scale_resids=True,
            # MoE params
            use_moe=True,
            N=16,
            k_shared=1,
            m=2,
            top_k=2,
            device_count=1,
            expert_balance_alpha=0.01,
            device_balance_alpha=0.1
        ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = EfficientAttention(
            embed_dim, num_heads, attn_dropout=attn_dropout, add_bias_kv=add_bias_kv,
            attention_mode=attention, input_seq_len=input_seq_len, lin_proj_dim=lin_proj_dim
        )
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.use_moe = use_moe
        if self.use_moe:
            self.ffn = MoEFFN(self.embed_dim, self.ffn_dim, N, k_shared, m, top_k, device_count, expert_balance_alpha, device_balance_alpha, activation, activation_dropout, scale_fc)
        else:
            self.ffn = FFNBlockSection(self.embed_dim, self.ffn_dim, activation, activation_dropout, scale_fc)

        self.c_attn = nn.Parameter(torch.ones(num_heads), requires_grad=True) if scale_heads else None
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None

    def forward(
            self,
            x,
            x_cls=None,
            padding_mask=None,
            attn_mask=None,
            return_final_attn_weight=False,
            return_qk_attn_weight_logits=False,
            use_qk_attn_weight_logits=None,   # (B, num_heads, Lq, Lkv)
        ):
        """
        x: (seq_len, batch, embed_dim)
           input to the layer
        x_cls: (1, batch, embed_dim) or None
               class token input to the layer
        padding_mask: (batch, seq_len) with True for padded positions.
        attn_mask: additive float mask; shape may be (batch*num_heads, seq_len, seq_len)
        """

        if x_cls is not None:
            # Class attention branch.
            # Prepend x_cls to x and update the padding mask.
            with torch.no_grad():
                # shape: (batch, 1+seq_len)
                padding_mask = torch.cat((
                    torch.zeros_like(padding_mask[:, :1]), padding_mask
                ), dim=1)
            residual = x_cls
            u = torch.cat((x_cls, x), dim=0)  # (seq_len+1, batch, embed_dim)
            u = self.pre_attn_norm(u)
            assert not return_qk_attn_weight_logits   # support can be added if need be
            assert use_qk_attn_weight_logits is None  # support can be added if need be
            # uses the class token as query and the rest as key/value.
            if return_final_attn_weight:
                x, attn_weight = self.attn(u[:1], u, u, key_padding_mask=padding_mask, return_final_attn_weight=True)
            else:
                x = self.attn(u[:1], u, u, key_padding_mask=padding_mask)
        else:
            # Self-attention branch.
            residual = x
            x = self.pre_attn_norm(x)
            
            if return_qk_attn_weight_logits:
                assert use_qk_attn_weight_logits is None
                return self.attn(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask, return_qk_attn_weight_logits=True)
            if use_qk_attn_weight_logits is not None:
                assert not return_final_attn_weight  # support can be added if need be
                assert not return_qk_attn_weight_logits
                x = self.attn(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask, use_qk_attn_weight_logits=use_qk_attn_weight_logits)
            else:
                if return_final_attn_weight:
                    x, attn_weight = self.attn(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask, return_final_attn_weight=True)
                else:
                    x = self.attn(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask)

        # Optionally apply head scaling.
        if self.c_attn is not None:
            # x: (tgt_len, batch, embed_dim) -> reshape to (tgt_len, batch, num_heads, head_dim)
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            # scale each head by its learned parameter.
            x = torch.einsum('tbhd,h->tbdh', x, self.c_attn)
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x = x + residual

        # ============ FFN Section ============
        residual = x
        x = self.pre_fc_norm(x)

        # ============ MOE Section ============
        if self.use_moe:
            x, expert_loss, device_loss = self.ffn(x)
            total_loss = expert_loss + device_loss
        else:
            x, _, _ = self.ffn(x)
            total_loss = torch.tensor(0.0, device=x.device)
        # ============ MOE Section ============
        
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = x + residual
        # ============ FFN Section ============

        if return_final_attn_weight:
            return x, attn_weight, total_loss
        else:
            return x, None, total_loss

