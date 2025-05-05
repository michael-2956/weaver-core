import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from linformer_pytorch import LinearAttentionHead, get_EF

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
            query,                  # (Lq, B, embed_dim)
            key,                    # (Lkv, B, embed_dim)
            value,                  # (Lkv, B, embed_dim)
            key_padding_mask=None,  # (B, Lkv)
            attn_mask=None          # (B*num_heads, Lq, Lkv)
        ):
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
            
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]):
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    # ensure dropout does not apply in eval mode
                    dropout_p=(self.attn_dropout if self.training else 0.0)
                )  # shape: (B, num_heads, Lq, head_dim)
        
        elif self.attention_mode == 'linformer':
            
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
        return self.out_proj(attn_output)


class AlteredBlock(nn.Module):
    def __init__(
            self, embed_dim=128, num_heads=8, ffn_ratio=4,
            dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
            add_bias_kv=False, activation='gelu',
            attention='classic', input_seq_len=None, lin_proj_dim=None,
            scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True
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
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.c_attn = nn.Parameter(torch.ones(num_heads), requires_grad=True) if scale_heads else None
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
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
            # uses the class token as query and the rest as key/value.
            x = self.attn(u[:1], u, u, key_padding_mask=padding_mask)
        else:
            # Self-attention branch.
            residual = x
            x = self.pre_attn_norm(x)
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
        x = x + residual
        return x

