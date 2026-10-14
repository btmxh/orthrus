"""Frozen Qwen3.5 multi-token-prediction (MTP) head, used to pre-draft diffusion blocks.

The checkpoint's `mtp.*` weights: fc(concat[norm(embed(x_{p+1})), norm(h_p)]) -> one gated
full-attention decoder layer -> norm, then the shared lm_head predicts x_{p+2}. Applying it
again to its own output and predicted token chains further ahead.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5MLP,
    Qwen3_5RMSNorm,
    apply_rotary_pos_emb,
)


class Qwen3_5MTPAttention(nn.Module):
    """Gated full attention with the checkpoint's parameter layout. SDPA, not flash:
    the MTP head is frozen, and chained steps need an irregular boolean mask."""

    def __init__(self, config):
        super().__init__()
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        num_q_heads, num_kv_heads = config.num_attention_heads, config.num_key_value_heads
        bias = config.attention_bias
        self.q_proj = nn.Linear(config.hidden_size, num_q_heads * self.head_dim * 2, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_q_heads * self.head_dim, config.hidden_size, bias=bias)
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, prefix_kv=None, attn_mask=None):
        """Causal over `hidden_states` when no prefix/mask is given; otherwise attends to
        cat(prefix_kv, own keys) under `attn_mask` (bool, True = visible).
        Returns (output, (keys, values)) where keys/values are this call's own."""
        batch_size, seq_len, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim * 2), 2, dim=-1
        )
        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        keys, values = key_states, value_states
        if prefix_kv is not None:
            keys = torch.cat([prefix_kv[0], key_states], dim=2)
            values = torch.cat([prefix_kv[1], value_states], dim=2)
        output = F.scaled_dot_product_attention(
            query_states, keys, values, attn_mask=attn_mask,
            is_causal=attn_mask is None, enable_gqa=True,
        )
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = output * torch.sigmoid(gate.reshape(batch_size, seq_len, -1))
        return self.o_proj(output), (key_states, value_states)


class Qwen3_5MTPLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3_5MTPAttention(config)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)

    def forward(self, hidden_states, position_embeddings, prefix_kv=None, attn_mask=None):
        attn_out, kv = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, prefix_kv, attn_mask
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, kv


class Qwen3_5MTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        if getattr(config, "mtp_num_hidden_layers", 1) != 1:
            raise NotImplementedError("Only single-layer MTP heads (Qwen3.5) are supported.")
        hidden_size = config.hidden_size
        self.fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList([Qwen3_5MTPLayer(config)])
        self.norm = Qwen3_5RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def step(self, token_embeds, hidden_states, position_embeddings, prefix_kv=None,
             attn_mask=None):
        """One MTP application: (embed(x_{p+1}), h_p) -> normed hidden predicting x_{p+2}.
        The normed output is also the `hidden_states` input of the next chained step."""
        fused = self.fc(torch.cat([
            self.pre_fc_norm_embedding(token_embeds), self.pre_fc_norm_hidden(hidden_states),
        ], dim=-1))
        fused, kv = self.layers[0](fused, position_embeddings, prefix_kv, attn_mask)
        return self.norm(fused), kv

    @torch.no_grad()
    def seed_tokens(self, ar_hidden, input_ids, anchors, num_seeds, embed_tokens, lm_head,
                    rope):
        """Greedy MTP drafts for the `num_seeds` tokens after each anchor: (B, A, num_seeds).
        `ar_hidden` is the AR view's last hidden state after the final norm.

        Step 1 runs causally over the row: position p reads (h_p, x_{p+1}) and predicts
        x_{p+2}, so anchor a takes p = a-1 and drafts x_{a+1} -- from state the AR pass
        already has when the anchor token arrives. Steps >= 2: see `chain`.
        """
        seq_len = ar_hidden.shape[1]
        token_embeds = embed_tokens(input_ids[:, 1:])
        positions = torch.arange(seq_len - 1, device=ar_hidden.device).expand(ar_hidden.shape[0], -1)
        out, prefix_kv = self.step(token_embeds, ar_hidden[:, :-1], rope(token_embeds, positions))
        hidden = torch.gather(
            out, 1, (anchors - 1).unsqueeze(-1).expand(-1, -1, out.shape[-1])
        )                                                            # (B, A, H)
        # Step-1 keys visible to each anchor's chain: positions p < a.
        prefix_visible = (
            torch.arange(seq_len - 1, device=ar_hidden.device).view(1, 1, -1)
            < anchors.unsqueeze(-1)
        )                                                            # (B, A, S-1)
        return self.chain(hidden, anchors, num_seeds, prefix_kv, prefix_visible,
                          embed_tokens, lm_head, rope)

    @torch.no_grad()
    def chain(self, hidden, anchors, num_seeds, prefix_kv, prefix_visible, embed_tokens,
              lm_head, rope):
        """Drafts from each anchor's step-1 output `hidden` (B, A, H). Step j >= 2 is a query
        per anchor at position a+j-2 reading (previous step's output, previous draft); it
        attends to the step-1 keys `prefix_kv` where `prefix_visible` (B, A, P) plus its own
        chain, as in incremental MTP decoding."""
        batch_size, num_anchors, _ = hidden.shape
        prefix_keys, prefix_values = prefix_kv
        same_anchor = torch.eye(num_anchors, dtype=torch.bool, device=hidden.device)
        drafts = [lm_head(hidden).argmax(dim=-1)]
        chain_keys, chain_values = [], []
        for step in range(2, num_seeds + 1):
            step_embeds = embed_tokens(drafts[-1])
            # Keys: step-1 prefix, earlier chain steps, then this step's own (laid out
            # step-major, anchor-minor); a chain query sees its own anchor's entries only.
            attn_mask = torch.cat([
                prefix_visible,
                same_anchor.repeat(1, step - 1).unsqueeze(0).expand(batch_size, -1, -1),
            ], dim=-1).unsqueeze(1)                                  # (B, 1, A, P+A*(step-1))
            step_kv = (
                torch.cat([prefix_keys, *chain_keys], dim=2),
                torch.cat([prefix_values, *chain_values], dim=2),
            )
            hidden, (step_keys, step_values) = self.step(
                step_embeds, hidden, rope(step_embeds, anchors + step - 2), step_kv, attn_mask
            )
            chain_keys.append(step_keys)
            chain_values.append(step_values)
            drafts.append(lm_head(hidden).argmax(dim=-1))
        return torch.stack(drafts, dim=-1)
