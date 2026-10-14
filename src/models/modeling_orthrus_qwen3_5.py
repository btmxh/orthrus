"""Orthrus dual-view diffusion on the Qwen3.5 hybrid backbone.
"""

from __future__ import annotations

import functools
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5MLP,
    Qwen3_5PreTrainedModel,
    Qwen3_5RMSNorm,
    Qwen3_5RMSNormGated,
    Qwen3_5TextRotaryEmbedding,
    apply_mask_to_padding_states,
    apply_rotary_pos_emb,
    eager_attention_forward,
    l2norm,
    torch_chunk_gated_delta_rule,
)

from fla.modules.conv import causal_conv1d as fla_causal_conv1d

from src.models.config_orthrus_qwen3_5 import OrthrusQwen3_5Config

_compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
_compiled_create_block_mask = torch.compile(create_block_mask, dynamic=False)

DIFF_PARAM_SUFFIXES = (
    # full attention
    "q_proj_diff", "k_proj_diff", "v_proj_diff", "o_proj_diff",
    "q_norm_diff", "k_norm_diff",
    # gated delta-net
    "in_proj_qkv_diff", "in_proj_z_diff", "in_proj_b_diff", "in_proj_a_diff",
    "conv1d_diff", "dt_bias_diff", "A_log_diff", "norm_diff", "out_proj_diff",
    # fixed-state gated delta-net (query-only projection)
    "in_proj_q_diff",
)


def _flex_flash_backend_available() -> bool:
    """Checks if the flash_attn.cute backend (Flash Attention 4) is available."""
    import importlib.util
    try:
        return importlib.util.find_spec("flash_attn.cute") is not None
    except ModuleNotFoundError:  # find_spec imports the parent; flash_attn itself is absent
        return False


_FLEX_FLASH_BACKEND = _flex_flash_backend_available()


@functools.cache
def _small_smem_bwd_options(device_index: int) -> tuple[tuple[str, int], ...]:
    """Smaller flex backward tiles for GPUs with <160KB smem/block (consumer Ampere/Ada):
    the default head_dim=256 backward config exceeds their ~99KB limit. A100/H100 unaffected."""
    smem = torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin
    if smem >= 160 * 1024:
        return ()
    return (("bwd_BLOCK_M1", 16), ("bwd_BLOCK_N1", 32), ("bwd_BLOCK_M2", 32),
            ("bwd_BLOCK_N2", 16), ("bwd_num_stages", 1))


def fused_flex_attention(query_states, key_states, value_states, mask=None):
    kernel_options = {}
    if _FLEX_FLASH_BACKEND:
        kernel_options["BACKEND"] = "FLASH"
    if query_states.is_cuda:
        kernel_options.update(_small_smem_bwd_options(query_states.device.index))
    if mask is not None:
        q_block_size, kv_block_size = mask.BLOCK_SIZE
        kernel_options["sparse_block_size"] = (int(q_block_size), int(kv_block_size))
    return _compiled_flex_attention(
        query_states, key_states, value_states,
        block_mask=mask, enable_gqa=True, kernel_options=kernel_options,
    )


def causal_conv1d(hidden_states, conv1d: nn.Conv1d, activation: str | None):
    output, _ = fla_causal_conv1d(
        hidden_states,
        weight=conv1d.weight.squeeze(1),
        bias=conv1d.bias,
        activation=activation,
    )
    return output


def build_diffusion_block_mask(
    anchors: torch.Tensor,
    ar_len: int,
    block_size: int,
    compiled: bool = True,
) -> BlockMask:
    """Dual-pass BlockMask over `cat([AR keys, diffusion keys])`."""
    num_queries = anchors.shape[1] * block_size

    def dual_pass_mask_fn(batch, head, q_idx, kv_idx):
        block_of_query = q_idx // block_size

        # AR keys: a STRICT prefix of this block's own anchor.
        is_ar_key = kv_idx < ar_len
        ar_visible = is_ar_key & (kv_idx < anchors[batch, block_of_query])

        # Diffusion keys: block-diagonal, so a query sees only its own block.
        block_of_key = (kv_idx - ar_len) // block_size
        diffusion_visible = (~is_ar_key) & (block_of_query == block_of_key)
        return ar_visible | diffusion_visible

    builder = _compiled_create_block_mask if compiled else create_block_mask
    return builder(
        dual_pass_mask_fn,
        B=anchors.shape[0],
        H=None,
        Q_LEN=num_queries,
        KV_LEN=ar_len + num_queries,
    )


FLASH_ATTENTION_IMPLS = ("flash_attention_2", "flash_attention_3", "flash_attention_4")


def is_flash_attention(impl: str) -> bool:
    # Without the flash_attn package, transformers swaps in the equivalent hub
    # kernel, e.g. flash_attention_2 -> kernels-community/flash-attn2.
    return impl in FLASH_ATTENTION_IMPLS or (
        impl.startswith("kernels-community/") and "flash-attn" in impl
    )


class OrthrusQwen3_5Attention(nn.Module):
    """Qwen3.5 Full Attention, with a frozen AR view and a trainable diffusion twin."""

    def __init__(self, config: OrthrusQwen3_5Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        num_q_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        bias = config.attention_bias

        # AR View (frozen)
        self.q_proj = nn.Linear(config.hidden_size, num_q_heads * self.head_dim * 2, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_q_heads * self.head_dim, config.hidden_size, bias=bias)
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Diffusion View (trainable)
        self.q_proj_diff = nn.Linear(
            config.hidden_size, num_q_heads * self.head_dim * 2, bias=bias
        )
        self.k_proj_diff = nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj_diff = nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj_diff = nn.Linear(
            num_q_heads * self.head_dim, config.hidden_size, bias=bias
        )
        self.q_norm_diff = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_diff = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _qkv(self, hidden_states, q_proj, k_proj, v_proj, q_norm, k_norm):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(
            q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = k_norm(k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states, gate, input_shape

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        """Frozen AR View"""
        cos, sin = position_embeddings
        query_states, key_states, value_states, gate, input_shape = self._qkv(
            hidden_states, self.q_proj, self.k_proj, self.v_proj, self.q_norm, self.k_norm
        )
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if getattr(self, "_capture_ar", False):
            self._ar_cache = (key_states.detach(), value_states.detach())

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), None

    def diffusion_blocks(
        self,
        block_hidden_states,
        block_pos_emb,
        ar_kv,
        flex_block_mask: BlockMask | None = None,
        ar_seq_len: int | None = None,
    ):
        """Diffusion View"""
        batch_size, num_anchors, block_size, _ = block_hidden_states.shape
        num_queries = num_anchors * block_size

        flat_blocks = block_hidden_states.reshape(batch_size, num_queries, -1)
        query_states, key_states, value_states, gate, _ = self._qkv(
            flat_blocks, self.q_proj_diff, self.k_proj_diff, self.v_proj_diff,
            self.q_norm_diff, self.k_norm_diff,
        )
        cos, sin = block_pos_emb
        # per-block absolute positions
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        ar_key_states, ar_value_states = ar_kv

        if ar_seq_len is not None:
            # Generation
            keys = torch.cat([ar_key_states[:, :, :ar_seq_len, :], key_states], dim=2)
            values = torch.cat([ar_value_states[:, :, :ar_seq_len, :], value_states], dim=2)
            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            attn_output, _ = attention_interface(
                self, query_states, keys, values, None,
                dropout=0.0, scaling=self.scaling, is_causal=False,
            )
            attn_output = attn_output.reshape(batch_size, num_queries, -1).contiguous()
        else:
            # Training
            keys = torch.cat([ar_key_states, key_states], dim=2)   # (batch, kv_heads, ar+q, hd)
            values = torch.cat([ar_value_states, value_states], dim=2)
            attn_output = fused_flex_attention(query_states, keys, values,
                                               mask=flex_block_mask)
            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(batch_size, num_queries, -1).contiguous()

        gated = attn_output * torch.sigmoid(gate)
        return self.o_proj_diff(gated).reshape(batch_size, num_anchors, block_size, -1)


class OrthrusQwen3_5GatedDeltaNet(nn.Module):
    """Qwen3.5 Gated DeltaNet.

    With `fixed_state`, the diffusion twin treats block inputs as carrying no information
    to write: it never updates the AR prefix state, only reads it. That is the gated delta
    rule with beta=0, whose output at t is exp(sum_{j<=t} g_j) * S_prefix^T q_t, so the twin
    drops the k/v projections and beta. The decay stays: although it is a per-head scale,
    prefix readouts mostly sit near norm_diff's eps, so it acts as an amplitude gate that
    fades decayed heads to ~0 rather than being normalized away.
    """

    def __init__(self, config: OrthrusQwen3_5Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.activation = config.hidden_act

        self.fixed_state = layer_idx in config.fixed_state_layers

        def _conv(channels=self.conv_dim):
            return nn.Conv1d(
                channels, channels, bias=False,
                kernel_size=self.conv_kernel_size, groups=channels,
                padding=self.conv_kernel_size - 1,
            )

        # AR View Components
        self.conv1d = _conv()
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # Diffusion View Components
        self.dt_bias_diff = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log_diff = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))
        self.in_proj_z_diff = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_a_diff = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        if self.fixed_state:
            # Query channels only: nothing is written, so no k/v/beta.
            self.conv1d_diff = _conv(self.key_dim)
            self.in_proj_q_diff = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        else:
            self.conv1d_diff = _conv()
            self.in_proj_qkv_diff = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
            self.in_proj_b_diff = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.norm_diff = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj_diff = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def _pre_conv(self, hidden_states, in_qkv, in_beta, in_gate, dt_bias, A_log):
        mixed_qkv = in_qkv(hidden_states)               # fla's conv layout; no transpose
        beta = in_beta(hidden_states).sigmoid()
        # .float() before exp: in fp16 A_log.exp() can otherwise overflow to -inf.
        gate_decay = -A_log.float().exp() * F.softplus(in_gate(hidden_states).float() + dt_bias)
        return mixed_qkv, gate_decay, beta

    def _conv_then_split(self, mixed_qkv, conv1d, keep_len):
        mixed_qkv = causal_conv1d(mixed_qkv, conv1d, self.activation)
        mixed_qkv = mixed_qkv[:, -keep_len:, :]
        query_states, key_states, value_states = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        batch_size = query_states.shape[0]
        query_states = query_states.reshape(batch_size, keep_len, -1, self.head_k_dim)
        key_states = key_states.reshape(batch_size, keep_len, -1, self.head_k_dim)
        value_states = value_states.reshape(batch_size, keep_len, -1, self.head_v_dim)
        if self.num_v_heads > self.num_k_heads:
            heads_per_kv = self.num_v_heads // self.num_k_heads
            query_states = query_states.repeat_interleave(heads_per_kv, dim=2)
            key_states = key_states.repeat_interleave(heads_per_kv, dim=2)
        return query_states, key_states, value_states

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        """Frozen AR Path"""
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        # On the module, not via kwargs: those do not survive FSDP2's wrapped __call__.
        if getattr(self, "_capture_ar", False):
            self._ar_cache = hidden_states.detach()

        mixed_qkv, gate_decay, beta = self._pre_conv(
            hidden_states,
            self.in_proj_qkv,
            self.in_proj_b,
            self.in_proj_a,
            self.dt_bias,
            self.A_log,
        )
        gate_input = self.in_proj_z(hidden_states).reshape(
            batch_size, seq_len, -1, self.head_v_dim
        )
        query_states, key_states, value_states = self._conv_then_split(
            mixed_qkv, self.conv1d, keep_len=seq_len
        )

        core, _ = torch_chunk_gated_delta_rule(
            query_states, key_states, value_states,
            g=gate_decay,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        core = self.norm(
            core.reshape(-1, self.head_v_dim), gate_input.reshape(-1, self.head_v_dim)
        )
        return self.out_proj(core.reshape(batch_size, seq_len, -1))

    @torch.no_grad()
    def ar_prefix_states(self, ar_hidden_states, anchors):
        mixed_qkv, gate_decay, beta = self._pre_conv(
            ar_hidden_states,
            self.in_proj_qkv,
            self.in_proj_b,
            self.in_proj_a,
            self.dt_bias,
            self.A_log,
        )  # (batch, seq_len, conv_dim)
        conv_left_width = self.conv_kernel_size - 1
        seq_len = mixed_qkv.shape[1]
        batch_size, num_anchors = anchors.shape
        num_blocks = batch_size * num_anchors

        offsets = torch.arange(-conv_left_width, 0, device=anchors.device)
        positions = anchors.reshape(batch_size, num_anchors, 1) + offsets.reshape(1, 1, -1)
        in_row = positions >= 0
        gather_index = (
            positions.clamp(min=0)
            .reshape(batch_size, num_anchors * conv_left_width, 1)
            .expand(batch_size, num_anchors * conv_left_width, self.conv_dim)
        )
        conv_lefts = torch.gather(mixed_qkv, 1, gather_index).reshape(
            num_blocks, conv_left_width, self.conv_dim
        )
        conv_lefts = conv_lefts * in_row.reshape(num_blocks, -1, 1).to(conv_lefts.dtype)

        query_states, key_states, value_states = self._conv_then_split(
            mixed_qkv, self.conv1d, keep_len=seq_len
        )
        recurrent_states = self._prefix_states_chained(
            query_states, key_states, value_states, gate_decay, beta, anchors
        )
        return recurrent_states, conv_lefts

    def _zero_state(self, ref):
        return ref.new_zeros(self.num_v_heads, self.head_k_dim, self.head_v_dim,
                             dtype=torch.float32)

    def _prefix_states_chained(
        self,
        query_states,
        key_states,
        value_states,
        gate_decay,
        beta,
        anchors,
    ):
        """Chain the state across each [prev_anchor:anchor) segment. Anchors must be ASCENDING."""
        device = query_states.device
        batch_size, num_anchors = anchors.shape
        states = torch.empty(
            batch_size, num_anchors, self.num_v_heads, self.head_k_dim, self.head_v_dim,
            device=device, dtype=torch.float32,
        )
        state = None
        prev_anchor = torch.zeros(batch_size, dtype=torch.long, device=device)
        anchors_host = anchors.to("cpu")  # one transfer; the loop bounds are host-side

        for anchor_idx in range(num_anchors):
            cur_anchor = anchors_host[:, anchor_idx]
            segment_lens = (cur_anchor - prev_anchor.cpu()).clamp(min=0)
            if int(segment_lens.max()) > 0:
                window_start = int(prev_anchor.cpu().min())
                window_width = int((cur_anchor - window_start).max())
                window = slice(window_start, window_start + window_width)
                segment_gate = gate_decay[:, window].clone()
                segment_beta = beta[:, window].clone()

                positions = torch.arange(window_width, device=device).reshape(1, -1)
                positions = positions + window_start
                in_segment = (
                    positions >= prev_anchor.reshape(batch_size, 1)
                ) & (positions < cur_anchor.to(device).reshape(batch_size, 1))
                segment_beta = segment_beta * in_segment.unsqueeze(-1).to(segment_beta.dtype)
                segment_gate = segment_gate * in_segment.unsqueeze(-1).to(segment_gate.dtype)

                _, state = torch_chunk_gated_delta_rule(
                    query_states[:, window], key_states[:, window], value_states[:, window],
                    g=segment_gate, beta=segment_beta,
                    initial_state=state, output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
                prev_anchor = torch.maximum(prev_anchor, cur_anchor.to(device))

            # Every row still at anchor 0 -> the zero state.
            states[:, anchor_idx] = 0.0 if state is None else state.float()
        return states.reshape(-1, self.num_v_heads, self.head_k_dim, self.head_v_dim)

    def diffusion_blocks(self, block_hidden_states, seeds):
        if self.fixed_state:
            return self._fixed_state_blocks(block_hidden_states, seeds)
        recurrent_states, conv_lefts = seeds
        batch_size, num_anchors, block_size, _ = block_hidden_states.shape
        num_blocks = batch_size * num_anchors

        # The recurrence is per block, so blocks go on the batch dim for the kernel.
        flat_blocks = block_hidden_states.reshape(num_blocks, block_size, -1)
        mixed_qkv, gate_decay, beta = self._pre_conv(
            flat_blocks, self.in_proj_qkv_diff, self.in_proj_b_diff,
            self.in_proj_a_diff, self.dt_bias_diff, self.A_log_diff,
        )
        gate_input = self.in_proj_z_diff(flat_blocks).reshape(
            num_blocks, block_size, -1, self.head_v_dim
        )

        # Prepend the AR conv left-context along the TOKEN axis.
        seeded_stream = torch.cat([conv_lefts.to(mixed_qkv), mixed_qkv], dim=1)
        query_states, key_states, value_states = self._conv_then_split(
            seeded_stream, self.conv1d_diff, keep_len=block_size
        )

        core, _ = torch_chunk_gated_delta_rule(
            query_states, key_states, value_states, g=gate_decay, beta=beta,
            initial_state=recurrent_states.to(query_states.dtype),
            output_final_state=False, use_qk_l2norm_in_kernel=True,
        )
        core = self.norm_diff(
            core.reshape(-1, self.head_v_dim), gate_input.reshape(-1, self.head_v_dim)
        )
        return self.out_proj_diff(
            core.reshape(batch_size, num_anchors, block_size, -1)
        )


    def _fixed_state_blocks(self, block_hidden_states, seeds):
        recurrent_states, conv_lefts = seeds   # (blocks, v_heads, k_dim, v_dim) fp32
        batch_size, num_anchors, block_size, _ = block_hidden_states.shape
        num_blocks = batch_size * num_anchors
        flat_blocks = block_hidden_states.reshape(num_blocks, block_size, -1)

        # Query stream through the twin conv, left-seeded with the AR query channels.
        query_mixed = self.in_proj_q_diff(flat_blocks)
        seeded_stream = torch.cat(
            [conv_lefts[..., : self.key_dim].to(query_mixed), query_mixed], dim=1
        )
        query_states = causal_conv1d(seeded_stream, self.conv1d_diff, self.activation)
        query_states = query_states[:, -block_size:].reshape(
            num_blocks, block_size, -1, self.head_k_dim
        )
        if self.num_v_heads > self.num_k_heads:
            query_states = query_states.repeat_interleave(
                self.num_v_heads // self.num_k_heads, dim=2
            )
        # Exactly the kernel's query normalization (fp32 l2norm, then 1/sqrt(dk)): the
        # absolute scale matters, since readouts are often near norm_diff's eps.
        query_states = l2norm(query_states.float(), dim=-1, eps=1e-6) * self.head_k_dim**-0.5

        # No writes: the state at t is the prefix state decayed by the gates up to t.
        gate_decay = -self.A_log_diff.float().exp() * F.softplus(
            self.in_proj_a_diff(flat_blocks).float() + self.dt_bias_diff
        )
        cum_decay = gate_decay.cumsum(dim=1).exp()           # (blocks, block, v_heads)
        core = torch.einsum("bthk,bhkv->bthv", query_states, recurrent_states)
        core = (core * cum_decay.unsqueeze(-1)).to(block_hidden_states.dtype)

        gate_input = self.in_proj_z_diff(flat_blocks)
        core = self.norm_diff(
            core.reshape(-1, self.head_v_dim), gate_input.reshape(-1, self.head_v_dim)
        )
        return self.out_proj_diff(core.reshape(batch_size, num_anchors, block_size, -1))


class OrthrusQwen3_5DecoderLayer(nn.Module):
    """Dispatches to the attention or delta-net mixer for both passes."""

    def __init__(self, config: OrthrusQwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = OrthrusQwen3_5GatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = OrthrusQwen3_5Attention(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @property
    def mixer(self):
        return self.linear_attn if self.layer_type == "linear_attention" else self.self_attn

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        # Diffusion-pass args:
        diffusion_mode: bool = False,
        ar_context=None,
        anchors=None,
        block_pos_emb=None,
        flex_block_mask=None,
        ar_seq_len=None,        # GENERATION: anchor position; enables the mask-free path
        **kwargs,
    ):
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)

        if diffusion_mode:
            if self.layer_type == "linear_attention":
                seeds = self.linear_attn.ar_prefix_states(ar_context, anchors)
                mixed = self.linear_attn.diffusion_blocks(normed, seeds)
            else:
                # flex_block_mask is an attention-only concept; the delta-net never sees it.
                mixed = self.self_attn.diffusion_blocks(
                    normed, block_pos_emb, ar_context, flex_block_mask,
                    ar_seq_len=ar_seq_len,
                )
        elif self.layer_type == "linear_attention":
            mixed = self.linear_attn(normed, attention_mask=attention_mask, **kwargs)
        else:
            mixed, _ = self.self_attn(
                normed, position_embeddings=position_embeddings,
                attention_mask=attention_mask, **kwargs,
            )

        hidden_states = residual + mixed
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class OrthrusQwen3_5PreTrainedModel(Qwen3_5PreTrainedModel):
    config_class = OrthrusQwen3_5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OrthrusQwen3_5DecoderLayer"]
    # Qwen3.5 checkpoints ship an mtp head and a vision tower; neither is the text decoder.
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^visual.*"]

    def _init_weights(self, module):
        """Replicate the delta-net init HF keys off its own mixer class (ours is not one)."""
        super()._init_weights(module)
        if isinstance(module, OrthrusQwen3_5GatedDeltaNet):
            with torch.no_grad():
                module.dt_bias.fill_(1.0)
                module.dt_bias_diff.fill_(1.0)
                module.A_log.copy_(torch.empty_like(module.A_log).uniform_(0, 16).log_())
                module.A_log_diff.copy_(torch.empty_like(module.A_log_diff).uniform_(0, 16).log_())


class OrthrusQwen3_5Model(OrthrusQwen3_5PreTrainedModel):
    def __init__(self, config: OrthrusQwen3_5Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [OrthrusQwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.block_size = config.block_size
        self.mask_token_id = config.mask_token_id
        self.post_init()

    def _rope(self, hidden_states, position_ids):
        """3D interleaved MRoPE; the rotary module broadcasts 2D position_ids to the 3 axes."""
        return self.rotary_emb(hidden_states, position_ids)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """Plain frozen-AR forward, no cache."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0).expand(inputs_embeds.shape[0], -1)

        hidden_states = inputs_embeds
        position_embeddings = self._rope(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings,
                attention_mask=attention_mask, **kwargs,
            )
        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden_states))

    def _block_rope(self, hidden_states, anchors, block_size):
        """MRoPE for the blocks, flattened to match `diffusion_blocks`' query dim."""
        batch_size, num_anchors = anchors.shape
        offsets = torch.arange(block_size, device=anchors.device)
        positions = anchors.reshape(batch_size, num_anchors, 1) + offsets.reshape(1, 1, -1)
        return self._rope(hidden_states, positions.reshape(batch_size, -1))

    def diffusion_train_forward(self, input_ids, anchors, attention_mask=None):
        """One frozen AR pass + one batched diffusion pass over the ASCENDING anchor grid."""
        block_size = self.block_size
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        num_anchors = anchors.shape[1]

        impl = self.config._attn_implementation
        if not is_flash_attention(impl):
            raise ValueError(
                f"Orthrus training requires flash attention for the AR pass "
                f"({', '.join(FLASH_ATTENTION_IMPLS)}), got _attn_implementation={impl!r}. "
            )

        # 1) Frozen AR pass, capturing per-layer context
        for layer in self.layers:
            layer.mixer._capture_ar = True
        try:
            with torch.no_grad():
                ar_embeds = self.embed_tokens(input_ids)
                ar_positions = torch.arange(seq_len, device=device)
                ar_positions = ar_positions.unsqueeze(0).expand(batch_size, -1)
                ar_pos_emb = self._rope(ar_embeds, ar_positions)
                ar_hidden = ar_embeds
                for layer in self.layers:
                    ar_hidden = layer(
                        ar_hidden, position_embeddings=ar_pos_emb,
                        attention_mask=attention_mask,
                    )
            ar_context = {layer.layer_idx: layer.mixer._ar_cache for layer in self.layers}
        finally:
            for layer in self.layers:
                layer.mixer._capture_ar = False
                layer.mixer._ar_cache = None

        # 2) Build the diffusion blocks
        block_tokens = torch.full(
            (batch_size, num_anchors, block_size), self.mask_token_id,
            dtype=torch.long, device=device,
        )
        block_tokens[:, :, 0] = torch.gather(input_ids, 1, anchors)
        block_hidden = self.embed_tokens(block_tokens)

        flat_blocks = block_hidden.reshape(batch_size, num_anchors * block_size, -1)
        block_pos_emb = self._block_rope(flat_blocks, anchors, block_size)

        flex_block_mask = build_diffusion_block_mask(
            anchors, ar_len=seq_len, block_size=block_size
        )

        hidden = block_hidden
        for layer in self.layers:
            hidden = layer(
                hidden,
                diffusion_mode=True,
                ar_context=ar_context[layer.layer_idx],
                anchors=anchors,
                block_pos_emb=block_pos_emb,
                flex_block_mask=flex_block_mask,
            )
        return self.norm(hidden)

    @torch.no_grad()
    def diffusion_block_forward(self, prefix_ids, diff_len=None, attention_mask=None):
        block_size = diff_len or self.block_size
        device = prefix_ids.device
        batch_size, prefix_len = prefix_ids.shape
        if batch_size != 1:
            raise ValueError("diffusion_block_forward is batch-1 (single acceptance rollback).")

        # Generation: the AR pass is a full-row recompute (q_len == kv_len), so flash's
        # implicit is_causal is well-defined. eager would not mask at all.
        impl = self.config._attn_implementation
        if not is_flash_attention(impl):
            raise ValueError(
                f"Generation requires flash attention for the AR pass "
                f"({', '.join(FLASH_ATTENTION_IMPLS)}), got _attn_implementation={impl!r}."
            )

        anchor = prefix_len - 1
        anchors = torch.tensor([[anchor]], dtype=torch.long, device=device)

        for layer in self.layers:
            layer.mixer._capture_ar = True
        try:
            ar_hidden = self.embed_tokens(prefix_ids)
            ar_positions = torch.arange(prefix_len, device=device).unsqueeze(0)
            ar_pos_emb = self._rope(ar_hidden, ar_positions)
            for layer in self.layers:
                ar_hidden = layer(
                    ar_hidden, position_embeddings=ar_pos_emb, attention_mask=attention_mask
                )
            ar_context = {layer.layer_idx: layer.mixer._ar_cache for layer in self.layers}
        finally:
            for layer in self.layers:
                layer.mixer._capture_ar = False
                layer.mixer._ar_cache = None

        block_tokens = torch.full(
            (1, 1, block_size), self.mask_token_id, dtype=torch.long, device=device
        )
        block_tokens[:, :, 0] = prefix_ids[:, anchor]
        block_hidden = self.embed_tokens(block_tokens)
        block_pos_emb = self._block_rope(
            block_hidden.reshape(1, block_size, -1), anchors, block_size
        )

        hidden = block_hidden
        for layer in self.layers:
            hidden = layer(
                hidden,
                diffusion_mode=True,
                ar_context=ar_context[layer.layer_idx],
                anchors=anchors,
                block_pos_emb=block_pos_emb,
                ar_seq_len=anchor,
            )
        return self.norm(hidden).reshape(1, block_size, -1)


class OrthrusQwen3_5ForCausalLM(OrthrusQwen3_5PreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: OrthrusQwen3_5Config):
        super().__init__(config)
        self.model = OrthrusQwen3_5Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.loss_fn = None
        self.post_init()

    def forward(
        self,
        input_ids=None,
        anchors=None,
        anchor_valid=None,
        supervise_mask=None,
        attention_mask=None,
        diffusion_train: bool = True,
        is_diffusion_pass: bool = False,
        position_ids=None,
        logits_to_keep: int = 0,
        diff_len: int | None = None,
        **kwargs,
    ):
        if not diffusion_train:
            if is_diffusion_pass:
                hidden = self.model.diffusion_block_forward(
                    input_ids, diff_len=diff_len, attention_mask=attention_mask
                )
            else:
                hidden = self.model(
                    input_ids=input_ids, position_ids=position_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state
                if logits_to_keep:
                    hidden = hidden[:, -logits_to_keep:]
            return self.lm_head(hidden).float()

        from src.train.loss import block_local_labels  # local import: avoids a cycle

        hidden = self.model.diffusion_train_forward(
            input_ids, anchors, attention_mask=attention_mask
        )
        labels = block_local_labels(
            input_ids, anchors, self.model.block_size,
            supervise_mask=supervise_mask, anchor_valid=anchor_valid,
        )
        if self.loss_fn is not None:
            loss = self.loss_fn(self.lm_head.weight, hidden, labels)
        else:
            logits = self.lm_head(hidden).float()
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
            )
        return loss, hidden, labels


@torch.no_grad()
def _copy_module_params(dst: nn.Module, src: nn.Module) -> None:
    src_params = dict(src.named_parameters())
    for name, param in dst.named_parameters():
        param.copy_(src_params[name])
    src_buffers = dict(src.named_buffers())
    for name, buffer in dst.named_buffers():
        if name in src_buffers:
            buffer.copy_(src_buffers[name])


@torch.no_grad()
def copy_diff_from_ar(model: OrthrusQwen3_5ForCausalLM) -> None:
    # Warm-start every `*_diff` parameter from its frozen AR counterpart
    for layer in model.model.layers:
        if layer.layer_type == "linear_attention":
            delta_net = layer.linear_attn
            twin_pairs = (
                (delta_net.in_proj_z_diff, delta_net.in_proj_z),
                (delta_net.in_proj_a_diff, delta_net.in_proj_a),
                (delta_net.out_proj_diff, delta_net.out_proj),
                (delta_net.norm_diff, delta_net.norm),
            )
            if delta_net.fixed_state:
                # Query channels lead the [q, k, v] split of the AR projection and conv.
                key_dim = delta_net.key_dim
                delta_net.in_proj_q_diff.weight.copy_(delta_net.in_proj_qkv.weight[:key_dim])
                delta_net.conv1d_diff.weight.copy_(delta_net.conv1d.weight[:key_dim])
            else:
                twin_pairs += (
                    (delta_net.in_proj_qkv_diff, delta_net.in_proj_qkv),
                    (delta_net.in_proj_b_diff, delta_net.in_proj_b),
                    (delta_net.conv1d_diff, delta_net.conv1d),
                )
            for diff_module, ar_module in twin_pairs:
                _copy_module_params(diff_module, ar_module)
            delta_net.dt_bias_diff.copy_(delta_net.dt_bias)
            delta_net.A_log_diff.copy_(delta_net.A_log)
        else:
            attention = layer.self_attn
            twin_pairs = (
                (attention.q_proj_diff, attention.q_proj),
                (attention.k_proj_diff, attention.k_proj),
                (attention.v_proj_diff, attention.v_proj),
                (attention.o_proj_diff, attention.o_proj),
                (attention.q_norm_diff, attention.q_norm),
                (attention.k_norm_diff, attention.k_norm),
            )
            for diff_module, ar_module in twin_pairs:
                _copy_module_params(diff_module, ar_module)


def freeze_to_diffusion(model: nn.Module) -> tuple[int, int]:
    """Freeze everything except the diffusion twins. Returns (n_trainable, n_total)."""
    num_trainable = num_total = 0
    for name, param in model.named_parameters():
        num_total += 1
        param.requires_grad = any(suffix in name for suffix in DIFF_PARAM_SUFFIXES)
        num_trainable += int(param.requires_grad)
    return num_trainable, num_total
