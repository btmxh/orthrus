"""Lossless Orthrus generation on the Qwen3.5 hybrid: the diffusion view drafts a block, the
frozen AR view verifies it, and the accepted prefix is committed to a cache shared by both.

Batch size 1. Mirrors the acceptance logic of src/model.py's generate: greedy prefix match
at temperature 0, speculative rejection sampling otherwise, so the output follows the AR
model's distribution exactly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class HybridCache:
    """Per-layer state of the committed prefix: (keys, values) for attention layers, and
    (recurrent state, conv left-context) for delta-net layers. A forward over new tokens
    leaves them pending; commit(n) keeps the first n (the accepted prefix)."""

    def __init__(self, model):
        self.layers = list(model.model.layers)
        self.entries = [None] * len(self.layers)
        self.pending = [None] * len(self.layers)
        self.length = 0

    def commit(self, num_tokens: int) -> None:
        for layer in self.layers:
            idx = layer.layer_idx
            self.entries[idx] = layer.mixer.commit(self.entries[idx], self.pending[idx], num_tokens)
            self.pending[idx] = None
        self.length += num_tokens


def _sample(logits, temperature, top_k, top_p):
    if temperature < 1e-5:
        return logits.argmax(dim=-1), None
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., [-1]]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        logits = logits.masked_fill(remove.scatter(-1, sorted_indices, remove), -float("inf"))
    probs = F.softmax(logits, dim=-1)
    tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.shape[:-1])
    return tokens, probs


@torch.inference_mode()
def generate(
    model,
    input_ids: torch.LongTensor,
    max_new_tokens: int | None = None,
    max_length: int | None = None,
    temperature: float = 0.0,
    top_k: int = 20,
    top_p: float = 0.8,
    eos_token_id: int | None = None,
    streamer=None,
    use_diffusion_mode: bool = True,
    return_stats: bool = False,
):
    """input_ids: (1, prompt_len). Returns output ids (1, len), plus with return_stats a dict
    with `accepted`: drafts accepted per verification step (all 0 without diffusion)."""
    if input_ids.shape[0] != 1:
        raise ValueError("generate is batch-1.")
    if (max_new_tokens is None) == (max_length is None):
        raise ValueError("Pass exactly one of max_new_tokens / max_length.")
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id
    inner = model.model
    device = input_ids.device
    prompt_len = input_ids.shape[1]
    max_length = max_length or prompt_len + max_new_tokens
    # Without diffusion the same loop verifies one token at a time: plain cached AR decoding.
    block_size = model.config.block_size if use_diffusion_mode else 1
    accepted_per_step = []

    def finish(output):
        if streamer:
            streamer.end()
        return (output, {"accepted": accepted_per_step}) if return_stats else output

    output_ids = input_ids.new_full((1, max_length), model.config.mask_token_id)
    output_ids[:, :prompt_len] = input_ids
    if streamer:
        streamer.put(input_ids)

    # Prefill: commit the prompt; its last logits give the first anchor token.
    cache = HybridCache(model)
    hidden = inner.forward_cached(input_ids, cache)
    cache.commit(prompt_len)
    next_token, _ = _sample(model.lm_head(hidden[:, -1]).float(), temperature, top_k, top_p)
    start_idx = prompt_len
    output_ids[:, start_idx] = next_token
    if streamer:
        streamer.put(next_token)
    if next_token.item() == eos_token_id:
        return finish(output_ids[:, : start_idx + 1])

    # Invariant: tokens [0, start_idx) are committed; output_ids[start_idx] is the anchor,
    # sampled from the AR view but not yet run through it.
    while start_idx < max_length - 1:
        diff_len = min(block_size, max_length - start_idx)
        anchor = output_ids[:, start_idx : start_idx + 1]
        if diff_len > 1:
            draft_hidden = inner.diffusion_draft(anchor, cache, diff_len)
            draft_logits = model.lm_head(draft_hidden[:, :-1]).float()
            draft_tokens, draft_probs = _sample(draft_logits, temperature, top_k, top_p)
        else:
            draft_tokens = anchor.new_empty((1, 0))
            draft_probs = None
        proposed = torch.cat([anchor, draft_tokens], dim=1)

        # Verify: the AR view over [anchor, drafts] from the committed prefix.
        ar_hidden = inner.forward_cached(proposed, cache)
        ar_tokens, ar_probs = _sample(
            model.lm_head(ar_hidden).float(), temperature, top_k, top_p
        )

        num_drafts = draft_tokens.shape[1]
        if temperature < 1e-5:
            matches = draft_tokens == ar_tokens[:, :-1]
            acceptance_len = int(matches.cumprod(dim=1).sum(dim=1)[0])
            next_token = ar_tokens[:, acceptance_len]
        else:
            acceptance_len = 0
            for i in range(num_drafts):
                token = draft_tokens[0, i]
                q_prob = draft_probs[0, i, token]
                p_prob = ar_probs[0, i, token]
                if torch.rand(1, device=device).item() < min(1.0, (p_prob / max(q_prob, 1e-8)).item()):
                    acceptance_len += 1
                else:
                    break
            p_dist = ar_probs[0, acceptance_len]
            if acceptance_len < num_drafts:
                residual = torch.clamp(p_dist - draft_probs[0, acceptance_len], min=0.0)
                residual_sum = residual.sum()
                next_token = torch.multinomial(
                    residual / residual_sum if residual_sum > 1e-5 else p_dist, 1
                )
            else:
                next_token = torch.multinomial(p_dist, 1)
        accepted_per_step.append(acceptance_len)

        accepted = proposed[:, : acceptance_len + 1]
        eos_positions = (accepted == eos_token_id).nonzero()
        if len(eos_positions) > 0:
            eos_offset = int(eos_positions[0, -1])
            output_ids[:, start_idx : start_idx + eos_offset + 1] = accepted[:, : eos_offset + 1]
            if streamer:
                streamer.put(accepted[:, 1 : eos_offset + 1])
            return finish(output_ids[:, : start_idx + eos_offset + 1])

        cache.commit(acceptance_len + 1)
        end_idx = start_idx + acceptance_len + 1
        output_ids[:, start_idx:end_idx] = accepted
        if streamer and acceptance_len > 0:
            streamer.put(accepted[:, 1:])
        start_idx = end_idx

        if start_idx < max_length:
            output_ids[:, start_idx] = next_token
            if streamer:
                streamer.put(next_token.view(1, -1))
            if next_token.item() == eos_token_id:
                return finish(output_ids[:, : start_idx + 1])

    return finish(output_ids[:, :max_length])
