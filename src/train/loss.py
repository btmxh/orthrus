from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def block_local_labels(
    input_ids,
    anchors,
    block_size: int,
    supervise_mask=None,
    anchor_valid=None,
):
    device = input_ids.device
    batch_size, num_anchors = anchors.shape
    seq_len = input_ids.shape[1]

    offsets = torch.arange(block_size, device=device)
    positions = anchors.reshape(batch_size, num_anchors, 1) + offsets.reshape(1, 1, -1)
    clean_tokens = torch.gather(
        input_ids.unsqueeze(1).expand(batch_size, num_anchors, seq_len), 2, positions
    )

    labels = torch.full(
        (batch_size, num_anchors, block_size), IGNORE_INDEX, dtype=torch.long, device=device
    )
    labels[:, :, :-1] = clean_tokens[:, :, 1:]           # position t predicts token t+1

    target_positions = positions[:, :, 1:]
    keep = None
    if supervise_mask is not None:
        # Masks per (block, position), which is precisely the positions past end-of-turn.
        keep = torch.gather(
            supervise_mask.unsqueeze(1).expand(batch_size, num_anchors, seq_len),
            2, target_positions,
        ).bool()
    if anchor_valid is not None:
        slot_valid = anchor_valid.reshape(batch_size, num_anchors, 1).expand_as(
            labels[:, :, :-1]
        )
        keep = slot_valid if keep is None else (keep & slot_valid)

    if keep is not None:
        keep[0, 0, 0] = True
        labels[:, :, :-1] = labels[:, :, :-1].masked_fill(~keep, IGNORE_INDEX)
    return labels


def kl_chunked(student_hidden, teacher_hidden, weight, chunk_size: int = 1024):
    """Mean forward KL(p_teacher || p_student) over rows, both distributions being
    softmax(hidden @ weight.T). Each chunk's logits are recomputed in backward, so only one
    (chunk_size, vocab) slice is ever live. Fallback for the fused Liger kernel."""
    from torch.utils.checkpoint import checkpoint

    def chunk_kl(student, teacher):
        student_logp = F.log_softmax(F.linear(student, weight).float(), dim=-1)
        teacher_logp = F.log_softmax(F.linear(teacher, weight).float(), dim=-1)
        return (teacher_logp.exp() * (teacher_logp - student_logp)).sum()

    total = student_hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, student_hidden.shape[0], chunk_size):
        rows = slice(start, start + chunk_size)
        total = total + checkpoint(
            chunk_kl, student_hidden[rows], teacher_hidden[rows], use_reentrant=False
        )
    return total / max(1, student_hidden.shape[0])


def build_loss_fn(kind: str = "kl", use_liger: bool = True):
    """kind="kl": forward KL(p_AR || p_diff) against the frozen AR view's full distribution
    at each supervised position (the paper's objective). kind="ce": cross-entropy on the
    data tokens. The returned fn takes (lm_head_weight, hidden, labels, teacher_hidden);
    `needs_teacher` tells the model to supply the AR states at the same positions, and
    labels only mask positions for KL."""
    if kind not in ("kl", "ce"):
        raise ValueError(f"unknown loss kind {kind!r}")
    fused = None
    if use_liger:
        try:
            if kind == "kl":
                from liger_kernel.transformers import LigerFusedLinearJSD

                # jsd_beta=0 is forward KL(teacher || student), mean over non-ignored rows.
                fused = LigerFusedLinearJSD(jsd_beta=0.0, ignore_index=IGNORE_INDEX)
            else:
                from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

                fused = LigerFusedLinearCrossEntropyLoss(
                    ignore_index=IGNORE_INDEX, reduction="mean"
                )
        except Exception:
            fused = None

    def loss_fn(lm_head_weight, hidden, labels, teacher_hidden=None):
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_labels = labels.reshape(-1)
        if kind == "ce":
            if fused is not None:
                return fused(lm_head_weight, flat_hidden, flat_labels)
            logits = F.linear(flat_hidden, lm_head_weight).float()
            return F.cross_entropy(logits, flat_labels, ignore_index=IGNORE_INDEX)

        flat_teacher = teacher_hidden.reshape(-1, teacher_hidden.shape[-1]).detach()
        if fused is not None:
            return fused(flat_hidden, lm_head_weight, flat_teacher, lm_head_weight, flat_labels)
        keep = flat_labels != IGNORE_INDEX
        return kl_chunked(flat_hidden[keep], flat_teacher[keep], lm_head_weight)

    loss_fn.kind = kind
    loss_fn.needs_teacher = kind == "kl"
    loss_fn.is_fused = fused is not None
    return loss_fn
