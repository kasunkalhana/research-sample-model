"""Action selection helpers."""

from __future__ import annotations

from typing import Dict

import torch


def _fit_mask_to_logits(mask: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """
    Ensure mask is the same length as logits:
    - pad with zeros (illegal) if mask is shorter,
    - truncate if mask is longer,
    - move to logits' device/dtype.
    """
    m = mask.to(dtype=logits.dtype, device=logits.device).view(-1)
    A = logits.numel()
    if m.numel() < A:
        pad = torch.zeros(A - m.numel(), dtype=logits.dtype, device=logits.device)
        m = torch.cat([m, pad], dim=0)
    elif m.numel() > A:
        m = m[:A]
    return m


def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply a 0/1 mask to logits. Illegal actions get a large negative logit.
    Mask is automatically padded/truncated to match logits length.
    """
    logits = logits.clone()
    m = _fit_mask_to_logits(mask, logits)
    logits[m <= 0] = -1e9
    return logits


def select_actions(
    logits: torch.Tensor,                   # shape: [N_j, A]
    legal_masks: Dict[str, torch.Tensor],   # jid -> mask (len <=/>= A)
    node_order: Dict[str, list[str]],
    epsilon: float = 0.0,
) -> Dict[str, int]:
    """
    Greedy (with optional ε-exploration) per-junction action selection.
    Robust to:
      - fewer logits rows than junction IDs,
      - mask length != logits length.
    """
    actions: Dict[str, int] = {}

    jids = node_order.get("junction", [])
    # Bound the iteration by available logits rows.
    N = min(len(jids), logits.size(0))

    for idx in range(N):
        jid = jids[idx]
        # default to all-ones mask for this row's action count
        default_mask = torch.ones(logits.size(1), dtype=logits.dtype, device=logits.device)
        mask = legal_masks.get(jid, default_mask)
        masked = mask_logits(logits[idx], mask)

        if epsilon > 0.0 and torch.rand((), device=masked.device).item() < epsilon:
            valid = (_fit_mask_to_logits(mask, masked) > 0).nonzero(as_tuple=False).flatten()
            if valid.numel() == 0:
                actions[jid] = int(torch.argmax(masked).item())
            else:
                rand_idx = torch.randint(valid.numel(), (1,), device=masked.device).item()
                actions[jid] = int(valid[rand_idx].item())
        else:
            actions[jid] = int(torch.argmax(masked).item())

    # If there are more jids than logits rows, we simply skip the extras.
    return actions


__all__ = ["mask_logits", "select_actions"]
