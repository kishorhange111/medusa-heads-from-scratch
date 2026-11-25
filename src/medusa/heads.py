"""Medusa heads (Cai et al. 2024): extra decoding heads on the frozen base model.

Head j (j = 0..K-1) reads the base model's final hidden state at position t and predicts
token t + 2 + j; the base LM head already predicts t + 1. Each head is a residual block
``h + SiLU(W_j h)`` followed by the **base model's own LM head** (shared and frozen). W_j
starts at zero, so every head starts as a copy of the base next-token head. Only the W_j
train (``K · d²`` parameters, 3.2M for Qwen2.5-0.5B with K = 4). Sharing the LM head avoids
K extra 136M-parameter vocab matrices.

Training is Medusa-1: the base stays frozen, and the targets come from **self-distillation**,
i.e. the base model's own greedy continuations. A head is only useful when it predicts what
the base model *would* emit, not what a dataset says.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MedusaHeads(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.w = nn.ModuleList(nn.Linear(d, d) for _ in range(n_heads))
        for lin in self.w:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def __len__(self):
        return len(self.w)

    def one(self, h, lm_head, j: int):
        w = self.w[j]
        return lm_head(h + F.silu(w(h.to(w.weight.dtype))).to(h.dtype))

    def forward(self, h, lm_head):
        """``h: (..., d)`` post-norm hidden states → logits ``(K, ..., vocab)``."""
        return torch.stack([self.one(h, lm_head, j) for j in range(len(self))])


def head_loss(heads, lm_head, h, targets, j: int):
    """Head j at position t is trained on ``targets[t + 2 + j]``. ``h: (b, L, d)`` are the
    hidden states of the sequence; ``-100`` in targets is ignored. Returns the loss and the
    number of correct top-1 predictions and of counted positions."""
    s = 2 + j
    lg, tgt = heads.one(h[:, :-s], lm_head, j), targets[:, s:]
    loss = F.cross_entropy(lg.float().flatten(0, 1), tgt.flatten(), ignore_index=-100)
    m = tgt != -100
    return loss, int((lg.argmax(-1)[m] == tgt[m]).sum()), int(m.sum())


def medusa_loss(heads, lm_head, h, targets, decay: float = 0.8):
    """Sum over heads of ``decay**j`` × head j's loss, and per-head top-1 accuracy."""
    total, accs = 0.0, []
    for j in range(len(heads)):
        loss, c, n = head_loss(heads, lm_head, h, targets, j)
        total = total + decay**j * loss
        accs.append(c / max(1, n))
    return total, accs
