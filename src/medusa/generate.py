"""Greedy decoding and Medusa tree decoding on the same decoder and cache (batch size 1).

One Medusa step:
1. The root is the token the base model already chose; the heads, applied to the hidden
   state that chose it, propose ``cands[head, rank]``.
2. The tree's tokens are written into the cache *tentatively* and run in one forward pass
   with the tree mask and depth-based positions.
3. The deepest node whose whole path matches the base model's own greedy choices is
   accepted. The cache keeps the root plus that path, moved into contiguous slots, and the
   base model's choice after the last accepted node becomes the next root.

Every emitted token is a base-model greedy choice, so the output equals greedy decoding.
The heads only decide how many tokens each forward pass yields.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .tree import Tree


@dataclass
class Result:
    tokens: list[int]
    forwards: int  # decoder calls after the prefill
    accepted: list[int] = field(default_factory=list)  # tokens emitted per step

    @property
    def tokens_per_step(self) -> float:
        return len(self.tokens) / max(1, self.forwards)


def _trim(tokens, n, eos):
    tokens = tokens[:n]
    if eos is not None and eos in tokens:
        tokens = tokens[: tokens.index(eos) + 1]
    return tokens


@torch.no_grad()
def greedy(dec, ids: torch.Tensor, n: int, eos: int | None = None) -> Result:
    cache = dec.new_cache(1, ids.shape[1] + n + 1)
    h = dec(ids, cache)
    cache.commit(ids.shape[1])
    out, steps = [], 0
    tok = int(dec.lm_head(h[0, -1]).argmax())
    while True:
        out.append(tok)
        if len(out) >= n or tok == eos:
            break
        h = dec(torch.tensor([[tok]], device=ids.device), cache)
        cache.commit(1)
        steps += 1
        tok = int(dec.lm_head(h[0, -1]).argmax())
    return Result(_trim(out, n, eos), steps, [1] * steps)


@torch.no_grad()
def medusa_generate(dec, heads, ids, n, tree: Tree, eos=None, propose=None) -> Result:
    """``propose(h_last, n_done) -> (n_heads, max_rank)`` overrides the heads (for tests)."""
    dev = ids.device
    cache = dec.new_cache(1, ids.shape[1] + n + len(tree) + 1)
    h = dec(ids, cache)
    cache.commit(ids.shape[1])
    h_last = h[0, -1]
    root = int(dec.lm_head(h_last).argmax())
    mask = tree.mask.to(dev)
    depth = torch.tensor(tree.depth, device=dev)
    out, res = [], Result([], 0)
    while len(out) < n and (eos is None or eos not in out):
        if propose is not None:
            cands = propose(h_last, len(out))
        else:
            cands = heads(h_last, dec.lm_head)[: tree.n_heads].topk(tree.max_rank, -1).indices
        toks = tree.tokens(root, cands)
        hid = dec(toks[None], cache, pos=cache.len + depth, tree_mask=mask)[0]
        res.forwards += 1
        greedy_next = dec.lm_head(hid).argmax(-1)
        best = tree.accept(toks, greedy_next)
        path = tree.path(best)
        out.extend(toks[path].tolist())
        res.accepted.append(len(path))
        cache.commit_indices(torch.tensor(path, device=dev))
        root, h_last = int(greedy_next[best]), hid[best]
    res.tokens = _trim(out, n, eos)
    return res
