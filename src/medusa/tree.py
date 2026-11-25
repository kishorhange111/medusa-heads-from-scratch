"""Candidate trees for Medusa decoding.

A tree is a list of *choices*: tuples of ranks, one per depth. ``(0,)`` is head 0's top
token; ``(0, 2)`` is head 1's third-ranked token placed after it. Node 0 is the root, the
token the base model has already chosen. A node at depth d is head (d-1)'s token at rank
``choice[-1]``. As in Medusa-1, every head proposes from the same hidden state, so a token's
candidates at depth d do not depend on its parent.

All nodes are verified in **one forward pass**: each attends to the cached prefix and to its
own ancestors (``mask``), at position ``prefix + depth``. Node i is accepted when its token
equals the base model's greedy choice after its parent, and so is every ancestor. The
deepest accepted node wins, so the output is exactly what greedy decoding would produce.
"""

from __future__ import annotations

import itertools
import math

import torch


class Tree:
    def __init__(self, choices):
        choices = sorted({tuple(c) for c in choices}, key=lambda c: (len(c), c))
        for c in choices:
            if len(c) > 1 and c[:-1] not in choices:
                raise ValueError(f"choice {c} has no parent {c[:-1]}")
        self.choices = choices
        index = {c: i + 1 for i, c in enumerate(choices)}
        n = len(choices) + 1
        self.parent = [-1] + [index.get(c[:-1], 0) for c in choices]
        self.depth = [0] + [len(c) for c in choices]
        self.head = [-1] + [len(c) - 1 for c in choices]  # which head proposes the node
        self.rank = [-1] + [c[-1] for c in choices]
        mask = torch.zeros(n, n, dtype=torch.bool)
        for i in range(n):
            j = i
            while j >= 0:
                mask[i, j] = True
                j = self.parent[j]
        self.mask = mask
        self.n_heads = max(self.depth)
        self.max_rank = max(self.rank) + 1 if choices else 0

    def __len__(self):
        return len(self.parent)

    def path(self, i: int) -> list[int]:
        out = []
        while i >= 0:
            out.append(i)
            i = self.parent[i]
        return out[::-1]

    def tokens(self, root: int, cands: torch.Tensor) -> torch.Tensor:
        """``cands: (n_heads, max_rank)`` proposals → token of every node, root first."""
        toks = [root] + [
            int(cands[h, r]) for h, r in zip(self.head[1:], self.rank[1:], strict=True)
        ]
        return torch.tensor(toks, device=cands.device)

    def accept(self, toks: torch.Tensor, greedy: torch.Tensor) -> int:
        """``greedy[i]`` is the base model's argmax after node i. Returns the deepest node
        whose whole path is accepted (0 = only the root)."""
        ok = (toks[1:] == greedy[torch.tensor(self.parent[1:], device=toks.device)]).tolist()
        good = [True] + [False] * (len(self) - 1)
        best = 0
        for i in range(1, len(self)):
            good[i] = ok[i - 1] and good[self.parent[i]]
            if good[i] and self.depth[i] > self.depth[best]:
                best = i
        return best


def chain(n_heads: int) -> Tree:
    """Top-1 of every head in sequence: the speculative-decoding special case."""
    return Tree([(0,) * d for d in range(1, n_heads + 1)])


def optimal_tree(acc, size: int) -> Tree:
    """Best tree with ``size`` non-root nodes under an independence assumption.

    ``acc[d][r]`` is the probability that head d's rank-r token is the right one (measured
    on held-out data). A path's chance of being accepted is the product along it, and the
    expected number of tokens per step is 1 plus the sum over nodes. Extending a path never
    raises its probability, so the ``size`` most probable paths always form a valid tree
    (ties go to shorter paths), and that set maximises the expectation.
    """
    K, R = len(acc), len(acc[0])
    scored = []
    for d in range(1, K + 1):
        for c in itertools.product(range(R), repeat=d):
            p = math.prod(acc[j][r] for j, r in enumerate(c))
            scored.append((-p, d, c))
    scored.sort()
    return Tree([c for _, _, c in scored[:size]])


def expected_tokens(tree: Tree, acc) -> float:
    return 1.0 + sum(math.prod(acc[j][r] for j, r in enumerate(c)) for c in tree.choices)
