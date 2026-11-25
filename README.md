# medusa-heads-from-scratch

[![CI](https://github.com/kishorhange111/medusa-heads-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/medusa-heads-from-scratch/actions/workflows/ci.yml)

**Medusa** (Cai et al. 2024) speeds up decoding without a draft model. Extra heads on the
base model guess the next few tokens, a **tree** of their top candidates is verified in
**one forward pass** with a tree-shaped attention mask, and the longest prefix the base
model agrees with is kept. This repo builds it from scratch on Qwen2.5-0.5B-Instruct. That
includes the attention and KV cache, because Medusa needs two things a stock cache cannot
do: attend through a tree mask, and *compact* the cache down to the accepted path.

```
                 ┌─ head 0 top-k ─┬─ head 1 top-k ─ …        one forward pass over all nodes,
root (greedy) ───┤                └─ …                       each sees the prefix + its ancestors
                 └─ …
accept deepest node whose whole path = the base model's greedy choices → emit 1 + depth tokens
```

## Pieces

**[`decoder.py`](src/medusa/decoder.py): the decoder and cache.** It is a Qwen2 forward
that reuses the Hugging Face model's embeddings, norms, MLPs, projections and RoPE, and has
its own attention. New tokens take **arbitrary position ids and an arbitrary mask among
themselves**. A tree node sits at position `prefix + depth` and sees only its ancestors.
`KVCache.commit_indices` keeps the accepted path out of the N tentative entries and moves it
into contiguous slots. Every node's keys were already computed at the right RoPE position,
so nothing is recomputed.

**[`heads.py`](src/medusa/heads.py): the Medusa-1 heads.** Head j reads the final hidden
state at t and predicts token t + 2 + j, through a residual block `h + SiLU(W_j h)` and the
**base model's own frozen LM head**. W_j starts at zero, so each head starts as a copy of
the next-token head. Four heads train 3.2M parameters. The targets are
**self-distilled**: the base model's own greedy replies, because acceptance depends on
agreeing with the base model, not with a dataset.

**[`tree.py`](src/medusa/tree.py): candidate trees.** A tree is a set of rank paths, for
example `(0,)`, `(1,)`, `(0, 0)`. `Tree` builds the ancestor mask, the depths and the
acceptance rule. `optimal_tree` picks the N paths with the highest acceptance probability
under an independence assumption, using measured per-head top-k accuracy. This maximises the
expected tokens per step, because extending a path never makes it more probable. `chain(k)`
is the top-1-only special case, which is speculative decoding with the heads as the draft.

**[`generate.py`](src/medusa/generate.py): greedy and Medusa decoding**, on the same
decoder and cache, so the comparison only measures the algorithm. Every emitted token is a
base-model greedy choice, so **Medusa's output equals greedy decoding** whatever the heads
propose. The heads only change how many tokens each forward pass yields.

## Tests

There are 20, on CPU, in float64, with a tiny random Qwen2. They check:
- the decoder reproduces Hugging Face's logits, with and without the cache (1e-10);
- **one tree forward equals running every root-to-node path separately**;
- after compaction, the next token's output equals a fresh forward over prefix + accepted
  path;
- **Medusa output equals greedy output exactly**, with random heads, chains and trees, and
  with EOS;
- with *oracle* proposals (the true future tokens, placed at rank 0 or at rank 1 behind a
  wrong decoy), every step accepts all 4 heads, which is 5 tokens per forward, and the
  output is still greedy;
- acceptance picks the deepest fully matching path, and the tree masks and depths are right;
- head j's loss uses token t + 2 + j, zero-initialised heads equal the base head, and only
  reply tokens count;
- **`optimal_tree` matches brute force over every valid tree** of sizes 1–5.

```
pip install -e ".[dev]"
pytest -q                                    # 20 tests, CPU
pip install -e ".[train]"
python -m medusa.train --out results         # GPU: self-distil, train heads, top-k accuracy
python -m medusa.bench --out results         # GPU: greedy vs chains vs trees, wall clock
```

## Results

**Pending** the Colab T4 run.
