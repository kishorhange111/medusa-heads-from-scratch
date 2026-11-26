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

There are 21, on CPU, in float64, with a tiny random Qwen2. They check:
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
- **`optimal_tree` matches brute force over every valid tree** of sizes 1–5;
- distillation is pure greedy even when the model's `generation_config` sets a repetition
  penalty.

```
pip install -e ".[dev]"
pytest -q                                    # 21 tests, CPU
pip install -e ".[train]"
python -m medusa.train --out results         # GPU: self-distil, train heads, top-k accuracy
python -m medusa.bench --out results         # GPU: greedy vs chains vs trees, wall clock
python -m medusa.numerics --out results      # GPU: decoder agreement, float16 vs float32
```

## Results (Tesla T4, float16, batch size 1)

**Setup.**
- Training data: 3,000 self-distilled Alpaca replies of up to 256 tokens each, generated
  in about 9 minutes.
- Training: 4 heads, 2 epochs (750 steps, 7.7 minutes).
- Benchmark: 20 held-out prompts, up to 128 new tokens each.
- Files: [`bench.md`](results/bench.md), [`acc.json`](results/acc.json) (per-head top-k
  accuracy), [`numerics.json`](results/numerics.json) and the
  [training log](results/train_log.txt).

| decoding | tree nodes | predicted tokens/step | tokens/step | tokens/s | speed-up | identical to greedy |
|---|---|---|---|---|---|---|
| greedy (this decoder) | – | – | 1.01 | 33.1 | 1.00× | 20/20 |
| HF `generate`, greedy | – | – | 1.00 | 34.5 | 1.04× | 20/20 |
| chain-1 | 2 | 1.31 | 1.30 | 35.9 | 1.08× | 17/20 |
| chain-4 | 5 | 1.36 | 1.38 | 37.4 | 1.13× | 18/20 |
| tree-4 | 5 | 1.53 | 1.52 | 41.5 | 1.25× | 17/20 |
| tree-8 | 9 | 1.66 | 1.70 | 44.8 | 1.35× | 18/20 |
| tree-16 | 17 | 1.78 | 1.81 | 47.4 | 1.43× | 17/20 |
| tree-32 | 33 | 1.87 | 1.92 | 48.8 | 1.47× | 18/20 |
| tree-64 | 65 | 1.94 | **2.00** | **50.8** | **1.53×** | 17/20 |

Held-out accuracy of the heads (rank 0 / top 5): head 0 (t+2) 30.6% / 56.5%, head 1 (t+3)
15.5% / 34.6%, head 2 (t+4) 10.2% / 26.7%, head 3 (t+5) 7.4% / 21.9%.

- **Trees, not depth, are where the speed-up comes from.** A chain of top-1 guesses
  saturates at 1.38 tokens per step, because head 1's top guess is right only 15% of the
  time. Spending nodes on the *alternatives* at shallow depth pays: a 64-node tree reaches
  2.00 tokens per step and 1.53× wall clock.
- **The acceptance model predicts the measurement.** `optimal_tree`'s expected tokens per
  step, computed only from the per-head accuracies, is within 4% of the measured value for
  every tree (1.94 against 2.00 at 64 nodes). Assuming each head is independent of whether
  the heads before it were right is good enough to choose trees by.
- **A tree forward is not free.** Greedy costs 30.5 ms per forward pass. A Medusa step costs
  36–39 ms: computing the heads, building the candidates and the Python acceptance loop add
  about 6 ms, and 64 tree tokens add about 3 ms more. So 2.00 tokens per step gives 1.53×,
  not 2×.
- **The heads are the limit.** Head 0 is right 31% of the time here. The Medusa paper
  reports about 60% for Vicuna-7B with far more training data. These heads saw 3,000 replies
  for two epochs, which took 8 minutes.

### Exactness, and the bug it caught

In exact arithmetic Medusa's output *is* greedy decoding, and the float64 CPU tests show
it is. On the GPU in float16, 17–18 of 20 Medusa outputs are identical to greedy. Verifying
in a tree is a different computation shape from decoding one token at a time, so rare
near-ties break differently, and after one flip the continuation diverges. This is
consistent with the precision check ([`numerics.py`](src/medusa/numerics.py)). There, a
teacher-forced forward reproduces 99.8% of float16 greedy tokens and 100% in float32, and
HF `generate` and this decoder agree on 20/20 prompts in both precisions. Medusa itself was
not benchmarked in float32.

The precision check also caught a real bug in the first run.
- **Cause:** Qwen2.5-Instruct's `generation_config` sets `repetition_penalty = 1.1`, and
  `generate` applied it.
- **Symptom:** the "greedy" distilled replies were not greedy, and the base model agreed
  with only 84–87% of the training targets, in float32 too. HF `generate` matched this
  decoder on only 2 of 20 prompts, with logit margins of 0.5–2.3 where they first differed.
  That is far too large to be rounding.
- **Fix:** every `generate` call now passes `repetition_penalty=1.0`, and a regression test
  checks that distillation equals greedy decoding when the model's default config sets a
  penalty.
- **Effect:** after the fix the base model agrees with 99.7% of the targets. Head 0's
  accuracy rose from 26% to 31%, and the 64-node speed-up from 1.52× to 1.53×.
