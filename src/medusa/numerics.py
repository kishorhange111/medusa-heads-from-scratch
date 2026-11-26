"""Precision check: how often do 'identical' greedy decoders disagree, and by how much?

    python -m medusa.numerics --out results

In exact arithmetic, Hugging Face ``generate``, this repo's greedy decoder, Medusa decoding
and a teacher-forced forward over a greedy reply all pick the same tokens. In floating
point they compute logits in different shapes (one token at a time, a whole sequence at
once, a tree, a padded batch), so near-ties can break differently, and after one flip the
continuations diverge. This script measures that, in float16 and float32, on the benchmark
prompts:

- **decode agreement:** HF ``generate`` (batch 1) against this repo's greedy decoder;
  identical outputs, and the base model's top-1/top-2 logit margin where they first differ;
- **teacher-forced agreement:** the fraction of reply tokens that are the argmax of a
  single forward over prompt + reply, for replies from batch-1 generation and from
  left-padded batched generation (the way ``train.distill`` makes its data).
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from .decoder import Decoder
from .generate import greedy
from .train import MODEL, chat_ids, distill, load_prompts, splits

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def teacher_forced(hf, prompt, reply):
    ids = torch.tensor([prompt + reply], device=hf.device)
    pred = hf(ids).logits[0, len(prompt) - 1 : -1].argmax(-1).tolist()
    return sum(p == r for p, r in zip(pred, reply, strict=True)), len(reply)


@torch.no_grad()
def margin_at(hf, prompt, prefix, a, b):
    """Logit of token a minus token b after prompt + prefix."""
    lg = hf(torch.tensor([prompt + prefix], device=hf.device)).logits[0, -1].float()
    return float(lg[a] - lg[b])


def run(name, dtype, tok, prompts, max_new, eos, log):
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(DEV).eval()
    dec = Decoder(hf)
    same, margins, tf = 0, [], {"batch1": [0, 0], "batched": [0, 0]}
    ids_list = [chat_ids(tok, p) for p in prompts]
    batched = distill(hf, tok, prompts, max_new, len(prompts), lambda s: None)
    for ids, (_, rb) in zip(ids_list, batched, strict=True):
        x = torch.tensor([ids], device=DEV)
        g = hf.generate(
            x, do_sample=False, max_new_tokens=max_new, eos_token_id=eos, temperature=None,
            top_p=None, top_k=None, repetition_penalty=1.0, pad_token_id=tok.pad_token_id,
        )[0, len(ids) :].tolist()  # fmt: skip
        mine = greedy(dec, x, max_new, eos).tokens
        if g == mine:
            same += 1
        else:
            i = next((k for k, (u, v) in enumerate(zip(g, mine, strict=False)) if u != v), None)
            if i is not None:
                margins.append(abs(margin_at(hf, ids, g[:i], g[i], mine[i])))
        for key, rep in (("batch1", g), ("batched", rb)):
            c, n = teacher_forced(hf, ids, rep)
            tf[key][0] += c
            tf[key][1] += n
    gc = hf.generation_config
    out = {
        "dtype": str(dtype).replace("torch.", ""),
        "default_generation_config": {
            k: getattr(gc, k, None) for k in ("repetition_penalty", "temperature", "top_p", "top_k")
        },
        "hf_vs_mine_identical": same,
        "n": len(prompts),
        "margins_at_first_divergence": sorted(round(m, 4) for m in margins),
        "teacher_forced_batch1_replies": tf["batch1"][0] / max(1, tf["batch1"][1]),
        "teacher_forced_batched_replies": tf["batched"][0] / max(1, tf["batched"][1]),
    }
    log(json.dumps(out))
    del hf, dec
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=128)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    _, _, bench_p = splits(load_prompts(), 0)
    os.makedirs(a.out, exist_ok=True)
    rows = [
        run(a.model, dt, tok, bench_p[: a.n], a.max_new, eos, print)
        for dt in (torch.float16, torch.float32)
    ]
    with open(os.path.join(a.out, "numerics.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
