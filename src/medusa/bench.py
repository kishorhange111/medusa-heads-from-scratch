"""Wall-clock benchmark of greedy vs Medusa decoding (GPU, batch size 1).

    python -m medusa.bench --out results     # after medusa.train

On held-out Alpaca prompts this runs greedy decoding on the same decoder, chain decoding
with 1–4 heads, and trees from ``optimal_tree`` built from the measured head accuracies.
For each it records tokens per forward pass, tokens per second and whether the output is
identical to greedy. Hugging Face ``generate`` (greedy) is timed as a reference.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .decoder import Decoder
from .generate import greedy, medusa_generate
from .heads import MedusaHeads
from .train import MODEL, chat_ids, load_prompts, splits
from .tree import chain, expected_tokens, optimal_tree


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def report(rows) -> str:
    g = next(r for r in rows if r["config"] == "greedy")
    out = [
        "| decoding | tree nodes | predicted tokens/step | tokens/step | tokens/s | speed-up"
        " | identical to greedy |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        pred = f"{r['predicted']:.2f}" if r.get("predicted") else "–"
        out.append(
            f"| {r['config']} | {r.get('nodes', '–')} | {pred} | {r['tokens_per_step']:.2f} "
            f"| {r['tok_s']:.1f} | {r['tok_s'] / g['tok_s']:.2f}× "
            f"| {r['identical']}/{r['n']} |"
        )
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--n", type=int, default=20, help="benchmark prompts")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--sizes", default="4,8,16,32,64")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    dtype = torch.float16 if a.device == "cuda" else torch.float32
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(a.device).eval()
    dec = Decoder(hf)
    heads = MedusaHeads(hf.config.hidden_size, a.heads).to(a.device)
    heads.load_state_dict(torch.load(os.path.join(a.out, "heads.pt"), map_location=a.device))
    acc = json.load(open(os.path.join(a.out, "acc.json")))["heads"]
    _, _, bench_p = splits(load_prompts(), 0)
    prompts = [torch.tensor([chat_ids(tok, p)], device=a.device) for p in bench_p[: a.n]]
    eos = tok.convert_tokens_to_ids("<|im_end|>")

    configs = [(f"chain-{k}", chain(k)) for k in range(1, a.heads + 1)]
    configs += [(f"tree-{s}", optimal_tree(acc, s)) for s in map(int, a.sizes.split(","))]
    path = os.path.join(a.out, "bench.jsonl")
    rows = []

    def timed(fn):
        sync()
        t = time.time()
        r = fn()
        sync()
        return r, time.time() - t

    greedy(dec, prompts[0], 8, eos)  # warm-up
    ref, total_t, total_tok, steps = [], 0.0, 0, 0
    for ids in prompts:
        r, t = timed(lambda ids=ids: greedy(dec, ids, a.max_new, eos))
        ref.append(r.tokens)
        total_t, total_tok, steps = total_t + t, total_tok + len(r.tokens), steps + r.forwards
    rows.append({"config": "greedy", "tokens_per_step": total_tok / steps,
                 "tok_s": total_tok / total_t, "identical": len(prompts), "n": len(prompts),
                 "tokens": total_tok})  # fmt: skip

    hf_t, hf_tok, hf_same = 0.0, 0, 0
    for ids, rt in zip(prompts, ref, strict=True):
        g, t = timed(
            lambda ids=ids: hf.generate(
                ids,
                do_sample=False,
                max_new_tokens=a.max_new,
                eos_token_id=eos,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id,
            )  # fmt: skip
        )
        new = g[0, ids.shape[1] :].tolist()
        hf_t, hf_tok, hf_same = hf_t + t, hf_tok + len(new), hf_same + (new == rt)
    rows.append({"config": "HF generate (greedy)", "tokens_per_step": 1.0,
                 "tok_s": hf_tok / hf_t, "identical": hf_same, "n": len(prompts),
                 "tokens": hf_tok})  # fmt: skip

    for name, tree in configs:
        total_t, total_tok, steps, same = 0.0, 0, 0, 0
        for ids, rt in zip(prompts, ref, strict=True):
            r, t = timed(
                lambda ids=ids, tree=tree: medusa_generate(dec, heads, ids, a.max_new, tree, eos)
            )
            total_t, total_tok, steps = total_t + t, total_tok + len(r.tokens), steps + r.forwards
            same += r.tokens == rt
        rows.append({"config": name, "nodes": len(tree), "predicted": expected_tokens(tree, acc),
                     "tokens_per_step": total_tok / steps, "tok_s": total_tok / total_t,
                     "identical": same, "n": len(prompts), "tokens": total_tok})  # fmt: skip
        print(json.dumps(rows[-1]), flush=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    gpu = torch.cuda.get_device_name() if a.device == "cuda" else "none (CPU)"
    text = f"GPU: {gpu}. {len(prompts)} prompts, up to {a.max_new} new tokens each.\n\n"
    text += report(rows)
    with open(os.path.join(a.out, "bench.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
