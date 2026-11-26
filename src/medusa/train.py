"""Self-distil, train Medusa-1 heads, and measure per-head top-k accuracy (GPU, resumable).

    python -m medusa.train --out results

Stages, each saved to ``--out`` and skipped when its file exists:
1. ``distill.pt``: the base model's greedy replies (``--max-new`` tokens) to Alpaca
   instructions, generated in batches. Prompts are split by a fixed shuffle into train,
   accuracy and benchmark sets that never overlap.
2. ``heads.pt``: K heads trained on the train replies with the base model frozen. The loss
   only counts reply tokens. Head j's loss is weighted 0.8^j.
3. ``acc.json``: ``acc[j][r]``, the fraction of held-out reply positions where head j's
   rank-r token is the target. This feeds ``tree.optimal_tree``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch

from .heads import MedusaHeads, head_loss

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_prompts():
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    out = []
    for r in ds:
        p = r["instruction"] + (("\n\n" + r["input"]) if r["input"] else "")
        out.append(p)
    return out


def splits(prompts, n_train, n_acc=200, n_bench=100, seed=0):
    idx = list(range(len(prompts)))
    random.Random(seed).shuffle(idx)
    bench = idx[:n_bench]
    acc = idx[n_bench : n_bench + n_acc]
    train = idx[n_bench + n_acc : n_bench + n_acc + n_train]
    return [prompts[i] for i in train], [prompts[i] for i in acc], [prompts[i] for i in bench]


def chat_ids(tok, prompt):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False).input_ids


@torch.no_grad()
def distill(model, tok, prompts, max_new, bs, log, deadline=None):
    """Greedy replies; returns a list of (prompt ids, reply ids)."""
    tok.padding_side = "left"
    out = []
    for i in range(0, len(prompts), bs):
        if deadline and time.time() > deadline:
            log(f"distill: time budget reached after {len(out)} prompts")
            break
        chunk = [chat_ids(tok, p) for p in prompts[i : i + bs]]
        L = max(map(len, chunk))
        ids = torch.full((len(chunk), L), tok.pad_token_id)
        att = torch.zeros_like(ids)
        for r, c in enumerate(chunk):
            ids[r, L - len(c) :] = torch.tensor(c)
            att[r, L - len(c) :] = 1
        g = model.generate(
            input_ids=ids.to(model.device),
            attention_mask=att.to(model.device),
            do_sample=False,
            max_new_tokens=max_new,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,  # Qwen2.5-Instruct defaults to a penalty
            pad_token_id=tok.pad_token_id,
        )
        for r, c in enumerate(chunk):
            reply = g[r, L:].tolist()
            if tok.eos_token_id in reply:
                reply = reply[: reply.index(tok.eos_token_id) + 1]
            reply = [t for t in reply if t != tok.pad_token_id]
            out.append((c, reply))
        log(f"distill {len(out)}/{len(prompts)}")
    return out


def batches(pairs, bs, pad, device, shuffle_seed=None):
    order = list(range(len(pairs)))
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(order)
    for i in range(0, len(order), bs):
        chunk = [pairs[j] for j in order[i : i + bs]]
        L = max(len(p) + len(r) for p, r in chunk)
        ids = torch.full((len(chunk), L), pad)
        tgt = torch.full((len(chunk), L), -100)
        for k, (p, r) in enumerate(chunk):
            ids[k, : len(p) + len(r)] = torch.tensor(p + r)
            tgt[k, len(p) : len(p) + len(r)] = torch.tensor(r)
        yield ids.to(device), tgt.to(device)


def hidden(model, ids):
    return model.model(input_ids=ids).last_hidden_state


def train_heads(model, pairs, n_heads, epochs, bs, lr, log, pad, deadline=None):
    d = model.config.hidden_size
    heads = MedusaHeads(d, n_heads).to(model.device)
    opt = torch.optim.AdamW(heads.parameters(), lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=model.device.type == "cuda")
    steps = epochs * math.ceil(len(pairs) / bs)
    step, t0 = 0, time.time()
    for ep in range(epochs):
        for ids, tgt in batches(pairs, bs, pad, model.device, shuffle_seed=ep):
            if deadline and time.time() > deadline:
                log(f"train: time budget reached at step {step}/{steps}")
                return heads
            step += 1
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, step / 50) * 0.5 * (1 + math.cos(math.pi * step / steps))
            with torch.no_grad():
                h = hidden(model, ids)
            accs = []
            for j in range(n_heads):
                loss, c, n = head_loss(heads, model.lm_head, h, tgt, j)
                scaler.scale(0.8**j * loss).backward()
                accs.append(c / max(1, n))
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if step % 25 == 0 or step == steps:
                a = " ".join(f"{x:.3f}" for x in accs)
                log(f"step {step}/{steps} loss {loss.item():.3f} top1 {a} {time.time() - t0:.0f}s")
    return heads


@torch.no_grad()
def topk_accuracy(model, heads, pairs, bs, pad, R=10):
    """``acc[j][r]``: fraction of positions where head j's rank-r token is the target."""
    K = len(heads)
    hits = torch.zeros(K, R)
    counts = torch.zeros(K)
    base, base_n = torch.zeros(R), 0
    for ids, tgt in batches(pairs, bs, pad, model.device):
        h = hidden(model, ids)
        for j in range(K):
            s = 2 + j
            top = heads.one(h[:, :-s], model.lm_head, j).topk(R, -1).indices
            t = tgt[:, s:]
            m = t != -100
            hits[j] += (top[m] == t[m][:, None]).float().sum(0).cpu()
            counts[j] += m.sum().cpu()
        t = tgt[:, 1:]
        m = t != -100
        top = model.lm_head(h[:, :-1]).topk(R, -1).indices
        base += (top[m] == t[m][:, None]).float().sum(0).cpu()
        base_n += int(m.sum())
    # The base head on its own greedy replies should be ~100% at rank 0: a sanity check
    # that prompts, padding and targets line up.
    return (hits / counts[:, None]).tolist(), (base / max(1, base_n)).tolist()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--gen-bs", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--distill-min", type=float, default=1e9, help="time budget for stage 1")
    ap.add_argument("--train-min", type=float, default=1e9, help="time budget for stage 2")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "train_log.txt"), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    tok = AutoTokenizer.from_pretrained(a.model)
    dtype = torch.float16 if a.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(a.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    train_p, acc_p, _ = splits(load_prompts(), a.n_train)
    dpath = os.path.join(a.out, "distill.pt")
    if os.path.exists(dpath):
        data = torch.load(dpath)
    else:
        deadline = time.time() + 60 * a.distill_min
        data = {
            "acc": distill(model, tok, acc_p, a.max_new, a.gen_bs, log),
            "train": distill(model, tok, train_p, a.max_new, a.gen_bs, log, deadline),
        }
        torch.save(data, dpath)
    log(f"distilled: {len(data['train'])} train, {len(data['acc'])} held-out replies")
    hpath = os.path.join(a.out, "heads.pt")
    pad = tok.pad_token_id
    if os.path.exists(hpath):
        heads = MedusaHeads(model.config.hidden_size, a.heads).to(a.device)
        heads.load_state_dict(torch.load(hpath, map_location=a.device))
    else:
        deadline = time.time() + 60 * a.train_min
        heads = train_heads(model, data["train"], a.heads, a.epochs, a.bs, a.lr, log, pad, deadline)
        torch.save(heads.state_dict(), hpath)
    acc, base = topk_accuracy(model, heads, data["acc"], a.bs, pad)
    with open(os.path.join(a.out, "acc.json"), "w") as f:
        json.dump({"heads": acc, "base_next_token": base}, f, indent=1)
    for j, row in enumerate(acc):
        log(f"head {j} (t+{j + 2}): top1 {row[0]:.3f} top5 {sum(row[:5]):.3f}")


if __name__ == "__main__":
    main()
