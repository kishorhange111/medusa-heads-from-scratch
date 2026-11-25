"""A Qwen2 forward pass with its own attention and KV cache, built from a Hugging Face model.

The embeddings, norms, MLPs, projections and RoPE are the Hugging Face modules themselves;
only attention and the cache are written here, so the cache can do what Medusa needs:

- accept an arbitrary **attention mask over the new tokens** (a tree, not just causal) with
  arbitrary **position ids** (a node's position is the prefix length plus its depth);
- **compact** after verification: of the N tree tokens just written, keep only the accepted
  path and move it to contiguous slots.

The tests check that this decoder reproduces Hugging Face's logits, with and without a
cache.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class KVCache:
    """Pre-allocated keys and values: ``(layers, batch, kv_heads, max_len, head_dim)``."""

    def __init__(self, layers, batch, kv_heads, max_len, head_dim, device, dtype):
        shape = (layers, batch, kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.len = 0  # committed length; tokens written past it are tentative

    def write(self, layer, k, v):
        n = k.shape[2]
        self.k[layer, :, :, self.len : self.len + n] = k
        self.v[layer, :, :, self.len : self.len + n] = v
        return self.k[layer, :, :, : self.len + n], self.v[layer, :, :, : self.len + n]

    def commit(self, n: int):
        self.len += n

    def commit_indices(self, idx: torch.Tensor):
        """Keep tentative tokens ``idx`` (offsets past ``len``, in order) and commit them."""
        src = self.len + idx
        dst = self.len + torch.arange(len(idx), device=idx.device)
        self.k[:, :, :, dst] = self.k[:, :, :, src]
        self.v[:, :, :, dst] = self.v[:, :, :, src]
        self.len += len(idx)


class Decoder(nn.Module):
    def __init__(self, hf):
        super().__init__()
        self.hf = hf
        c = hf.config
        self.h = c.num_attention_heads
        self.kvh = c.num_key_value_heads
        a = hf.model.layers[0].self_attn
        self.hd = a.q_proj.out_features // self.h
        self.n_layers = len(hf.model.layers)

    @property
    def lm_head(self):
        return self.hf.lm_head

    def new_cache(self, batch, max_len):
        p = self.hf.lm_head.weight
        return KVCache(self.n_layers, batch, self.kvh, max_len, self.hd, p.device, p.dtype)

    def _attn(self, layer, attn, x, cos, sin, cache, mask):
        b, n, _ = x.shape
        q = attn.q_proj(x).view(b, n, self.h, self.hd).transpose(1, 2)
        k = attn.k_proj(x).view(b, n, self.kvh, self.hd).transpose(1, 2)
        v = attn.v_proj(x).view(b, n, self.kvh, self.hd).transpose(1, 2)
        cos, sin = cos[:, None], sin[:, None]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        if cache is not None:
            k, v = cache.write(layer, k, v)
        rep = self.h // self.kvh
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return attn.o_proj(y.transpose(1, 2).reshape(b, n, -1))

    def forward(self, ids, cache=None, pos=None, tree_mask=None):
        """``ids: (b, n)`` new tokens. ``pos: (n,)`` their positions (default: consecutive
        after the cache). ``tree_mask: (n, n)`` bool, which new tokens each may attend to
        (default: causal). All cached tokens are visible. Returns post-norm hidden states
        ``(b, n, d)``; the caller commits the cache."""
        b, n = ids.shape
        past = cache.len if cache is not None else 0
        if pos is None:
            pos = torch.arange(past, past + n, device=ids.device)
        if tree_mask is None:
            tree_mask = torch.ones(n, n, dtype=torch.bool, device=ids.device).tril()
        mask = torch.cat([torch.ones(n, past, dtype=torch.bool, device=ids.device), tree_mask], 1)
        x = self.hf.model.embed_tokens(ids)
        cos, sin = self.hf.model.rotary_emb(x, pos[None].expand(b, n))
        for i, layer in enumerate(self.hf.model.layers):
            x = x + self._attn(i, layer.self_attn, layer.input_layernorm(x), cos, sin, cache, mask)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.hf.model.norm(x)


def _rotate_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat([-b, a], -1)
