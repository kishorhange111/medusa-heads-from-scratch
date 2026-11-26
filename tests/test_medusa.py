import itertools
import math

import pytest
import torch

transformers = pytest.importorskip("transformers")

from medusa.decoder import Decoder  # noqa: E402
from medusa.generate import greedy, medusa_generate  # noqa: E402
from medusa.heads import MedusaHeads, head_loss  # noqa: E402
from medusa.train import batches  # noqa: E402
from medusa.tree import Tree, chain, expected_tokens, optimal_tree  # noqa: E402

V = 64


@pytest.fixture(scope="module")
def dec():
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(
        vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        tie_word_embeddings=True, rope_theta=10000.0,
    )  # fmt: skip
    hf = transformers.Qwen2ForCausalLM(cfg).double().eval()
    for p in hf.parameters():
        p.data.normal_(0, 0.3)  # larger than init, so greedy outputs are not degenerate
    return Decoder(hf)


def logits(dec, h):
    return dec.lm_head(h)


def test_decoder_matches_hugging_face(dec):
    ids = torch.randint(0, V, (2, 17))
    ref = dec.hf(ids).logits
    torch.testing.assert_close(logits(dec, dec(ids)), ref, atol=1e-10, rtol=1e-10)


def test_cached_decoding_matches_full_forward(dec):
    ids = torch.randint(0, V, (1, 12))
    full = logits(dec, dec(ids))
    cache = dec.new_cache(1, 32)
    parts = [dec(ids[:, :5], cache)]
    cache.commit(5)
    for t in range(5, 12):
        parts.append(dec(ids[:, t : t + 1], cache))
        cache.commit(1)
    torch.testing.assert_close(logits(dec, torch.cat(parts, 1)), full, atol=1e-10, rtol=1e-10)


def test_tree_mask_and_depths():
    t = Tree([(0,), (1,), (0, 0), (0, 1), (1, 0), (0, 0, 0)])
    assert t.choices == [(0,), (1,), (0, 0), (0, 1), (1, 0), (0, 0, 0)]
    assert t.parent == [-1, 0, 0, 1, 1, 2, 3]
    assert t.depth == [0, 1, 1, 2, 2, 2, 3]
    assert t.path(6) == [0, 1, 3, 6]
    for i in range(len(t)):  # attends exactly to its ancestors and itself
        assert set(t.mask[i].nonzero().flatten().tolist()) == set(t.path(i))
    with pytest.raises(ValueError):
        Tree([(0, 0)])


def test_accept_picks_deepest_fully_matching_path():
    t = Tree([(0,), (1,), (0, 0), (1, 0)])  # nodes: 0 root, 1 (0,), 2 (1,), 3 (0,0), 4 (1,0)
    toks = torch.tensor([5, 7, 8, 9, 3])
    # the base model after root wants 8 (node 2), after node 2 wants 3 (node 4)
    greedy_next = torch.tensor([8, 1, 3, 2, 2])
    assert t.accept(toks, greedy_next) == 4
    greedy_next = torch.tensor([7, 1, 3, 2, 2])  # node 1 matches, but its child 3 does not
    assert t.accept(toks, greedy_next) == 1
    greedy_next = torch.tensor([6, 1, 3, 2, 2])
    assert t.accept(toks, greedy_next) == 0


def test_tree_forward_equals_running_each_path(dec):
    t = Tree([(0,), (1,), (2,), (0, 0), (0, 1), (1, 0), (0, 0, 0), (0, 1, 0)])
    prefix = torch.randint(0, V, (1, 9))
    toks = torch.randint(0, V, (len(t),))
    cache = dec.new_cache(1, 64)
    dec(prefix, cache)
    cache.commit(9)
    depth = torch.tensor(t.depth)
    hid = dec(toks[None], cache, pos=9 + depth, tree_mask=t.mask)[0]
    for i in range(len(t)):
        seq = torch.cat([prefix[0], toks[t.path(i)]])[None]
        torch.testing.assert_close(hid[i], dec(seq)[0, -1], atol=1e-10, rtol=1e-10)


def test_cache_compaction_keeps_the_accepted_path(dec):
    t = Tree([(0,), (1,), (0, 0), (1, 0), (1, 1)])
    prefix = torch.randint(0, V, (1, 6))
    toks = torch.randint(0, V, (len(t),))
    cache = dec.new_cache(1, 64)
    dec(prefix, cache)
    cache.commit(6)
    dec(toks[None], cache, pos=6 + torch.tensor(t.depth), tree_mask=t.mask)
    path = t.path(5)  # root → (1,) → (1, 1)
    cache.commit_indices(torch.tensor(path))
    nxt = torch.randint(0, V, (1, 1))
    got = dec(nxt, cache)[0, -1]
    want = dec(torch.cat([prefix[0], toks[path], nxt[0]])[None])[0, -1]
    torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("tree", [chain(1), chain(3), optimal_tree([[0.5, 0.2, 0.1]] * 3, 9)])
def test_medusa_equals_greedy_with_random_heads(dec, tree):
    torch.manual_seed(1)
    heads = MedusaHeads(32, 3).double()
    for p in heads.parameters():
        p.data.normal_(0, 0.5)
    for _ in range(3):
        ids = torch.randint(0, V, (1, 7))
        ref = greedy(dec, ids, 40)
        got = medusa_generate(dec, heads, ids, 40, tree)
        assert got.tokens == ref.tokens


@pytest.mark.parametrize("rank", [0, 1])
def test_oracle_proposals_accept_every_head(dec, rank):
    """When the right tokens are proposed, each step emits 1 + depth tokens, through
    whichever rank the tree holds them at, and the output is still exactly greedy."""
    ids = torch.randint(0, V, (1, 5))
    n = 41
    ref = greedy(dec, ids, n + 10).tokens
    K = 4
    tree = Tree([(rank,) * d for d in range(1, K + 1)] + [(1 - rank,)])

    def oracle(h_last, n_done):
        c = torch.full((K, 2), V - 1)
        for j in range(K):
            c[j, rank] = ref[n_done + 1 + j]
            c[j, 1 - rank] = (ref[n_done + 1 + j] + 1) % V  # a wrong alternative
        return c

    got = medusa_generate(dec, None, ids, n, tree, propose=oracle)
    assert got.tokens == ref[:n]
    assert got.accepted[:-1] == [K + 1] * (len(got.accepted) - 1)
    assert got.forwards == math.ceil(n / (K + 1))


def test_eos_stops_both_decoders(dec):
    ids = torch.randint(0, V, (1, 6))
    ref = greedy(dec, ids, 30).tokens
    eos = ref[10]
    g = greedy(dec, ids, 30, eos=eos).tokens
    assert g == ref[: ref.index(eos) + 1]
    heads = MedusaHeads(32, 2).double()
    assert medusa_generate(dec, heads, ids, 30, chain(2), eos=eos).tokens == g


def test_heads_start_as_the_base_head(dec):
    heads = MedusaHeads(32, 3).double()
    h = torch.randn(2, 5, 32, dtype=torch.float64)
    out = heads(h, dec.lm_head)
    for j in range(3):
        torch.testing.assert_close(out[j], dec.lm_head(h))


def test_head_j_is_trained_on_token_t_plus_2_plus_j(dec):
    torch.manual_seed(0)
    heads = MedusaHeads(32, 3).double()
    for p in heads.parameters():
        p.data.normal_(0, 0.3)
    h = torch.randn(1, 12, 32, dtype=torch.float64)
    tgt = torch.full((1, 12), -100)
    tgt[0, 9] = 17
    for j in range(3):
        loss, _, n = head_loss(heads, dec.lm_head, h, tgt, j)
        assert n == 1
        lg = heads.one(h[0, 9 - 2 - j], dec.lm_head, j)
        want = torch.nn.functional.cross_entropy(lg[None].float(), torch.tensor([17]))
        torch.testing.assert_close(loss, want)


def test_batches_count_only_reply_tokens():
    pairs = [([1, 2, 3], [4, 5]), ([6], [7, 8, 9, 10])]
    ids, tgt = next(batches(pairs, 2, pad=0, device="cpu"))
    assert ids.tolist() == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    assert tgt.tolist() == [[-100, -100, -100, 4, 5], [-100, 7, 8, 9, 10]]


def _valid_trees(K, R, size):
    nodes = [c for d in range(1, K + 1) for c in itertools.product(range(R), repeat=d)]
    for sub in itertools.combinations(nodes, size):
        s = set(sub)
        if all(len(c) == 1 or c[:-1] in s for c in sub):
            yield sub


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5])
def test_optimal_tree_beats_every_valid_tree(size):
    acc = [[0.6, 0.25, 0.1], [0.4, 0.2, 0.15], [0.3, 0.1, 0.05]]
    best = max(expected_tokens(Tree(c), acc) for c in _valid_trees(3, 3, size))
    t = optimal_tree(acc, size)
    assert len(t.choices) == size
    assert expected_tokens(t, acc) == pytest.approx(best)


class _StubTok:
    """Just enough tokenizer for ``train.distill``: a prompt is its own list of ids."""

    padding_side = "right"
    pad_token_id = V - 1
    eos_token_id = V - 2

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
        return msgs[0]["content"]

    def __call__(self, text, add_special_tokens):
        return type("Enc", (), {"input_ids": [int(t) for t in text.split()]})()


def test_distillation_is_pure_greedy_even_if_the_model_defaults_to_a_penalty(dec):
    """Qwen2.5-Instruct's generation_config sets a repetition penalty; distilled replies
    must ignore it, or the heads learn tokens the base model would not choose greedily."""
    from medusa.train import distill

    dec.hf.generation_config.repetition_penalty = 1.5
    try:
        tok = _StubTok()
        prompts = ["3 9 4 1 7", "5 5 2", "8 1 1 1 6 2"]
        out = distill(dec.hf, tok, prompts, 25, 3, lambda s: None)
        for ids, reply in out:
            ref = greedy(dec, torch.tensor([ids]), 25, eos=tok.eos_token_id).tokens
            assert reply == [t for t in ref if t != tok.pad_token_id]
    finally:
        dec.hf.generation_config.repetition_penalty = 1.0
