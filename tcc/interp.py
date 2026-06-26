"""
TransformerLens helpers for E2 -- induction-head attribution. This is the heaviest module
(imports transformer_lens); kept separate from the generation harness so payloads/scoring
stay torch-free.

Two pieces:
  induction_scores : per-head induction score on a repeated-random sequence (the standard
                     Olsson detector) -- identifies which heads are induction heads.
  ablate_heads_hook: zero the output of named (layer, head) pairs, for the causal test
                     that reproduction is carried by those heads.
"""
from __future__ import annotations
import numpy as np


def load_hooked(name="gpt2", device=None):
    import torch
    from transformer_lens import HookedTransformer
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    model = HookedTransformer.from_pretrained(name, device=device)
    model.eval()
    return model


def induction_scores(model, seq_len=50, seed=0):
    """Per-head induction score. Feed [BOS, r, r] where r is a random token sequence; at each
    second-copy position i the current token last occurred at i-seq_len, and an induction head
    attends to the token *after* that occurrence (position i-seq_len+1) to copy it. The score
    is the mean attention on that stripe. Returns array [n_layers, n_heads]."""
    import torch
    rng = np.random.default_rng(seed)
    V = model.cfg.d_vocab
    r = rng.integers(1000, V - 1000, size=seq_len).tolist()
    bos = model.tokenizer.bos_token_id
    toks = torch.tensor([[bos] + r + r], device=model.cfg.device)
    _, cache = model.run_with_cache(toks, return_type=None)
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    scores = np.zeros((nL, nH))
    for L in range(nL):
        patt = cache["pattern", L][0]                    # [n_heads, seq, seq]
        for H in range(nH):
            vals = [float(patt[H, i, i - seq_len + 1])
                    for i in range(seq_len + 1, 2 * seq_len + 1)]
            scores[L, H] = float(np.mean(vals))
    return scores


def top_heads(scores, k):
    """The k highest-scoring (layer, head) pairs, descending."""
    flat = [(scores[L, H], L, H) for L in range(scores.shape[0]) for H in range(scores.shape[1])]
    flat.sort(reverse=True)
    return [(L, H) for _, L, H in flat[:k]]


def make_ablation_hooks(model, heads):
    """Forward hooks that zero the z-output of the given (layer, head) pairs. Returns a list
    of (hook_name, fn) for model.run_with_hooks."""
    by_layer = {}
    for (L, H) in heads:
        by_layer.setdefault(L, []).append(H)

    def hook_factory(hs):
        def fn(z, hook):                                 # z: [batch, seq, n_heads, d_head]
            z[:, :, hs, :] = 0.0
            return z
        return fn

    return [(f"blocks.{L}.attn.hook_z", hook_factory(hs)) for L, hs in by_layer.items()]


def target_prob(model, context_ids, target_id, hooks=None):
    """Probability the model assigns to `target_id` as the next token after `context_ids`,
    optionally under ablation hooks. The logit-space analogue of `P(reproduce)`."""
    import torch
    toks = torch.tensor([list(context_ids)], device=model.cfg.device)
    with torch.no_grad():
        if hooks:
            logits = model.run_with_hooks(toks, fwd_hooks=hooks, return_type="logits")
        else:
            logits = model(toks, return_type="logits")
        p = torch.softmax(logits[0, -1], dim=-1)[int(target_id)]
    return float(p)
