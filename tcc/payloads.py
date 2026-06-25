"""
Out-of-distribution payload construction, at the TOKEN level (the plan's first threat to
validity: "string length p" is in tokens, not characters, and OOD spans tokenize
unstably). A payload is a short cyclic string of token ids that the model would essentially
never emit on its own, so any reproduction is the copy mechanism, not the model's taste --
the real-transformer analogue of the toy's rainbow / de Bruijn quines.

Pure numpy: the token-space constructors need no tokenizer. `rare_token_pool` takes an
already-loaded HF tokenizer (passed in) but imports nothing torch at module load.
"""
from __future__ import annotations
import numpy as np


def rainbow_payload(token_pool, length, rng):
    """An order-1 quine analogue: `length` DISTINCT tokens forming a cycle
    x0 -> x1 -> ... -> x0. Distinctness makes the order-1 successor map single-valued."""
    pool = list(token_pool)
    if length > len(pool):
        raise ValueError("token_pool too small for a distinct-token payload")
    return [int(t) for t in rng.choice(pool, size=length, replace=False)]


def random_payload(token_pool, length, rng):
    """A generic span (tokens may repeat) -- exercises higher effective order and the
    approximate-quine regime."""
    return [int(t) for t in rng.choice(list(token_pool), size=length, replace=True)]


def build_context(payload, n_reps, prefix=None):
    """The context fed to the model: optional `prefix` (filler) followed by `n_reps`
    copies of the payload. Generation continues from the END, which is mid-payload, so
    the model is primed inside the basin -- the same discipline as the toy's reproduce()."""
    ctx = list(prefix) if prefix else []
    ctx.extend(payload * int(n_reps))
    return ctx


def rare_token_pool(tokenizer, size=512, max_id=None):
    """A pool of 'safe, rare-ish' single tokens to draw OOD payloads from: tokens that
    decode to short, plain alphanumeric pieces, excluding special / added tokens. This is
    a heuristic floor -- refine with a real unigram-frequency estimate when available
    (a TODO noted in the plan). Returns a list of token ids.

    `tokenizer` is an already-constructed HF tokenizer; this function imports no torch.
    """
    special = set(tokenizer.all_special_ids or [])
    added = set(getattr(tokenizer, 'added_tokens_decoder', {}).keys())
    vocab = tokenizer.get_vocab()
    n = max_id or tokenizer.vocab_size
    pool = []
    for tok_id in range(n):
        if tok_id in special or tok_id in added:
            continue
        s = tokenizer.decode([tok_id]).strip()
        # plain, short, non-empty pieces; avoids whitespace-only and exotic glyphs
        if 1 <= len(s) <= 6 and s.isalnum() and s.isascii():
            pool.append(tok_id)
        if len(pool) >= size:
            break
    if len(pool) < 8:
        # last resort: a mid-vocab band, minus specials
        pool = [t for t in range(n // 4, n // 4 + size) if t not in special][:size]
    return pool
