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


def roundtrip_stable(tokenizer, ids):
    """True iff decoding `ids` to text and re-encoding gives back exactly `ids`. A payload
    that fails this is not actually present in the context as the tokens we score against
    (the E1 tokenization-instability threat) and should be discarded."""
    text = tokenizer.decode(ids)
    re = tokenizer.encode(text, add_special_tokens=False)
    return list(re) == list(ids)


def sample_payloads(pool, length, n, rng, kind="rainbow", tokenizer=None, max_tries=4000):
    """Draw `n` DISTINCT payloads of the given length and kind, each round-trip-stable
    (if a tokenizer is given). Averaging over payloads is what turns the smoke test into a
    measurement -- it removes the which-tokens-got-picked artifact."""
    out, seen = [], set()
    builder = rainbow_payload if kind == "rainbow" else random_payload
    for _ in range(max_tries):
        if len(out) >= n:
            break
        S = builder(pool, length, rng)
        key = tuple(S)
        if key in seen:
            continue
        if tokenizer is not None and not roundtrip_stable(tokenizer, S):
            continue
        seen.add(key)
        out.append(S)
    if len(out) < n:
        raise RuntimeError(f"only found {len(out)}/{n} stable payloads (length {length}); "
                           f"enlarge the pool or lower n")
    return out


def scrambled_context(payload, n_reps, rng):
    """CONTROL. The same multiset of tokens as `payload * n_reps`, but in a shuffled order
    so the payload n-gram does NOT recur -- same token frequencies, no repeated pattern.
    Forced to end in `payload[-1]` so the generation-start token matches the main
    condition. If reproduction needs the PATTERN (induction) rather than mere token
    presence, this control should not lock."""
    toks = list(payload) * int(n_reps)
    toks = [toks[i] for i in rng.permutation(len(toks))]
    if toks and toks[-1] != payload[-1]:
        for i in range(len(toks)):
            if toks[i] == payload[-1]:
                toks[i], toks[-1] = toks[-1], toks[i]
                break
    return toks


def nopayload_context(payload, pool, rng):
    """CONTROL (N=0 / spontaneous emission). A short filler of random pool tokens ending in
    `payload[-1]`, so the model is primed at the same start token but has NEVER seen the
    payload. P(reproduce) here is the spontaneous floor -- must be ~0 for a genuine OOD
    payload."""
    filler = [int(t) for t in rng.choice(list(pool), size=max(1, len(payload)), replace=True)]
    return filler + [payload[-1]]


def neutral_token(pool, exclude, rng):
    """A pool token NOT in `exclude` -- a non-matching cue: it has no prior occurrence in
    the payload, so an induction (copy-on-match) mechanism cannot be triggered by it,
    whereas a frequency bias would be cue-independent."""
    excl = set(int(e) for e in exclude)
    cand = [int(t) for t in pool if int(t) not in excl]
    return int(rng.choice(cand))


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
