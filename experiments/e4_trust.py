"""
E4-lite -- poisonability tracks TRUST (induction), not USEFULNESS (tests C4).

The toy's headline: in-context exploitability follows how much the model *trusts* a span,
not how *useful* it is. On GPT-2-small we test the dissociation with a 2x2 of contexts:

                    useful (real text)        useless (OOD gibberish)
  repetitive        a real sentence x k       a random payload x k
  non-repetitive    a real passage            random tokens

and measure three things per context:
  U (usefulness)   : naturalness of the content unit = mean per-token log-prob of one copy
                     (real text high, gibberish low). A base-model proxy for "meaningful".
  T (trust)        : induction-head engagement -- mean attention the induction heads pay to
                     the token after a previous occurrence (high for repetition, ~0 without).
  P (poisonability): copy-rate of a greedy continuation = fraction of generated tokens that
                     fall inside a 4-gram copied verbatim from the context.

Prediction (C4): P tracks T (the repetition axis), not U. The repetitive-USELESS context
(gibberish) is just as poisonable as repetitive-useful, and more than non-repetitive-useful
prose -- so poisonability is orthogonal to usefulness and aligned with induction trust.

Run on CPU (TransformerLens + GPT-2-small):  python -m experiments.e4_trust
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import torch

from tcc import payloads, interp

REAL_SENTENCES = [
    "The capital of France is Paris, a city famous for the Eiffel Tower.",
    "Water boils at one hundred degrees Celsius at sea level under normal pressure.",
    "Photosynthesis lets plants convert sunlight, water, and carbon dioxide into sugar.",
    "The square root of one hundred and forty-four is exactly twelve.",
]
REAL_PASSAGES = [
    "The river wound slowly through the valley, past old stone bridges and quiet farms, "
    "while the afternoon light faded behind the western hills and the birds went still.",
    "She opened the heavy wooden door and stepped into a long hall lined with portraits, "
    "each face watching the visitors who rarely came this far into the old museum.",
    "Engineers spent months tracing the fault, replacing cables and testing every joint, "
    "until at last the signal held steady and the bridge reopened to morning traffic.",
    "By the time the storm passed, the streets were littered with branches and the power "
    "was out across the whole district, so neighbors gathered candles and shared the news.",
]


def content_logprob(model, ids):
    """Mean per-token log-prob the model assigns to the content presented once = U."""
    toks = torch.tensor([ids])
    with torch.no_grad():
        logp = model(toks, return_type="logits")[0].log_softmax(-1)
    return float(np.mean([logp[i, ids[i + 1]].item() for i in range(len(ids) - 1)]))


def induction_engagement(model, ids, ind_heads):
    """Mean attention the induction heads pay to the continuation of a REPEATED BIGRAM = T.
    Requiring a 2-gram match (current and previous token both recur together) filters the
    coincidental single-token repeats that pollute random contexts, so this fires on genuine
    induction triggers, not chance collisions."""
    toks = torch.tensor([ids])
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, return_type=None)
    n, vals = len(ids), []
    for i in range(2, n):
        js = [j for j in range(1, i) if ids[j] == ids[i] and ids[j - 1] == ids[i - 1]]
        if not js or js[-1] + 1 > i:
            continue
        k = js[-1] + 1
        vals.append(np.mean([cache["pattern", L][0, H, i, k].item() for L, H in ind_heads]))
    return float(np.mean(vals)) if vals else 0.0


def copy_rate(model, ids, gen_len=40, ngram=4):
    """Greedy-continue and measure the fraction of generated tokens inside a 4-gram copied
    verbatim from the context = P (poisonability / regurgitation)."""
    toks = torch.tensor([ids])
    with torch.no_grad():
        out = model.generate(toks, max_new_tokens=gen_len, do_sample=False,
                             verbose=False, return_type="tokens")
    gen = out[0, len(ids):].tolist()
    ctx_ngrams = {tuple(ids[i:i + ngram]) for i in range(len(ids) - ngram + 1)}
    full = list(ids) + gen
    copied = sum(1 for i in range(len(ids), len(full))
                 if i - ngram + 1 >= 0 and tuple(full[i - ngram + 1:i + 1]) in ctx_ngrams)
    return copied / max(1, len(gen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ctx-len', type=int, default=72, help='target context length (tokens)')
    ap.add_argument('--payload-p', type=int, default=3)
    ap.add_argument('--k-heads', type=int, default=8)
    ap.add_argument('--gen-len', type=int, default=40)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--prefix', default='results/e4_trust')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model = interp.load_hooked("gpt2", device="cpu")
    tok = model.tokenizer
    ind_heads = interp.top_heads(interp.induction_scores(model, seed=args.seed), args.k_heads)
    pool = payloads.rare_token_pool(tok)
    print(f"induction heads: {ind_heads}\n")

    def enc(s):
        return tok.encode(s)

    def rep_to(ids, L):
        out = []
        while len(out) < L:
            out += ids
        return out[:L]

    rows = []   # (type, useful, repetitive, U, T, P)
    # repetitive + useful : a real sentence repeated
    for s in REAL_SENTENCES:
        base = enc(s)
        ctx = rep_to(base, args.ctx_len)
        rows.append(("rep+useful", 1, 1, content_logprob(model, base),
                     induction_engagement(model, ctx, ind_heads),
                     copy_rate(model, ctx, args.gen_len)))
    # repetitive + useless : an OOD payload repeated
    for _ in range(4):
        S = payloads.sample_payloads(pool, args.payload_p, 1, rng, tokenizer=tok)[0]
        ctx = rep_to(S, args.ctx_len)
        rows.append(("rep+useless", 0, 1, content_logprob(model, S),
                     induction_engagement(model, ctx, ind_heads),
                     copy_rate(model, ctx, args.gen_len)))
    # non-repetitive + useful : a real passage
    for s in REAL_PASSAGES:
        ctx = enc(s)[:args.ctx_len]
        rows.append(("nonrep+useful", 1, 0, content_logprob(model, ctx),
                     induction_engagement(model, ctx, ind_heads),
                     copy_rate(model, ctx, args.gen_len)))
    # non-repetitive + useless : random tokens
    for _ in range(4):
        ctx = [int(t) for t in rng.choice(pool, size=args.ctx_len, replace=True)]
        rows.append(("nonrep+useless", 0, 0, content_logprob(model, ctx),
                     induction_engagement(model, ctx, ind_heads),
                     copy_rate(model, ctx, args.gen_len)))

    print(f"{'type':16s} {'U(logprob)':>11s} {'T(induct)':>10s} {'P(copy)':>8s}")
    agg = {}
    for r in rows:
        print(f"{r[0]:16s} {r[3]:11.2f} {r[4]:10.3f} {r[5]:8.3f}")
    for t in ["rep+useful", "rep+useless", "nonrep+useful", "nonrep+useless"]:
        sub = [r for r in rows if r[0] == t]
        agg[t] = dict(U=float(np.mean([r[3] for r in sub])),
                      T=float(np.mean([r[4] for r in sub])),
                      P=float(np.mean([r[5] for r in sub])))
    U = np.array([r[3] for r in rows]); T = np.array([r[4] for r in rows]); P = np.array([r[5] for r in rows])
    rPU = float(np.corrcoef(P, U)[0, 1]); rPT = float(np.corrcoef(P, T)[0, 1])
    print(f"\ncorr(P, U) = {rPU:+.2f}   corr(P, T) = {rPT:+.2f}")
    print("=> poisonability tracks TRUST not USEFULNESS" if rPT > 0.5 and abs(rPU) < 0.4 else "=> see scatter")

    os.makedirs(os.path.dirname(args.prefix) or '.', exist_ok=True)
    with open(f"{args.prefix}.json", 'w') as f:
        json.dump(dict(agg=agg, corr_PU=rPU, corr_PT=rPT, ind_heads=ind_heads,
                       rows=[list(r) for r in rows]), f, indent=2)
    plot(agg, rows, rPU, rPT, args)


def plot(agg, rows, rPU, rPT, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    types = ["rep+useful", "rep+useless", "nonrep+useful", "nonrep+useless"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.7))

    # Panel A: per-type trust (T) and poisonability (P), normalized to [0,1]
    T = np.array([agg[t]['T'] for t in types]); P = np.array([agg[t]['P'] for t in types])
    Tn = T / (T.max() or 1); Pn = P / (P.max() or 1)
    x = np.arange(len(types)); w = 0.38
    ax.bar(x - w / 2, Tn, w, color='navy', label='trust T (induction engagement)')
    ax.bar(x + w / 2, Pn, w, color='crimson', label='poisonability P (copy-rate)')
    ax.set_xticks(x); ax.set_xticklabels(types, rotation=15, fontsize=8)
    ax.set_ylabel('normalized'); ax.set_ylim(0, 1.1)
    ax.set_title('Poisonability follows the repetition/trust axis,\nnot the usefulness axis')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # Panel B: P vs U (orthogonal) and P vs T (aligned)
    colors = {'rep+useful': 'crimson', 'rep+useless': 'darkorange',
              'nonrep+useful': 'navy', 'nonrep+useless': 'gray'}
    for r in rows:
        ax2.scatter(r[3], r[5], color=colors[r[0]], s=55, marker='o', zorder=3)
    ax2.set_xlabel('usefulness U (content log-prob $\\rightarrow$ natural)')
    ax2.set_ylabel('poisonability P (copy-rate)')
    ax2.set_title(f'P vs usefulness: corr={rPU:+.2f}  (vs trust: corr={rPT:+.2f})\n'
                  'useless gibberish is highly poisonable; useful prose is not')
    for t in types:
        ax2.scatter([], [], color=colors[t], label=t)
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{args.prefix}.png", dpi=130)
    print(f"\nwrote {args.prefix}.png and {args.prefix}.json")


if __name__ == "__main__":
    main()
