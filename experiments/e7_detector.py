"""
E7 -- the paired detector (the defensive complement to E4/E5/E6), with the baseline it
must beat and enough negatives for its AUC to mean something.

The attacks all key on one quantity: induction trust earned by a repeated span (E4 shows
poisonability tracks it, not usefulness). So the defense is to *watch that same quantity*.
The detector scores a context by its mean induction engagement -- the attention the
canonical GPT-2 induction heads pay to repeated-bigram continuations
(tcc.interp.repeat_induction_profile) -- which is high on a poisoned (repeated) span and
~0 on natural non-repetitive text, regardless of whether the content is meaningful.

What the first version of this experiment did NOT do, and this one does:
  * It had THREE clean documents. An AUC over 3 negatives is not a measurement. The negative
    class is now 32 distinct clean paragraphs (tcc.corpus), and AUCs carry a bootstrap CI.
  * It had no model-free baseline. An induction-attention score only earns its place if it
    beats "does any n-gram repeat?" -- a count that needs no model. Both are reported at
    every k. If the n-gram count matches, the detector is an n-gram repeat detector and the
    induction machinery is an explanation of WHY repetition is dangerous, not a better alarm.
  * It reported no false-positive rate. Legitimate content repeats (license headers, code,
    numbered lists, a chorus, a CSV dump). Those are scored as a separate BENIGN-REPETITIVE
    set and the fraction flagged at the operating threshold is reported.

Binary test: clean paragraphs vs. the same paragraphs with an OOD payload repeated k times
planted at the end. AUC is the Mann-Whitney probability that a poisoned context outscores a
clean one. The operating threshold is the max clean score (zero false positives on clean);
the benign-repetitive FPR is measured at that threshold.

Run on CPU (TransformerLens + GPT-2-small):  python -m experiments.e7_detector
Model-free check of the metrics:              python -m experiments.e7_detector --selftest
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np

from tcc import payloads, corpus
from tcc import snapshot


def auc(pos, neg):
    """ROC-AUC as P(pos > neg) over all pairs (Mann-Whitney), ties counted as 0.5."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))


def auc_ci(pos, neg, rng, n_boot=1000):
    """Bootstrap (resample both classes) 95% interval on the AUC."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    vals = []
    for _ in range(n_boot):
        p = pos[rng.integers(0, len(pos), len(pos))]
        n = neg[rng.integers(0, len(neg), len(neg))]
        vals.append(auc(p, n))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ngram_repeat_score(ids, n=3):
    """MODEL-FREE baseline: fraction of positions whose n-gram ending there already occurred
    earlier in the context. Zero on text with no repeated n-gram; rises with planted
    repetition exactly as the induction score is expected to."""
    ids = list(ids)
    if len(ids) < n + 1:
        return 0.0
    seen, hits = set(), 0
    for i in range(n - 1, len(ids)):
        g = tuple(ids[i - n + 1:i + 1])
        if g in seen:
            hits += 1
        seen.add(g)
    return hits / (len(ids) - n + 1)


def fpr_at_threshold(scores, thr):
    """Fraction of `scores` strictly above the operating threshold."""
    scores = np.asarray(scores, float)
    return float(np.mean(scores > thr)) if len(scores) else float('nan')


def run(score_fns, tok, pool, args, rng):
    """score_fns: {name: fn(ids)->float}. Returns the full results dict."""
    clean_docs = [tok.encode(d) for d in corpus.clean_paragraphs()]
    benign_rep = {name: tok.encode(t) for name, t in corpus.BENIGN_REPETITIVE.items()}
    out = {"n_clean": len(clean_docs), "n_benign_repetitive": len(benign_rep), "ks": args.ks,
           "detectors": {}}
    for name, fn in score_fns.items():
        clean = [fn(d) for d in clean_docs]
        thr = float(max(clean))                              # zero-FP-on-clean operating point
        brep = {k: fn(ids) for k, ids in benign_rep.items()}
        res = {"clean": clean, "threshold": thr, "benign_repetitive": brep,
               "benign_repetitive_fpr": fpr_at_threshold(list(brep.values()), thr),
               "poisoned": {}, "auc": {}, "auc_ci": {}, "recall_at_threshold": {}}
        print(f"\n[{name}] clean mean={np.mean(clean):.4f} max={thr:.4f}  "
              f"benign-repetitive FPR @thr={res['benign_repetitive_fpr']:.2f} "
              f"({sum(v > thr for v in brep.values())}/{len(brep)} flagged: "
              f"{[k for k, v in brep.items() if v > thr]})")
        for k in args.ks:
            scores = []
            for d in clean_docs:
                for _ in range(args.instances):
                    S = payloads.sample_payloads(pool, args.payload_p, 1, rng, tokenizer=tok)[0]
                    scores.append(fn(d + list(S) * k))        # payload planted at the end
            a = auc(scores, clean)
            lo, hi = auc_ci(scores, clean, rng)
            res["poisoned"][str(k)] = scores
            res["auc"][str(k)] = a
            res["auc_ci"][str(k)] = [lo, hi]
            res["recall_at_threshold"][str(k)] = fpr_at_threshold(scores, thr)
            print(f"  k={k:3d}  poisoned mean={np.mean(scores):.4f}  AUC={a:.3f} "
                  f"[{lo:.3f},{hi:.3f}]  recall@thr={res['recall_at_threshold'][str(k)]:.2f}")
        out["detectors"][name] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--payload-p', type=int, default=3)
    ap.add_argument('--ks', type=int, nargs='+', default=[2, 4, 8, 16])
    ap.add_argument('--instances', type=int, default=2, help='payloads per (doc,k)')
    ap.add_argument('--k-heads', type=int, default=8)
    ap.add_argument('--ngram', type=int, default=3, help='n for the model-free baseline')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--prefix', default='results/e7_detector')
    snapshot.add_args(ap)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    snapshot.arm(args)

    if args.selftest:
        selftest()
        return

    from tcc import interp
    rng = np.random.default_rng(args.seed)
    model = interp.load_hooked("gpt2", device="cpu")
    tok = model.tokenizer
    heads = interp.top_heads(interp.induction_scores(model, seed=args.seed), args.k_heads)
    pool = payloads.short_token_pool(tok)
    print(f"induction heads: {heads}")

    score_fns = {
        "induction": lambda ids: float(np.mean(interp.repeat_induction_profile(model, ids, heads))),
        f"ngram{args.ngram}_repeat": lambda ids: ngram_repeat_score(ids, args.ngram),
    }
    out = run(score_fns, tok, pool, args, rng)
    out["heads"] = heads
    os.makedirs(os.path.dirname(args.prefix) or '.', exist_ok=True)
    with open(f"{args.prefix}.json", 'w') as f:
        json.dump(out, f, indent=2)
    plot(args, out)


def selftest():
    # n-gram baseline: zero on distinct tokens, positive and rising on planted repetition
    assert ngram_repeat_score(list(range(50)), 3) == 0.0
    s2 = ngram_repeat_score(list(range(50)) + [900, 901, 902] * 2, 3)
    s8 = ngram_repeat_score(list(range(50)) + [900, 901, 902] * 8, 3)
    assert 0 < s2 < s8, (s2, s8)
    # AUC: perfect separation, chance, ties
    assert auc([1, 2, 3], [0, 0, 0]) == 1.0
    assert auc([0, 0], [0, 0]) == 0.5
    assert abs(auc([1, 3], [2, 2]) - 0.5) < 1e-9
    # operating threshold / FPR
    assert fpr_at_threshold([0.1, 0.5, 0.9], 0.5) == 1 / 3
    # corpus: enough distinct negatives, no duplicate sentences across paragraphs
    paras = corpus.clean_paragraphs()
    assert len(paras) >= 16, len(paras)
    assert len(set(paras)) == len(paras)
    assert len(corpus.BENIGN_REPETITIVE) >= 6
    # the benign-repetitive set really does repeat n-grams, and the clean paragraphs do not
    # (word-level proxy for BPE: lower-cased, punctuation split off as its own token)
    import re
    words = lambda s: re.findall(r"\w+|[^\w\s]", s.lower())
    for name, t in corpus.BENIGN_REPETITIVE.items():
        assert ngram_repeat_score(words(t), 2) > 0, name
    for p in paras:
        assert ngram_repeat_score(words(p), 3) == 0.0, p
    print(f"selftest: ngram baseline + AUC/CI + corpus OK ({len(paras)} clean paragraphs, "
          f"{len(corpus.BENIGN_REPETITIVE)} benign-repetitive)")


def plot(args, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dets = out["detectors"]
    fig, axes = plt.subplots(1, len(dets) + 1, figsize=(5.2 * (len(dets) + 1), 4.7))
    for ax, (name, res) in zip(axes, dets.items()):
        clean = res["clean"]
        ax.scatter(np.zeros(len(clean)), clean, color='navy', s=18, label=f'clean (n={len(clean)})', zorder=3)
        pos = [0]
        for i, k in enumerate([str(x) for x in args.ks], 1):
            ys = res["poisoned"][k]
            ax.scatter(np.full(len(ys), i) + np.random.uniform(-0.08, 0.08, len(ys)), ys,
                       color='crimson', s=14, alpha=0.6)
            pos.append(i)
        bx = len(pos)
        bvals = list(res["benign_repetitive"].values())
        ax.scatter(np.full(len(bvals), bx), bvals, color='darkgoldenrod', s=30, marker='D',
                   label=f'benign repetitive (FPR {res["benign_repetitive_fpr"]:.2f})', zorder=4)
        ax.axhline(res["threshold"], color='k', lw=0.7, ls=':', label='threshold (max clean)')
        ax.scatter([], [], color='crimson', s=14, label='poisoned (payload x k)')
        ax.set_xticks(pos + [bx]); ax.set_xticklabels(['clean'] + [f'k={k}' for k in args.ks] + ['benign\nrep.'])
        ax.set_ylabel('score'); ax.set_title(name); ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')
    ax = axes[-1]
    ks = [int(x) for x in args.ks]
    for name, res in dets.items():
        a = [res["auc"][str(k)] for k in ks]
        lo = [a[i] - res["auc_ci"][str(k)][0] for i, k in enumerate(ks)]
        hi = [res["auc_ci"][str(k)][1] - a[i] for i, k in enumerate(ks)]
        ax.errorbar(ks, a, yerr=[lo, hi], fmt='o-', lw=2, capsize=3, label=name)
    ax.axhline(1.0, color='k', lw=0.6, ls=':'); ax.axhline(0.5, color='gray', lw=0.6, ls=':')
    ax.set_xscale('log', base=2); ax.set_xlabel('payload repetitions $k$')
    ax.set_ylabel('AUC (poisoned vs clean), bootstrap 95% CI')
    ax.set_title('Induction score vs model-free n-gram baseline'); ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(f"{args.prefix}.png", dpi=130)
    print(f"\nwrote {args.prefix}.png and {args.prefix}.json")


if __name__ == "__main__":
    main()
