"""
E7 -- the paired detector (the defensive complement to E4/E5/E6).

The attacks all key on one quantity: induction trust earned by a repeated span (E4 shows
poisonability tracks it, not usefulness). So the defense is to *watch that same quantity*.
The detector scores a context by its mean induction engagement -- the attention the
canonical GPT-2 induction heads pay to repeated-bigram continuations
(tcc.interp.repeat_induction_profile) -- which is high on a poisoned (repeated) span and
~0 on natural non-repetitive text, regardless of whether the content is meaningful.

We test it as a binary detector: clean prose documents vs. the same documents with an OOD
payload repeated k times planted in them. Report the score separation (ROC-AUC, computed as
the Mann-Whitney probability that a poisoned context outscores a clean one) as a function of
k. The point: detection sensitivity *rises with k*, i.e. it grows exactly as the context
becomes more poisonable -- so a defender can flag (or cap the trust on) repeated spans that
approach the condensation knee, which is precisely what neutralizes the E5 hijack and the E6
worm, and it does so on the trust signal, independent of usefulness.

Run on CPU (TransformerLens + GPT-2-small):  python -m experiments.e7_detector
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np

from tcc import payloads, interp
from experiments.e5_hijack import DOCS


def auc(pos, neg):
    """ROC-AUC as P(pos > neg) over all pairs (Mann-Whitney), ties counted as 0.5."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--payload-p', type=int, default=3)
    ap.add_argument('--ks', type=int, nargs='+', default=[2, 4, 8, 16])
    ap.add_argument('--instances', type=int, default=4, help='payloads per (doc,k)')
    ap.add_argument('--k-heads', type=int, default=8)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--prefix', default='results/e7_detector')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model = interp.load_hooked("gpt2", device="cpu")
    tok = model.tokenizer
    heads = interp.top_heads(interp.induction_scores(model, seed=args.seed), args.k_heads)
    pool = payloads.rare_token_pool(tok)
    docs = [tok.encode(d) for d in DOCS]
    print(f"induction heads: {heads}\n")

    def score(ids):
        return float(np.mean(interp.repeat_induction_profile(model, ids, heads)))

    clean = [score(d) for d in docs]
    print(f"clean docs score: {[round(c, 4) for c in clean]}")

    poisoned = {}   # k -> list of scores
    for k in args.ks:
        scores = []
        for d in docs:
            for _ in range(args.instances):
                S = payloads.sample_payloads(pool, args.payload_p, 1, rng, tokenizer=tok)[0]
                scores.append(score(d + list(S) * k))     # payload planted at the end
        poisoned[str(k)] = scores
        print(f"k={k:3d}  poisoned score mean={np.mean(scores):.4f}  "
              f"AUC(vs clean)={auc(scores, clean):.3f}")

    aucs = {k: auc(poisoned[str(k)], clean) for k in [str(x) for x in args.ks]}
    os.makedirs(os.path.dirname(args.prefix) or '.', exist_ok=True)
    with open(f"{args.prefix}.json", 'w') as f:
        json.dump(dict(clean=clean, poisoned=poisoned, auc=aucs, heads=heads), f, indent=2)
    plot(args, clean, poisoned)


def plot(args, clean, poisoned):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.7))

    # Panel A: score distributions, clean vs poisoned at each k
    pos = [0]
    ax.scatter(np.zeros(len(clean)), clean, color='navy', s=40, label='clean docs', zorder=3)
    for i, k in enumerate([str(x) for x in args.ks], 1):
        ys = poisoned[k]
        ax.scatter(np.full(len(ys), i) + np.random.uniform(-0.08, 0.08, len(ys)), ys,
                   color='crimson', s=20, alpha=0.7)
        pos.append(i)
    ax.scatter([], [], color='crimson', s=20, label='poisoned (payload $\\times k$)')
    ax.set_xticks(pos); ax.set_xticklabels(['clean'] + [f'k={k}' for k in args.ks])
    ax.set_ylabel('detector score (mean induction trust)')
    ax.set_title('Detector cleanly separates poisoned from clean\n'
                 'score rises with $k$ (toward the condensation knee)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    # Panel B: AUC vs k
    ks = [int(x) for x in args.ks]
    ax2.plot(ks, [auc(poisoned[str(k)], clean) for k in ks], 'o-', color='crimson', lw=2)
    ax2.axhline(1.0, color='k', lw=0.6, ls=':'); ax2.axhline(0.5, color='gray', lw=0.6, ls=':')
    ax2.set_xscale('log', base=2); ax2.set_xlabel('payload repetitions $k$')
    ax2.set_ylabel('detection AUC (poisoned vs clean)')
    ax2.set_title('Detection sensitivity grows with poisonability\n'
                  '(defend by flagging / capping trust on repeated spans)')
    ax2.set_ylim(0.45, 1.02); ax2.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(f"{args.prefix}.png", dpi=130)
    print(f"\nwrote {args.prefix}.png and {args.prefix}.json")


if __name__ == "__main__":
    main()
