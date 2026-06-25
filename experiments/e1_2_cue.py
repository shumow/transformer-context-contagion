"""
E1.2 -- copy vs. frequency: is the reproduction of a repeated span CUE-DEPENDENT?

E1.1 found a clear knee, but its scrambled control was degenerate at short length, leaving
open whether short-string reproduction is genuine pattern-copying (induction) or just a
frequency bias (the repeated tokens become likely regardless of context). This experiment
separates them behaviorally -- no activation access needed, so it runs on a small model.

After streaming the payload S a fixed (above-knee) number of times N, we prime generation
two ways and ask whether the continuation regenerates S:

  self-cue    : prime at S's own tail (a MATCHING cue). Induction continues S; frequency
                also would.
  neutral-cue : append a fresh token z not in S, then generate (a NON-MATCHING cue).
                Induction is silent (z never occurred, nothing to copy); a pure frequency
                bias would still leak S.

The signature is the gap  G(p) = P(self) - P(neutral)  vs string length p:
  * induction (copy-on-match)  -> large gap (reproduction is cue-triggered);
  * frequency bias             -> small gap (reproduction is cue-independent).
Predicted from E1.1: gap large at p>=3, small at p=1 (the frequency regime).

Usage:
  python -m experiments.e1_2_cue --model gpt2 --lengths 1 2 3 5 8 --reps 16 32 \
      --payloads 6 --trials 12
"""
from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np

from tcc import payloads, scoring


def pr(lm, ctx, S, prefix, args):
    conts = lm.continue_ids(ctx, args.max_new, n=args.trials,
                            temperature=args.temperature, greedy=args.greedy)
    k = sum(scoring.reproduces(c, S, k=1, prefix=prefix, fid_threshold=args.fid_threshold)
            for c in conts)
    return k, len(conts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt2')
    ap.add_argument('--lengths', type=int, nargs='+', default=[1, 2, 3, 5, 8])
    ap.add_argument('--reps', type=int, nargs='+', default=[16, 32], help='above-knee N')
    ap.add_argument('--payloads', type=int, default=6)
    ap.add_argument('--trials', type=int, default=12)
    ap.add_argument('--max-new', type=int, default=48)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--greedy', action='store_true')
    ap.add_argument('--fid-threshold', type=float, default=0.5)
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--prefix', default='results/e1_2')
    args = ap.parse_args()

    try:
        from tcc.models import LM
    except ModuleNotFoundError as e:
        raise SystemExit(f"E1.2 needs torch/transformers: {e}\n  pip install -r requirements.txt")

    rng = np.random.default_rng(args.seed)
    lm = LM(args.model, device=args.device)
    pool = payloads.rare_token_pool(lm.tokenizer)
    t0 = time.time()
    print(f"model={args.model}  device={lm.device}  payloads/length={args.payloads}  "
          f"trials={args.trials}  N={args.reps}\n")

    grid = {}
    for p in args.lengths:
        pls = payloads.sample_payloads(pool, p, args.payloads, rng, tokenizer=lm.tokenizer)
        grid[str(p)] = {}
        for N in args.reps:
            sk = sn = nk = nn = 0
            for S in pls:
                ctx = payloads.build_context(S, N)
                k, n = pr(lm, ctx, S, ctx[-1:], args)            # self-cue (matching)
                sk += k; sn += n
                z = payloads.neutral_token(pool, S, rng)
                k2, n2 = pr(lm, ctx + [z], S, [z], args)         # neutral-cue (non-matching)
                nk += k2; nn += n2
            self_ci = scoring.wilson_interval(sk, sn)
            neu_ci = scoring.wilson_interval(nk, nn)
            grid[str(p)][str(N)] = dict(self=self_ci, neutral=neu_ci,
                                        gap=self_ci[0] - neu_ci[0])
            print(f"p={p:2d}  N={N:3d}  P(self)={self_ci[0]:.2f}  P(neutral)={neu_ci[0]:.2f}  "
                  f"gap={self_ci[0]-neu_ci[0]:+.2f}  [{time.time()-t0:.0f}s]")

    os.makedirs(os.path.dirname(args.prefix) or '.', exist_ok=True)
    with open(f"{args.prefix}_{args.model.replace('/', '_')}.json", 'w') as f:
        json.dump(dict(model=args.model, reps=args.reps, grid=grid,
                       seconds=round(time.time() - t0, 1)), f, indent=2)
    plot(args, grid)


def plot(args, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lens = sorted(grid, key=int)
    Nshow = str(args.reps[-1])     # the most-saturated N for the bar panel

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.7))
    x = np.arange(len(lens)); w = 0.38
    sv = [grid[p][Nshow]['self'] for p in lens]
    nv = [grid[p][Nshow]['neutral'] for p in lens]
    ax.bar(x - w / 2, [s[0] for s in sv], w, color='crimson', label='self-cue (matching)',
           yerr=[[s[0] - s[1] for s in sv], [s[2] - s[0] for s in sv]], capsize=3)
    ax.bar(x + w / 2, [n[0] for n in nv], w, color='slategray', label='neutral-cue (non-matching)',
           yerr=[[n[0] - n[1] for n in nv], [n[2] - n[0] for n in nv]], capsize=3)
    ax.set_xticks(x); ax.set_xticklabels([f'p={p}' for p in lens])
    ax.set_ylabel('P(regenerates the string)'); ax.set_ylim(0, 1.04)
    ax.set_title(f'E1.2 copy vs frequency on {args.model}  (N={Nshow})\n'
                 'matching cue triggers copy; non-matching does not (if induction)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    for N in args.reps:
        ax2.plot([int(p) for p in lens], [grid[p][str(N)]['gap'] for p in lens],
                 'o-', lw=1.9, label=f'N={N}')
    ax2.axhline(0, color='k', lw=0.6)
    ax2.set_xlabel('string length $p$'); ax2.set_ylabel('cue gap  $P_{self}-P_{neutral}$')
    ax2.set_title('Cue-dependence vs length\n(large = induction/copy; ~0 = frequency bias)')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{args.prefix}_{args.model.replace('/', '_')}.png", dpi=130)
    print(f"\nwrote {args.prefix}_{args.model.replace('/', '_')}.png")


if __name__ == "__main__":
    main()
