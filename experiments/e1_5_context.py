"""
E1.5 -- is the knee an N effect or a context-length effect?

In E1.1 the context is S*N, so as N grows the number of repetitions AND the total context
length grow together -- a confound. E1.5 disentangles them with two tests.

Test 1 (N vs total length). Compare two conditions at each N:
  growing : context = S*N            (length = N*p, grows with N -- this is E1.1)
  fixed   : context = filler + S*N   (filler pads to a constant total length)
If the two curves coincide, reproduction depends on the number of copies N, not on total
context length -- the knee is an N-phenomenon.

Test 2 (distance / recency). Hold N above the knee and insert a gap of G neutral tokens
between the payload and a re-presented matching cue:  context = S*N + filler(G) + [S[-1]].
Sweep G. If reproduction survives large G, induction matches across distance (a range
property); if it decays, the mechanism is recency-limited.

Usage:
  python -m experiments.e1_5_context --model gpt2 --lengths 1 3 5 \
      --reps 1 2 4 8 16 32 --gaps 0 8 16 32 64 128 --payloads 4 --trials 10
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
    ap.add_argument('--lengths', type=int, nargs='+', default=[1, 3, 5])
    ap.add_argument('--reps', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    ap.add_argument('--gaps', type=int, nargs='+', default=[0, 8, 16, 32, 64, 128])
    ap.add_argument('--gap-N', type=int, default=16, help='fixed (above-knee) N for the gap test')
    ap.add_argument('--payloads', type=int, default=4)
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--max-new', type=int, default=48)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--greedy', action='store_true')
    ap.add_argument('--fid-threshold', type=float, default=0.5)
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--prefix', default='results/e1_5')
    args = ap.parse_args()

    try:
        from tcc.models import LM
    except ModuleNotFoundError as e:
        raise SystemExit(f"E1.5 needs torch/transformers: {e}\n  pip install -r requirements.txt")

    rng = np.random.default_rng(args.seed)
    lm = LM(args.model, device=args.device)
    pool = payloads.rare_token_pool(lm.tokenizer)
    t0 = time.time()
    print(f"model={args.model}  device={lm.device}  payloads/length={args.payloads}\n")

    grid = {}
    for p in args.lengths:
        pls = payloads.sample_payloads(pool, p, args.payloads, rng, tokenizer=lm.tokenizer)
        Lmax = max(args.reps) * p
        grow, fixed, gap = [], [], []
        # Test 1: growing vs fixed total length
        for N in args.reps:
            gk = gn = fk = fn = 0
            for S in pls:
                ctx_g = payloads.build_context(S, N)
                k, n = pr(lm, ctx_g, S, ctx_g[-1:], args); gk += k; gn += n
                fil = payloads.filler_tokens(pool, Lmax - N * p, S, rng)
                ctx_f = fil + list(S) * N
                k2, n2 = pr(lm, ctx_f, S, ctx_f[-1:], args); fk += k2; fn += n2
            grow.append(scoring.wilson_interval(gk, gn))
            fixed.append(scoring.wilson_interval(fk, fn))
        # Test 2: distance / recency
        for G in args.gaps:
            zk = zn = 0
            for S in pls:
                fil = payloads.filler_tokens(pool, G, S, rng)
                ctx = list(S) * args.gap_N + fil + [S[-1]]
                k, n = pr(lm, ctx, S, [S[-1]], args); zk += k; zn += n
            gap.append(scoring.wilson_interval(zk, zn))
        grid[str(p)] = dict(grow=grow, fixed=fixed, gap=gap, Lmax=Lmax)
        print(f"p={p:2d}  growing@maxN={grow[-1][0]:.2f}  fixed@maxN={fixed[-1][0]:.2f}  "
              f"gap0={gap[0][0]:.2f} -> gap{args.gaps[-1]}={gap[-1][0]:.2f}  [{time.time()-t0:.0f}s]")

    os.makedirs(os.path.dirname(args.prefix) or '.', exist_ok=True)
    with open(f"{args.prefix}_{args.model.replace('/', '_')}.json", 'w') as f:
        json.dump(dict(model=args.model, reps=args.reps, gaps=args.gaps, gap_N=args.gap_N,
                       grid=grid, seconds=round(time.time() - t0, 1)), f, indent=2)
    plot(args, grid)


def plot(args, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lens = sorted(grid, key=int)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(lens)))
    reps = np.array(args.reps, float); gaps = np.array(args.gaps, float)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.7))
    for pk, col in zip(lens, colors):
        g = grid[pk]
        ax.plot(reps, [m[0] for m in g['grow']], 'o-', color=col, lw=1.9, label=f'p={pk} growing')
        ax.plot(reps, [m[0] for m in g['fixed']], 's--', color=col, lw=1.5, alpha=0.8)
    ax.plot([], [], 'ks--', alpha=0.8, label='fixed total length')
    ax.axhline(0.5, color='k', lw=0.6, ls=':')
    ax.set_xscale('log', base=2); ax.set_xlabel('repetitions $N$')
    ax.set_ylabel('P(reproduce)'); ax.set_ylim(-0.02, 1.04)
    ax.set_title('Test 1: knee is an $N$ effect, not context length\n'
                 '(growing solid vs fixed-length dashed should coincide)')
    ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3, which='both')

    for pk, col in zip(lens, colors):
        g = grid[pk]
        ax2.plot(gaps, [m[0] for m in g['gap']], 'o-', color=col, lw=1.9, label=f'p={pk}')
    ax2.axhline(0.5, color='k', lw=0.6, ls=':')
    ax2.set_xlabel(f'gap $G$ between payload and cue  (N={args.gap_N})')
    ax2.set_ylabel('P(reproduce)'); ax2.set_ylim(-0.02, 1.04)
    ax2.set_title('Test 2: distance / recency\n(flat = induction matches across distance)')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{args.prefix}_{args.model.replace('/', '_')}.png", dpi=130)
    print(f"\nwrote {args.prefix}_{args.model.replace('/', '_')}.png")


if __name__ == "__main__":
    main()
