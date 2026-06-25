"""
E1 -- the condensation knee (keystone of the plan, tests claim C1).

Build an out-of-distribution token string S of length p, repeat it N times in the context,
let the model continue, and measure whether the continuation REGENERATES S (transition
fidelity above threshold). Sweep N and p; P(reproduce) vs N is the real-transformer
analogue of the toy's condensation transition.

Predicted (from the toy's induction surrogate, milestone 3.1): a sharp knee at small N --
real models, dominated by induction heads, should copy a repeated novel span after only a
few presentations -- and a knee location roughly flat in p (length-robust). Falsified if
there is no threshold, or if reps-to-reproduce climbs steeply with p (count-cache-like
brittleness).

This is Phase-0 scaffolding: one payload per length, sampled continuations. TODOs for
Phase 1: average over multiple payloads/seeds, sweep temperature and model size, and add
the E2 induction-head attribution.

Usage:
  python -m experiments.e1_condensation_knee --model gpt2 \
      --lengths 1 2 3 5 8 --reps 1 2 4 8 16 32 --trials 16 --max-new 64
"""
from __future__ import annotations
import argparse
import os
import numpy as np

from tcc import payloads, scoring


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt2')
    ap.add_argument('--lengths', type=int, nargs='+', default=[1, 2, 3, 5, 8])
    ap.add_argument('--reps', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    ap.add_argument('--trials', type=int, default=16)
    ap.add_argument('--max-new', type=int, default=64)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--fid-threshold', type=float, default=0.5)
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default='results/e1_knee.png')
    args = ap.parse_args()

    try:
        from tcc.models import LM
    except ModuleNotFoundError as e:
        raise SystemExit(f"E1 needs the model stack (torch/transformers): {e}\n"
                         f"  pip install -r requirements.txt")

    rng = np.random.default_rng(args.seed)
    lm = LM(args.model, device=args.device)
    pool = payloads.rare_token_pool(lm.tokenizer)
    print(f"model={args.model}  device={lm.device}  rare-token pool={len(pool)}\n")

    grid = {}   # p -> list of P(reproduce) over reps
    for p in args.lengths:
        S = payloads.rainbow_payload(pool, p, rng)
        probs = []
        for N in args.reps:
            ctx = payloads.build_context(S, N)
            conts = lm.continue_ids(ctx, args.max_new, n=args.trials,
                                    temperature=args.temperature)
            hits = [scoring.reproduces(c, S, k=1, prefix=ctx[-1:],
                                       fid_threshold=args.fid_threshold) for c in conts]
            probs.append(float(np.mean(hits)))
        grid[p] = probs
        kn = scoring.knee(args.reps, probs, args.fid_threshold)
        print(f"p={p:2d}  S={S}  knee N*={kn}")
        for N, q in zip(args.reps, probs):
            print(f"    N={N:3d}  P(reproduce)={q:.3f}")

    plot(args, grid)


def plot(args, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(grid)))
    for (p, probs), col in zip(grid.items(), colors):
        ax.plot(args.reps, probs, 'o-', color=col, lw=1.9, label=f'string length p={p}')
        kn = scoring.knee(args.reps, probs, args.fid_threshold)
        if kn is not None:
            ax.axvline(kn, color=col, ls=':', lw=1.0, alpha=0.6)
    ax.axhline(args.fid_threshold, color='k', lw=0.6, ls=':')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('repetitions of the payload in context  $N$')
    ax.set_ylabel('P(model regenerates the string)')
    ax.set_title(f'E1: condensation knee on {args.model}\n'
                 '(does a repeated OOD string get copied back?)')
    ax.set_ylim(-0.02, 1.04)
    ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
