"""
E2 -- induction-head attribution (mechanistic test of C2).

E1.2 showed reproduction is cue-triggered copy (behavioral induction). E2 asks whether the
actual induction heads carry it. Two steps:

  1. Identify induction heads in GPT-2-small via the standard repeated-random-sequence
     induction score (tcc.interp.induction_scores).
  2. Causal test: build a payload context S*N (N above the knee), and measure the model's
     probability of the correct next payload token. Then ablate the top induction heads and
     re-measure; compare to ablating the same number of RANDOM heads. If induction heads
     carry reproduction, ablating them collapses the probability while random ablation does
     not.

Differential prediction from E1: induction ablation should hit reproduction at all lengths,
hardest at p>=3; the p=1 case may retain a residual (the frequency tail E1.2 found).

Usage:
  python -m experiments.e2_induction --model gpt2 --lengths 1 3 5 --N 16 \
      --payloads 6 --k-heads 8
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np

from tcc import payloads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt2')
    ap.add_argument('--lengths', type=int, nargs='+', default=[1, 3, 5])
    ap.add_argument('--reps', type=int, nargs='+', default=[2, 3, 4, 8, 16],
                    help='sweep N: the circuit is load-bearing near the knee, redundant at saturation')
    ap.add_argument('--payloads', type=int, default=5)
    ap.add_argument('--k-heads', type=int, default=8, help='# top induction heads to ablate')
    ap.add_argument('--ind-seqlen', type=int, default=50)
    ap.add_argument('--device', default=None,
                    help="cpu recommended: TransformerLens warns MPS may be silently wrong on torch 2.8")
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--prefix', default='results/e2')
    ap.add_argument('--fresh', action='store_true',
                    help="ignore any existing results JSON and recompute all cells (default: resume)")
    args = ap.parse_args()

    try:
        from tcc import interp
    except ModuleNotFoundError as e:
        raise SystemExit(f"E2 needs transformer_lens: {e}\n  pip install transformer_lens")

    model = interp.load_hooked(args.model, device=args.device)
    print(f"loaded {args.model}  ({model.cfg.n_layers}L x {model.cfg.n_heads}H)  "
          f"device={model.cfg.device}")

    # 1. identify induction heads
    scores = interp.induction_scores(model, seq_len=args.ind_seqlen, seed=args.seed)
    ind_heads = interp.top_heads(scores, args.k_heads)
    print(f"\ntop {args.k_heads} induction heads (layer, head | score):")
    for (L, H) in ind_heads:
        print(f"  L{L}H{H}  {scores[L,H]:.3f}")
    ind_hooks = interp.make_ablation_hooks(model, ind_heads)

    # a matched random control set (heads that are NOT in the induction set)
    rng = np.random.default_rng(args.seed + 1)
    all_heads = [(L, H) for L in range(model.cfg.n_layers) for H in range(model.cfg.n_heads)]
    pool_heads = [h for h in all_heads if h not in set(ind_heads)]
    rand_heads = [tuple(pool_heads[i]) for i in rng.choice(len(pool_heads), args.k_heads, replace=False)]
    rand_hooks = interp.make_ablation_hooks(model, rand_heads)

    # token pool for payloads (HF tokenizer lives on the hooked model)
    pool = payloads.rare_token_pool(model.tokenizer)

    # 2. causal test: per length, sweep N (the circuit is necessary near the knee).
    # Each (p,N) cell is independent forward passes, so we checkpoint after every cell:
    # the grid is flushed to the output JSON as it fills, and on restart we skip cells
    # already present (resume). A Spot eviction thus loses at most one in-flight cell.
    outpath = f"{args.prefix}_{args.model.replace('/', '_')}.json"
    os.makedirs(os.path.dirname(args.prefix) or '.', exist_ok=True)

    grid = {}
    if os.path.exists(outpath) and not args.fresh:
        try:
            prev = json.load(open(outpath))
            if prev.get('model') == args.model and prev.get('k_heads') == args.k_heads:
                grid = prev.get('grid', {})
                print(f"resuming: {sum(len(v) for v in grid.values())} (p,N) cells already done")
            else:
                print("existing results have a different config; starting fresh")
        except Exception as e:
            print(f"could not read {outpath} for resume ({e}); starting fresh")

    out = dict(model=args.model, reps=args.reps, k_heads=args.k_heads,
               induction_heads=[[int(L), int(H)] for L, H in ind_heads],
               induction_scores_top=[float(scores[L, H]) for L, H in ind_heads],
               random_heads=[[int(L), int(H)] for L, H in rand_heads], grid=grid)

    def flush():
        with open(outpath, 'w') as f:
            json.dump(out, f, indent=2)

    for p in args.lengths:
        pls = payloads.sample_payloads(pool, p, args.payloads, rng, tokenizer=model.tokenizer)
        cells = grid.setdefault(str(p), {})
        for N in args.reps:
            if str(N) in cells:                            # already computed -- resume
                print(f"p={p:2d} N={N:3d}  (cached, skip)")
                continue
            base, abl_ind, abl_rand = [], [], []
            for S in pls:
                ctx = payloads.build_context(S, N)
                target = S[0]                              # correct next token after S[-1]
                base.append(interp.target_prob(model, ctx, target))
                abl_ind.append(interp.target_prob(model, ctx, target, hooks=ind_hooks))
                abl_rand.append(interp.target_prob(model, ctx, target, hooks=rand_hooks))
            b, ai, ar = float(np.mean(base)), float(np.mean(abl_ind)), float(np.mean(abl_rand))
            cells[str(N)] = dict(base=b, abl_induction=ai, abl_random=ar)
            flush()                                        # checkpoint: Spot-eviction safe
            di = f"-{100*(1-ai/b):.0f}%" if b else "n/a"
            dr = f"-{100*(1-ar/b):.0f}%" if b else "n/a"
            print(f"p={p:2d} N={N:3d}  base={b:.3f}  abl-induction={ai:.3f} ({di})  "
                  f"abl-random={ar:.3f} ({dr})")

    flush()
    plot(args, grid)


def plot(args, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lens = sorted(grid, key=int)
    reps = np.array(args.reps, float)
    fig, axes = plt.subplots(1, len(lens), figsize=(4.4 * len(lens), 4.4), sharey=True)
    if len(lens) == 1:
        axes = [axes]
    for ax, pk in zip(axes, lens):
        g = grid[pk]
        ax.plot(reps, [g[str(N)]['base'] for N in args.reps], 'o-', color='crimson', lw=2, label='baseline')
        ax.plot(reps, [g[str(N)]['abl_induction'] for N in args.reps], 's-', color='navy', lw=2,
                label=f'ablate {args.k_heads} induction heads')
        ax.plot(reps, [g[str(N)]['abl_random'] for N in args.reps], '^--', color='slategray', lw=1.6,
                alpha=0.8, label=f'ablate {args.k_heads} random heads')
        ax.set_xscale('log', base=2); ax.set_xlabel('repetitions $N$')
        ax.set_title(f'p={pk}'); ax.set_ylim(-0.02, 1.04); ax.grid(alpha=0.3, which='both')
    axes[0].set_ylabel('P(correct next payload token)')
    axes[0].legend(fontsize=7, loc='lower right')
    fig.suptitle(f'E2: induction heads carry reproduction near the knee ({args.model})\n'
                 'ablating induction heads suppresses it at low N; random heads do not', fontsize=11)
    fig.tight_layout(); fig.savefig(f"{args.prefix}_{args.model.replace('/', '_')}.png", dpi=130)
    print(f"\nwrote {args.prefix}_{args.model.replace('/', '_')}.png")


if __name__ == "__main__":
    main()
