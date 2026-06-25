"""
E1 -- the condensation knee (keystone of the plan, tests claims C1 and C3).

Build an out-of-distribution token string S of length p, repeat it N times in the context,
let the model continue, and measure whether the continuation REGENERATES S (transition
fidelity above threshold). Sweep N and p; P(reproduce) vs N is the real-transformer
analogue of the toy's condensation transition.

This is the E1.1 *measured* version (see e1_plan.md): it averages over multiple random
payloads per length (Wilson confidence intervals), and runs two controls --

  scrambled : the same tokens, shuffled so the payload n-gram does NOT recur (same token
              frequencies, no repeated pattern). If reproduction needs the PATTERN
              (induction) rather than mere token presence, this must not lock.
  no-payload: N=0 spontaneous emission -- primed at the same start token but never shown
              the payload. The floor; must be ~0 for a genuine OOD payload.

Predicted (from the toy's induction surrogate, milestone 3.1): a sharp knee at small N,
roughly flat or decreasing in p (length-robust), well above both controls. Falsified if
there is no threshold, if N* climbs steeply with p, or if the scrambled control
reproduces as well (then it is token-presence, not pattern-copying).

Usage:
  python -m experiments.e1_condensation_knee --model gpt2 \
      --lengths 1 2 3 5 --reps 1 2 4 8 16 32 --payloads 4 --trials 10
"""
from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np

from tcc import payloads, scoring


def run_cell(lm, ctx, S, args):
    """Generate `trials` continuations of one context and return (#reproduced, #trials)."""
    conts = lm.continue_ids(ctx, args.max_new, n=args.trials,
                            temperature=args.temperature, greedy=args.greedy)
    k = sum(scoring.reproduces(c, S, k=1, prefix=ctx[-1:], fid_threshold=args.fid_threshold)
            for c in conts)
    return k, len(conts)


def bootstrap_knee(reps, per_payload_P, threshold, n_boot=500, seed=0):
    """Knee CI by resampling payloads. per_payload_P: array [n_payloads, n_reps]."""
    rng = np.random.default_rng(seed)
    P = np.asarray(per_payload_P)
    npl = P.shape[0]
    ks = []
    for _ in range(n_boot):
        curve = P[rng.integers(0, npl, npl)].mean(axis=0)
        kv = scoring.knee(reps, curve, threshold)
        if kv is not None:
            ks.append(kv)
    if not ks:
        return (None, None, None)
    return (float(np.median(ks)), float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt2')
    ap.add_argument('--lengths', type=int, nargs='+', default=[1, 2, 3, 5])
    ap.add_argument('--reps', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    ap.add_argument('--payloads', type=int, default=4, help='distinct payloads per length')
    ap.add_argument('--trials', type=int, default=10, help='sampled continuations per cell')
    ap.add_argument('--max-new', type=int, default=48)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--greedy', action='store_true')
    ap.add_argument('--no-controls', action='store_true')
    ap.add_argument('--fid-threshold', type=float, default=0.5)
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--prefix', default='results/e1_1')
    args = ap.parse_args()
    controls = not args.no_controls

    try:
        from tcc.models import LM
    except ModuleNotFoundError as e:
        raise SystemExit(f"E1 needs the model stack (torch/transformers): {e}\n"
                         f"  pip install -r requirements.txt")

    rng = np.random.default_rng(args.seed)
    lm = LM(args.model, device=args.device)
    pool = payloads.rare_token_pool(lm.tokenizer)
    t0 = time.time()
    print(f"model={args.model}  device={lm.device}  pool={len(pool)}  "
          f"payloads/length={args.payloads}  trials={args.trials}  controls={controls}\n")

    grid = {}
    for p in args.lengths:
        pls = payloads.sample_payloads(pool, p, args.payloads, rng, tokenizer=lm.tokenizer)
        per_pl = np.zeros((len(pls), len(args.reps)))     # per-payload P(reproduce), main
        main, scr = [], []
        for j, N in enumerate(args.reps):
            mk = mn = sk = sn = 0
            for pi, S in enumerate(pls):
                k, n = run_cell(lm, payloads.build_context(S, N), S, args)
                mk += k; mn += n; per_pl[pi, j] = k / n
                if controls:
                    k2, n2 = run_cell(lm, payloads.scrambled_context(S, N, rng), S, args)
                    sk += k2; sn += n2
            main.append(scoring.wilson_interval(mk, mn))
            if controls:
                scr.append(scoring.wilson_interval(sk, sn))
        nopay = None
        if controls:
            zk = zn = 0
            for S in pls:
                k, n = run_cell(lm, payloads.nopayload_context(S, pool, rng), S, args)
                zk += k; zn += n
            nopay = scoring.wilson_interval(zk, zn)
        n_star, width = scoring.knee_width(args.reps, [m[0] for m in main])
        boot = bootstrap_knee(args.reps, per_pl, args.fid_threshold, seed=args.seed)
        grid[str(p)] = dict(reps=args.reps, main=main, scrambled=scr, nopayload=nopay,
                            knee=n_star, width=width, knee_ci=boot)
        sc0 = scr[0][0] if scr else None
        print(f"p={p:2d}  knee N*={None if n_star is None else round(n_star,2)} "
              f"(95% CI {None if boot[1] is None else round(boot[1],1)}"
              f"-{None if boot[2] is None else round(boot[2],1)})  width={None if width is None else round(width,2)}"
              f"  scrambled P@maxN={round(scr[-1][0],2) if scr else '-'}  "
              f"nopay P={round(nopay[0],3) if nopay else '-'}  [{time.time()-t0:.0f}s]")

    os.makedirs(os.path.dirname(args.prefix) or '.', exist_ok=True)
    meta = dict(model=args.model, lengths=args.lengths, reps=args.reps,
                payloads=args.payloads, trials=args.trials, temperature=args.temperature,
                greedy=args.greedy, fid_threshold=args.fid_threshold, controls=controls,
                seconds=round(time.time() - t0, 1), grid=grid)
    with open(f"{args.prefix}_{args.model.replace('/', '_')}.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {args.prefix}_{args.model.replace('/', '_')}.json")
    plot(args, grid)


def plot(args, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    reps = np.array(args.reps, float)
    lens = sorted(grid, key=int)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(lens)))

    # Figure 1: P(reproduce) vs N, per length, with Wilson CI bands + controls
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for pk, col in zip(lens, colors):
        g = grid[pk]
        ph = np.array([m[0] for m in g['main']]); lo = np.array([m[1] for m in g['main']])
        hi = np.array([m[2] for m in g['main']])
        ax.plot(reps, ph, 'o-', color=col, lw=1.9, label=f'p={pk}')
        ax.fill_between(reps, lo, hi, color=col, alpha=0.15)
        if g['scrambled']:
            ax.plot(reps, [s[0] for s in g['scrambled']], 'x--', color=col, lw=1.0, alpha=0.6)
    ax.axhline(args.fid_threshold, color='k', lw=0.6, ls=':')
    ax.plot([], [], 'kx--', alpha=0.6, label='scrambled control')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('repetitions of the payload in context  $N$')
    ax.set_ylabel('P(model regenerates the string)')
    ax.set_title(f'E1.1: condensation knee on {args.model}  '
                 f'({args.payloads} payloads x {args.trials} trials)\n'
                 'solid = repeated payload (Wilson CI);  dashed = scrambled control')
    ax.set_ylim(-0.02, 1.04)
    ax.legend(fontsize=8, loc='center right'); ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(f"{args.prefix}_knee_{args.model.replace('/', '_')}.png", dpi=130)

    # Figure 2: knee N* vs length, with bootstrap CI
    fig2, ax2 = plt.subplots(figsize=(6.2, 4.6))
    ps, ns, los, his = [], [], [], []
    for pk in lens:
        b = grid[pk]['knee_ci']
        if b[0] is not None:
            ps.append(int(pk)); ns.append(b[0]); los.append(b[0] - b[1]); his.append(b[2] - b[0])
    if ps:
        ax2.errorbar(ps, ns, yerr=[los, his], fmt='o-', color='crimson', lw=1.9, capsize=4)
    ax2.set_xlabel('string length $p$ (tokens)')
    ax2.set_ylabel('knee $N^*$ (repetitions to reproduce)')
    ax2.set_title('E1.1: knee vs string length\n(flat/decreasing = induction-like; rising = brittle)')
    ax2.grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig(f"{args.prefix}_Nstar_{args.model.replace('/', '_')}.png", dpi=130)
    print(f"wrote {args.prefix}_knee_*.png and {args.prefix}_Nstar_*.png")


if __name__ == "__main__":
    main()
