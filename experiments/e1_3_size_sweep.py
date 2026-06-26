"""
E1.3-lite -- how does the condensation knee move with model size?

Runs the E1.1 knee measurement (main condition only; controls were validated on
GPT-2-small) across the open models that fit an 8 GB machine: GPT-2 {small, medium,
large} and Pythia {160M, 410M}. Loads one model at a time and tears it down before the
next, so peak memory stays at one-model scale; robust to any single model failing.

Prediction (induction strengthens with scale): the knee should move DOWN and/or SHARPEN
with model size, and the length-robustness (N* flat-to-decreasing in p) should persist or
strengthen. Bigger models with longer-range induction may also push out the distance limit
seen in E1.5 (not tested here).

Usage:
  python -m experiments.e1_3_size_sweep \
      --models gpt2 gpt2-medium gpt2-large EleutherAI/pythia-160m EleutherAI/pythia-410m \
      --lengths 1 3 5 --reps 1 2 4 8 16 32 --payloads 3 --trials 8
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import time
import numpy as np

from tcc import payloads, scoring

# approximate parameter counts (millions) for the size axis
PARAMS_M = {
    'gpt2': 124, 'gpt2-medium': 355, 'gpt2-large': 774, 'gpt2-xl': 1558,
    'EleutherAI/pythia-160m': 162, 'EleutherAI/pythia-410m': 410,
    'EleutherAI/pythia-1b': 1011,
}


def measure_model(name, args):
    """Knee N*(p) for one model. Returns dict p -> (N*, lo, hi) or None on failure."""
    from tcc.models import LM
    import torch
    rng = np.random.default_rng(args.seed)
    lm = LM(name, device=args.device)
    pool = payloads.rare_token_pool(lm.tokenizer)
    out = {}
    for p in args.lengths:
        pls = payloads.sample_payloads(pool, p, args.payloads, rng, tokenizer=lm.tokenizer)
        per_pl = np.zeros((len(pls), len(args.reps)))
        for j, N in enumerate(args.reps):
            for pi, S in enumerate(pls):
                ctx = payloads.build_context(S, N)
                conts = lm.continue_ids(ctx, args.max_new, n=args.trials,
                                        temperature=args.temperature)
                per_pl[pi, j] = np.mean([scoring.reproduces(c, S, k=1, prefix=ctx[-1:],
                                         fid_threshold=args.fid_threshold) for c in conts])
        curve = per_pl.mean(axis=0)
        nstar = scoring.knee(args.reps, curve, args.fid_threshold)
        # bootstrap CI over payloads
        boot = []
        for _ in range(400):
            c = per_pl[rng.integers(0, len(pls), len(pls))].mean(axis=0)
            kv = scoring.knee(args.reps, c, args.fid_threshold)
            if kv is not None:
                boot.append(kv)
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (None, None)
        out[str(p)] = (nstar, ci[0], ci[1])
    # teardown -- free memory before the next model (critical on 8 GB)
    del lm
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+',
                    default=['gpt2', 'gpt2-medium', 'gpt2-large',
                             'EleutherAI/pythia-160m', 'EleutherAI/pythia-410m'])
    ap.add_argument('--lengths', type=int, nargs='+', default=[1, 3, 5])
    ap.add_argument('--reps', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    ap.add_argument('--payloads', type=int, default=3)
    ap.add_argument('--trials', type=int, default=8)
    ap.add_argument('--max-new', type=int, default=48)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--fid-threshold', type=float, default=0.5)
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--prefix', default='results/e1_3')
    args = ap.parse_args()

    try:
        import tcc.models  # noqa: F401
    except ModuleNotFoundError as e:
        raise SystemExit(f"E1.3 needs torch/transformers: {e}\n  pip install -r requirements.txt")

    t0 = time.time()
    results = {}
    for name in args.models:
        try:
            res = measure_model(name, args)
            results[name] = res
            knees = {p: (None if v[0] is None else round(v[0], 1)) for p, v in res.items()}
            print(f"[{time.time()-t0:5.0f}s] {name:28s} ({PARAMS_M.get(name,'?')}M)  knees {knees}")
        except Exception as e:                       # OOM, download failure, etc.
            results[name] = {'error': str(e)[:200]}
            print(f"[{time.time()-t0:5.0f}s] {name:28s}  FAILED: {str(e)[:120]}")

    os.makedirs(os.path.dirname(args.prefix) or '.', exist_ok=True)
    with open(f"{args.prefix}.json", 'w') as f:
        json.dump(dict(models=args.models, lengths=args.lengths, reps=args.reps,
                       params_M=PARAMS_M, results=results,
                       seconds=round(time.time() - t0, 1)), f, indent=2)
    print(f"\nwrote {args.prefix}.json")
    plot(args, results)


def plot(args, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = {m: r for m, r in results.items() if 'error' not in r}
    if not ok:
        print("no successful models to plot"); return

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.7))
    order = sorted(ok, key=lambda m: PARAMS_M.get(m, 1e9))
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(order)))
    for m, col in zip(order, colors):
        ps = [int(p) for p in sorted(ok[m], key=int) if ok[m][p][0] is not None]
        ns = [ok[m][str(p)][0] for p in ps]
        ax.plot(ps, ns, 'o-', color=col, lw=1.9,
                label=f"{m.split('/')[-1]} ({PARAMS_M.get(m,'?')}M)")
    ax.set_xlabel('string length $p$ (tokens)'); ax.set_ylabel('knee $N^*$')
    ax.set_title('Knee vs length, across model size\n(lower = easier to plant)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # knee vs params, averaged over lengths
    xs, ys = [], []
    for m in order:
        vals = [ok[m][p][0] for p in ok[m] if ok[m][p][0] is not None]
        if vals:
            xs.append(PARAMS_M.get(m, np.nan)); ys.append(np.mean(vals))
    ax2.plot(xs, ys, 'o-', color='crimson', lw=1.9)
    for x, y, m in zip(xs, ys, order):
        ax2.annotate(m.split('/')[-1], (x, y), fontsize=7,
                     textcoords='offset points', xytext=(4, 4))
    ax2.set_xscale('log'); ax2.set_xlabel('parameters (millions)')
    ax2.set_ylabel('mean knee $N^*$ (over lengths)')
    ax2.set_title('Does the knee move with scale?')
    ax2.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(f"{args.prefix}.png", dpi=130)
    print(f"wrote {args.prefix}.png")


if __name__ == "__main__":
    main()
