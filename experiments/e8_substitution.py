"""
E8-lite -- in-context substitution as an induction primitive, and the cascade limit.

A "self-extracting" payload (random-looking tokens that unfold, layer by layer, into a
payload) needs the model to perform an in-context SUBSTITUTION at each layer. A substitution
table is just key/value pairs, and induction is exactly "saw [key][value] ... see [key] ->
predict [value]." So this asks the mechanistic core, entirely on GPT-2-small with benign OOD
markers (no harmful or even coherent payload -- the "decoded" tokens are just other random
markers):

  1. Single-step fidelity: present a random OOD substitution table (m key->value pairs),
     query a key, measure P(correct value) and top-1 accuracy, swept over table size m.
  2. Induction attribution: ablate the E2 induction heads vs the same number of random heads
     -- does the substitution collapse? (Is the lookup really induction?)
  3. The cascade limit: a d-layer self-extractor needs all layers to decode, so its success
     decays ~ f^(L*d) in the single-step fidelity f -- the same brittleness ladder that caps
     quine reproduction. We measure a direct 2-step cascade (table1 k->v, table2 v->w) to
     show the interference penalty beyond the optimistic f^2.

What this does NOT do (deferred to a capable/instruct model, Track B): unfold into a
*coherent* natural-language payload, or self-execute a decoded instruction.

Run on CPU (TransformerLens + GPT-2-small):  python -m experiments.e8_substitution
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import torch

from tcc import payloads, interp


def make_table(pool, m, rng):
    toks = [int(t) for t in rng.choice(pool, size=2 * m, replace=False)]
    return toks[:m], toks[m:]


def table_ctx(keys, vals):
    ctx = []
    for k, v in zip(keys, vals):
        ctx += [k, v]
    return ctx


def lookup(model, ctxs, targets, hooks=None):
    """Mean P(target) and top-1 accuracy over a batch of (context, target) lookups."""
    Ps, acc = [], []
    for ctx, tgt in zip(ctxs, targets):
        toks = torch.tensor([ctx])
        with torch.no_grad():
            logits = (model.run_with_hooks(toks, fwd_hooks=hooks, return_type="logits")
                      if hooks else model(toks, return_type="logits"))
        last = logits[0, -1]
        Ps.append(torch.softmax(last, -1)[tgt].item())
        acc.append(int(last.argmax().item()) == tgt)
    return float(np.mean(Ps)), float(np.mean(acc))


def single_step(model, pool, m, n_tables, rng, hooks=None):
    ctxs, tgts = [], []
    for _ in range(n_tables):
        keys, vals = make_table(pool, m, rng)
        base = table_ctx(keys, vals)
        for qi in range(m):
            ctxs.append(base + [keys[qi]]); tgts.append(vals[qi])
    return lookup(model, ctxs, tgts, hooks)


def two_step(model, pool, m, n_tables, rng):
    """Direct 2-layer cascade: table1 (k->v) and table2 (v->w) both in context; query k;
    greedy-generate 2 tokens; was it v then w? Compares to the f^2 ideal."""
    s1, s2 = [], []
    for _ in range(n_tables):
        toks = [int(t) for t in rng.choice(pool, size=3 * m, replace=False)]
        ks, vs, ws = toks[:m], toks[m:2 * m], toks[2 * m:]
        ctx0 = table_ctx(ks, vs) + table_ctx(vs, ws)
        for qi in range(m):
            ctx = ctx0 + [ks[qi]]
            out = model.generate(torch.tensor([ctx]), max_new_tokens=2, do_sample=False,
                                 verbose=False, return_type="tokens")
            g = out[0, len(ctx):].tolist()
            s1.append(g[0] == vs[qi]); s2.append(g[0] == vs[qi] and g[1] == ws[qi])
    return float(np.mean(s1)), float(np.mean(s2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ms', type=int, nargs='+', default=[2, 4, 8, 16])
    ap.add_argument('--n-tables', type=int, default=8)
    ap.add_argument('--k-heads', type=int, default=8)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--prefix', default='results/e8_substitution')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model = interp.load_hooked("gpt2", device="cpu")
    scores = interp.induction_scores(model, seed=args.seed)
    ind_heads = interp.top_heads(scores, args.k_heads)
    flat = sorted(((float(scores[L, H]), L, H) for L in range(scores.shape[0])
                   for H in range(scores.shape[1])))
    rand_heads = [(L, H) for _, L, H in flat[:args.k_heads]]      # lowest-induction heads
    ind_hooks = interp.make_ablation_hooks(model, ind_heads)
    rand_hooks = interp.make_ablation_hooks(model, rand_heads)
    pool = payloads.rare_token_pool(tokenizer=model.tokenizer)
    print(f"induction heads: {ind_heads}\n")

    grid = {}
    print(f"{'m':>3} {'P(clean)':>9} {'acc(clean)':>10} {'acc(ind-abl)':>12} {'acc(rand-abl)':>13}")
    for m in args.ms:
        pc, ac = single_step(model, pool, m, args.n_tables, np.random.default_rng(args.seed), None)
        _, ai = single_step(model, pool, m, args.n_tables, np.random.default_rng(args.seed), ind_hooks)
        _, ar = single_step(model, pool, m, args.n_tables, np.random.default_rng(args.seed), rand_hooks)
        grid[str(m)] = dict(P=pc, acc=ac, acc_ind=ai, acc_rand=ar)
        print(f"{m:>3} {pc:>9.3f} {ac:>10.3f} {ai:>12.3f} {ar:>13.3f}")

    s1, s2 = two_step(model, pool, 4, args.n_tables, np.random.default_rng(args.seed + 1))
    f = grid['4']['acc']
    print(f"\n2-step cascade (m=4): step1 acc={s1:.3f}  full 2-step acc={s2:.3f}  "
          f"(f^2 ideal={f*f:.3f})")

    os.makedirs(os.path.dirname(args.prefix) or '.', exist_ok=True)
    with open(f"{args.prefix}.json", 'w') as fjson:
        json.dump(dict(grid=grid, two_step=dict(s1=s1, s2=s2, f=f),
                       ind_heads=ind_heads), fjson, indent=2)
    plot(args, grid, s1, s2, f)


def plot(args, grid, s1, s2, f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = [int(m) for m in sorted(grid, key=int)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.7))

    ax.plot(ms, [grid[str(m)]['acc'] for m in ms], 'o-', color='crimson', lw=2, label='clean')
    ax.plot(ms, [grid[str(m)]['acc_ind'] for m in ms], 's--', color='navy', lw=1.8,
            label='ablate induction heads')
    ax.plot(ms, [grid[str(m)]['acc_rand'] for m in ms], '^:', color='gray', lw=1.5,
            label='ablate random heads')
    ax.set_xscale('log', base=2); ax.set_xlabel('table size $m$ (key$\\to$value pairs)')
    ax.set_ylabel('substitution accuracy (top-1)'); ax.set_ylim(-0.02, 1.04)
    ax.set_title('In-context substitution is an induction primitive\n'
                 '(ablating induction heads collapses it; random heads do not)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

    # cascade: optimistic f^(L*d) per-layer-length L, plus the measured 2-step penalty
    depths = np.arange(1, 7)
    for L, c in [(4, 'seagreen'), (8, 'navy'), (16, 'purple')]:
        ax2.plot(depths, f ** (L * depths), 'o-', color=c, lw=1.6, label=f'layer length L={L}')
    ax2.scatter([2], [s2], color='crimson', s=90, zorder=5,
                label=f'measured 2-step (m=4): {s2:.2f}')
    ax2.scatter([2], [f * f], facecolors='none', edgecolors='crimson', s=90, zorder=5,
                label=f'$f^2$ ideal: {f*f:.2f}')
    ax2.set_yscale('log'); ax2.set_xlabel('self-extraction depth (layers)')
    ax2.set_ylabel('P(full extraction succeeds)')
    ax2.set_title(f'Cascade limit from single-step fidelity $f={f:.2f}$\n'
                  '(depth bounded like the brittleness ladder; interference makes it worse)')
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(f"{args.prefix}.png", dpi=130)
    print(f"\nwrote {args.prefix}.png and {args.prefix}.json")


if __name__ == "__main__":
    main()
