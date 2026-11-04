"""
Cross-tokenizer firebreak: is the same-tokenizer diagonal actually stronger?

Reads one or more E6 Part-B matrices (rows = patient-zero tokenizer family, columns = host
family, entry = mean surviving load) and tests the claim "same-tokenizer transmission is
strongest" with statistics that do not depend on eyeballing:

  * diagonal mean vs off-diagonal mean, and their ratio;
  * per row, the rank of the diagonal entry among the row's entries (1 = largest);
  * an EXACT permutation test. Null: the tokenizer identity of the host is irrelevant, so
    within each row every column is exchangeable and the "diagonal" is just one of the
    columns. Enumerate all column^rows assignments of a "diagonal" position per row (5^5 =
    3125 for a 5x5 matrix) and compute the fraction whose diag-minus-offdiag mean is >= the
    observed one. That fraction is the one-sided p-value.

Two matrices were run on 2026-07-02 (both are in data/trackC/logs/e6.log); only the second
was reported. This script is meant to be run on BOTH:

  python -m experiments.analysis_e6_firebreak \\
      data/trackC/e6_scale_partB_3hop_reps16.json data/trackC/e6_scale.json
"""
from __future__ import annotations
import argparse
import itertools
import json
import os
import numpy as np


def matrix_from(d):
    cross = d["cross"]
    fams = list(cross)
    M = np.array([[float(cross[X][Y]) for Y in fams] for X in fams])
    return fams, M


def diag_stat(M, pos):
    """Mean of the chosen 'diagonal' entries minus mean of the rest, for one position per row."""
    n = M.shape[0]
    mask = np.zeros_like(M, bool)
    for i, j in enumerate(pos):
        mask[i, j] = True
    return float(M[mask].mean() - M[~mask].mean())


def analyze(fams, M):
    n = M.shape[0]
    obs = diag_stat(M, range(n))
    ranks = []
    for i in range(n):
        row = M[i]
        ranks.append(int(1 + np.sum(row > row[i]) + 0.5 * (np.sum(row == row[i]) - 1)))
    stats = [diag_stat(M, pos) for pos in itertools.product(range(n), repeat=n)]
    p_val = float(np.mean([s >= obs - 1e-12 for s in stats]))
    diag = np.diag(M)
    off = M[~np.eye(n, dtype=bool)]
    return dict(fams=fams, diag_mean=float(diag.mean()), offdiag_mean=float(off.mean()),
                ratio=float(diag.mean() / off.mean()) if off.mean() > 0 else None,
                diag_minus_off=obs, p_perm_one_sided=p_val, n_assignments=len(stats),
                rows_diag_is_max=int(sum(r == 1 for r in ranks)), diag_ranks=ranks,
                # host (column) and patient-zero (row) marginal means: is transmission set by
                # WHO hosts rather than by whether the tokenizer matches?
                col_means={f: float(M[:, j].mean()) for j, f in enumerate(fams)},
                row_means={f: float(M[i, :].mean()) for i, f in enumerate(fams)},
                diag_zero_rows=[fams[i] for i in range(n) if diag[i] == 0],
                offdiag_exceeding_any_diag=[(fams[i], fams[j], float(M[i, j]))
                                            for i in range(n) for j in range(n)
                                            if i != j and M[i, j] > diag[i]])


def bootstrap_ratio(d, rng, n_boot=2000):
    """If the file carries per-trial surviving loads ('cross_trials', written by e6_scale.py
    since 2026-11), resample trials within every cell and return a 95% interval on the
    diagonal/off-diagonal ratio and on the diag-minus-off difference."""
    ct = d.get("cross_trials")
    if not ct:
        return None
    fams = list(ct)
    cells = {(X, Y): np.asarray(ct[X][Y], float) for X in fams for Y in fams}
    ratios, diffs = [], []
    for _ in range(n_boot):
        M = np.array([[cells[(X, Y)][rng.integers(0, len(cells[(X, Y)]), len(cells[(X, Y)]))].mean()
                       for Y in fams] for X in fams])
        diag = np.diag(M).mean(); off = M[~np.eye(len(fams), dtype=bool)].mean()
        ratios.append(diag / off if off > 0 else np.nan); diffs.append(diag - off)
    return dict(n_trials_per_cell=int(len(next(iter(cells.values())))),
                ratio_ci=[float(np.nanpercentile(ratios, 2.5)), float(np.nanpercentile(ratios, 97.5))],
                diag_minus_off_ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="JSON files with a 'cross' matrix (and optional 'config')")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    results = {}
    for path in args.paths:
        d = json.load(open(path))
        fams, M = matrix_from(d)
        r = analyze(fams, M)
        cfg = d.get("config") or d.get("cross_config") or {}     # e6_scale.py writes cross_config
        boot = bootstrap_ratio(d, rng)
        results[path] = dict(config=cfg, bootstrap=boot, **r)
        print(f"\n=== {path}  "
              f"(hops={cfg.get('cross_hops', '?')}, host0-reps={cfg.get('host0_reps', '?')}, "
              f"trials={cfg.get('trials', '?')}) ===")
        print("      " + "  ".join(f"{Y:>6s}" for Y in fams))
        for i, X in enumerate(fams):
            print(f"{X:>6s}" + "  ".join(f"{M[i, j]:6.1f}" for j in range(len(fams))) + f"   diag rank {r['diag_ranks'][i]}/{len(fams)}")
        print(f"diag mean {r['diag_mean']:.2f}  off-diag mean {r['offdiag_mean']:.2f}  "
              f"ratio {('%.2f' % r['ratio']) if r['ratio'] is not None else 'n/a'}  "
              f"diag-off {r['diag_minus_off']:+.2f}")
        print("host (column) means: " + "  ".join(f"{f} {v:.1f}" for f, v in r["col_means"].items())
              + "   | patient-zero (row) means: " + "  ".join(f"{f} {v:.1f}" for f, v in r["row_means"].items()))
        print(f"rows where diagonal is the max: {r['rows_diag_is_max']}/{len(fams)}; "
              f"diagonal = 0 in rows: {r['diag_zero_rows'] or 'none'}")
        print(f"exact permutation p (one-sided, {r['n_assignments']} assignments): {r['p_perm_one_sided']:.3f}")
        if boot:
            print(f"bootstrap over trials ({boot['n_trials_per_cell']}/cell): ratio 95% CI "
                  f"[{boot['ratio_ci'][0]:.2f}, {boot['ratio_ci'][1]:.2f}]; diag-off CI "
                  f"[{boot['diag_minus_off_ci'][0]:+.2f}, {boot['diag_minus_off_ci'][1]:+.2f}]")
        if r["offdiag_exceeding_any_diag"]:
            print("off-diagonal entries exceeding their row's diagonal: " +
                  ", ".join(f"{a}->{b} {v:.1f}" for a, b, v in r["offdiag_exceeding_any_diag"]))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


def selftest():
    fams = list("abcde")
    I = np.eye(5) * 10 + 1                      # strong diagonal
    r = analyze(fams, I)
    assert r["rows_diag_is_max"] == 5 and r["p_perm_one_sided"] == 1 / 3125, r["p_perm_one_sided"]
    rng = np.random.default_rng(0)
    R = rng.uniform(0, 10, (5, 5))              # no structure -> p not small
    r2 = analyze(fams, R)
    assert 0.0 < r2["p_perm_one_sided"] <= 1.0
    flat = np.ones((5, 5))
    assert analyze(fams, flat)["p_perm_one_sided"] == 1.0
    print("selftest OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
