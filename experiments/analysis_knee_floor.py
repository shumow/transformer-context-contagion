"""
Knee-floor analysis: is the "condensation knee" a measurement or the grid floor?

For a cyclic payload S of length p >= 2, the transition S[-1] -> S[0] has NEVER occurred in
a context of N=1 copies, so a copy mechanism cannot reproduce S at N=1 except by chance;
at N=2 that transition has occurred exactly once. Any induction-type mechanism therefore
jumps between N=1 and N=2, and the interpolated P=1/2 crossing N* lands in (1, 2] -- the
first grid interval -- whenever the ONE-SHOT copy rate P(reproduce | N=2) is >= 1/2. In
that regime N* carries no information beyond that rate: a "sharp knee at N*~1.6 that is
flat across scale" is the statement "one prior occurrence suffices in every model", which
is the definition of an induction head doing its job.

This script reads any E1 result file and reports, per (model, p):
  * N*, whether it is at the floor (N* <= second grid point),
  * the one-shot rate P(N=2) directly when the file has curves (E1.1 format), or the
    rate IMPLIED by N* via the inverse of the linear interpolation when only knees were
    recorded (E1.3 / transcribed format; assumes P(N=1)=0, which the E1.1/E1.5 curves show),
and a summary: the fraction of p >= 3 cells at the floor.

  python -m experiments.analysis_knee_floor data/trackB/e1_3_e1_4_knees_transcribed.json
  python -m experiments.analysis_knee_floor results/e1_1_gpt2.json
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np

from tcc import scoring


def rows_from_e11(d, model=None):
    """e1_condensation_knee.py output: grid[p] = {reps, main: [(phat,lo,hi)...], knee, ...}."""
    out = []
    for p, g in d["grid"].items():
        reps, probs = g["reps"], [m[0] for m in g["main"]]
        n_star = g.get("knee")
        osr = scoring.one_shot_rate(reps, probs)
        out.append(dict(model=model or d.get("model", "?"), p=int(p), n_star=n_star,
                        at_floor=scoring.knee_at_floor(n_star, reps),
                        p_n1=None if osr is None else osr[0], p_n2=None if osr is None else osr[1],
                        implied_p_n2=scoring.implied_one_shot_rate(n_star, reps[:2]), source="curve"))
    return out


def rows_from_e13(d):
    """e1_3_size_sweep.py output: results[model][p] = [nstar, lo, hi]; reps at top level."""
    out = []
    reps = d["reps"]
    for model, res in d["results"].items():
        if "error" in res:
            continue
        for p, (n_star, lo, hi) in res.items():
            out.append(dict(model=model, p=int(p), n_star=n_star,
                            at_floor=scoring.knee_at_floor(n_star, reps), p_n1=None, p_n2=None,
                            implied_p_n2=scoring.implied_one_shot_rate(n_star, reps[:2]), source="knee-only"))
    return out


def rows_from_transcribed(d):
    out = []
    reps = d["reps"]
    for group in ("base", "instruct"):
        for model, res in d.get(group, {}).items():
            for p, n_star in res.items():
                if p == "params_M":
                    continue
                out.append(dict(model=f"{model} [{group}]", p=int(p), n_star=n_star,
                                at_floor=scoring.knee_at_floor(n_star, reps), p_n1=None, p_n2=None,
                                implied_p_n2=scoring.implied_one_shot_rate(n_star, reps[:2]), source="transcribed"))
    return out


def load_rows(path):
    d = json.load(open(path))
    if "grid" in d:
        return rows_from_e11(d)
    if "results" in d:
        return rows_from_e13(d)
    if "base" in d or "instruct" in d:
        return rows_from_transcribed(d)
    raise ValueError(f"{path}: unrecognized E1 result format")


def fmt(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def summarize(rows, min_p=3):
    multi = [r for r in rows if r["p"] >= min_p and r["n_star"] is not None]
    n_floor = sum(r["at_floor"] for r in multi)
    implied = [r["implied_p_n2"] for r in multi if r["implied_p_n2"] is not None]
    direct = [r["p_n2"] for r in multi if r["p_n2"] is not None]
    return dict(n_cells=len(multi), n_at_floor=n_floor,
                frac_at_floor=(n_floor / len(multi)) if multi else None,
                implied_one_shot_median=float(np.median(implied)) if implied else None,
                implied_one_shot_min=float(min(implied)) if implied else None,
                direct_one_shot_median=float(np.median(direct)) if direct else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--min-p", type=int, default=3, help="smallest p counted as a multi-token payload")
    ap.add_argument("--out", default=None, help="write the table + summary as JSON here")
    args = ap.parse_args()

    rows = []
    for path in args.paths:
        rows += load_rows(path)
    print("| model | p | N* | at floor? | P(N=1) | P(N=2) | implied P(N=2) from N* |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        imp = r["implied_p_n2"]
        if imp is not None and imp > 1.0:
            r["implied_p_n2"] = None
            r["note"] = f"N*={r['n_star']} < 1.5 implies P(N=1) > 0 in this cell; inversion invalid"
            imp_s = "n/a (P(N=1)>0)"
        else:
            imp_s = fmt(imp)
        print(f"| {r['model']} | {r['p']} | {fmt(r['n_star'], 1)} | {'**yes**' if r['at_floor'] else 'no'} | "
              f"{fmt(r['p_n1'])} | {fmt(r['p_n2'])} | {imp_s} |")
    s = summarize(rows, args.min_p)
    print(f"\nmulti-token cells (p >= {args.min_p}): {s['n_cells']}; at the grid floor (N* <= 2): "
          f"{s['n_at_floor']} ({100 * (s['frac_at_floor'] or 0):.0f}%)")
    if s["implied_one_shot_median"] is not None:
        print(f"implied one-shot copy rate P(N=2) over floor cells: median {s['implied_one_shot_median']:.2f}, "
              f"min {s['implied_one_shot_min']:.2f}  (assumes P(N=1)=0)")
    if s["direct_one_shot_median"] is not None:
        print(f"direct one-shot copy rate P(N=2) (curves present): median {s['direct_one_shot_median']:.2f}")
    print("\nReading: a cell 'at floor' means the curve crosses 1/2 inside the first grid interval, i.e. one\n"
          "prior occurrence of the span already suffices more often than not. For those cells N* is a\n"
          "re-parameterization of the one-shot rate, not an independent threshold measurement.")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(dict(rows=rows, summary=s, min_p=args.min_p, sources=args.paths), open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
