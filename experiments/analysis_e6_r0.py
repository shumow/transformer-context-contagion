"""
R0 and critical length from E6 serial-passage data.

The plan and the paper's abstract speak of a reproduction number R0 that falls with payload
length; no R0 was ever computed. This script computes what the data supports.

For a chain of hosts the per-hop viral load k_0 -> k_1 -> ... -> k_H is a trajectory of the
one-hop map k_out = f(k_in). A reproduction number is the growth factor of that map near the
extinction threshold; here we report two estimates per (model, p):

  R_geo   geometric mean of k_{t+1}/k_t over LIVING hops (both k_t >= 1 and k_{t+1} >= 1):
          the average per-hop multiplication while the payload is alive. Excludes the
          extinction step, which would otherwise drive the mean to ~0 and hide the decay rate;
  R_tail  k_{t+1}/k_t at the LAST hop with k_t >= 1 -- the step closest to threshold, which
          IS the extinction step (ratio 0) for chains that die.

and the derived verdicts: endemic (load >= 1 at the last hop), the hop of extinction, and
per model the largest p that stays endemic ("critical length" lower bound).

LIMITATION, stated in the output: the committed Track A/C files hold the MEAN load over
trials per hop, not per-trial trajectories, so these are ratios of means. A mean trajectory
can look endemic when 2 of 5 trials are locked in a loop at full load and 3 are extinct. The
per-trial data needed for a real R0 (with a bootstrap CI) is now stored by e6_worm.py and
e6_scale.py under 'runs'; when a file has it, this script uses it and reports the CI.

  python -m experiments.analysis_e6_r0 data/trackA/e6_worm_transcribed.json data/trackC/e6_scale.json
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np


def ratios(traj, floor=1.0, living_only=False):
    t = np.asarray(traj, float)
    return [(t[i + 1] / t[i]) for i in range(len(t) - 1)
            if t[i] >= floor and (not living_only or t[i + 1] >= floor)]


def summarize_traj(traj):
    t = np.asarray(traj, float)
    r_all = ratios(t)
    r_live = ratios(t, living_only=True)
    alive = t >= 1
    ext = next((i for i, a in enumerate(alive) if not a), None)
    return dict(R_geo=float(np.exp(np.mean(np.log(r_live)))) if r_live else None,
                R_tail=float(r_all[-1]) if r_all else None,
                endemic=bool(alive[-1]), extinction_hop=ext, last_load=float(t[-1]), n_hops=len(t) - 1)


def summarize_runs(runs, rng, n_boot=1000):
    """Per-trial trajectories -> R_geo per trial, survival, bootstrap CIs."""
    runs = np.asarray(runs, float)
    per_trial = [summarize_traj(tr)["R_geo"] for tr in runs]
    rg = [v for v in per_trial if v is not None]
    surv = float(np.mean(runs[:, -1] >= 1))
    boots_r, boots_s = [], []
    for _ in range(n_boot):
        pick = runs[rng.integers(0, len(runs), len(runs))]
        pr = [summarize_traj(tr)["R_geo"] for tr in pick]
        pr = [v for v in pr if v is not None]
        boots_r.append(np.median(pr) if pr else np.nan)
        boots_s.append(np.mean(pick[:, -1] >= 1))
    return dict(R_geo_median=float(np.median(rg)) if rg else None,
                R_geo_ci=[float(np.nanpercentile(boots_r, 2.5)), float(np.nanpercentile(boots_r, 97.5))] if rg else None,
                survive=surv, survive_ci=[float(np.percentile(boots_s, 2.5)), float(np.percentile(boots_s, 97.5))],
                n_trials=len(runs), fully_locked=int(np.sum(runs[:, -1] >= 0.9 * runs[:, -1].max())) if runs[:, -1].max() > 0 else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    results = {}
    for path in args.paths:
        d = json.load(open(path))
        serial = d.get("serial", {})
        print(f"\n=== {path} ===")
        has_runs = any("runs" in c for m in serial.values() if "error" not in m for c in m.values())
        print("per-trial trajectories: " + ("present (CIs reported)" if has_runs else
              "ABSENT -- ratios of MEAN loads only; no CI possible"))
        print("| model | p | load per hop | R_geo | R_tail | endemic | extinct at hop |")
        print("|---|---|---|---|---|---|---|")
        results[path] = {}
        for model, curves in serial.items():
            if "error" in curves:
                continue
            crit = None
            for p in sorted(curves, key=int):
                c = curves[p]
                s = summarize_traj(c["mean"])
                if "runs" in c:
                    s["per_trial"] = summarize_runs(c["runs"], rng)
                results[path].setdefault(model, {})[p] = s
                if s["endemic"]:
                    crit = int(p)
                loads = " → ".join(f"{v:.0f}" for v in c["mean"])
                rg = "–" if s["R_geo"] is None else f"{s['R_geo']:.2f}"
                rt = "–" if s["R_tail"] is None else f"{s['R_tail']:.2f}"
                extra = ""
                if "per_trial" in s:
                    pt = s["per_trial"]
                    extra = (f" (per-trial R_geo median {pt['R_geo_median']:.2f} "
                             f"[{pt['R_geo_ci'][0]:.2f},{pt['R_geo_ci'][1]:.2f}]; survive {pt['survive']:.2f} "
                             f"[{pt['survive_ci'][0]:.2f},{pt['survive_ci'][1]:.2f}], n={pt['n_trials']})") if pt["R_geo_median"] is not None else ""
                print(f"| {model} | {p} | {loads} | {rg} | {rt} | {'yes' if s['endemic'] else 'no'} | "
                      f"{'–' if s['extinction_hop'] is None else s['extinction_hop']} |{extra}")
            results[path][model]["_largest_endemic_p"] = crit
            print(f"|  ↳ {model}: largest p still endemic at the last hop = {crit if crit is not None else 'none'} |||||||")
    print("\nReading R_geo: < 1 means the mean load shrinks every hop (dies); ≈ 1 means a stable loop; "
          "> 1 is amplification. A constant load across hops (R_geo = 1.00 exactly) is the signature of hosts\n"
          "emitting T tokens of pure payload each: a fully locked repetition loop, not a growing epidemic.")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


def selftest():
    s = summarize_traj([16, 8, 4, 2, 1, 0.5, 0])
    assert abs(s["R_geo"] - 0.5) < 1e-9 and s["extinction_hop"] == 5 and not s["endemic"]
    assert abs(s["R_tail"] - 0.5) < 1e-9                       # last step from a living load (1 -> 0.5)
    s_die = summarize_traj([20, 10, 0])
    assert abs(s_die["R_geo"] - 0.5) < 1e-9 and s_die["R_tail"] == 0.0 and s_die["extinction_hop"] == 2
    s2 = summarize_traj([20, 20, 20, 20])
    assert abs(s2["R_geo"] - 1.0) < 1e-9 and s2["endemic"] and s2["extinction_hop"] is None
    s3 = summarize_traj([0, 0, 0])
    assert s3["R_geo"] is None and s3["extinction_hop"] == 0
    rng = np.random.default_rng(0)
    pt = summarize_runs([[10, 10, 10], [10, 0, 0], [10, 10, 10]], rng, n_boot=50)
    assert abs(pt["survive"] - 2 / 3) < 1e-9 and pt["n_trials"] == 3
    print("selftest OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
