# Committed data snapshots

`results/` is **gitignored** (outputs are meant to be reproducible from `experiments/*.py`).
That is fine for runs anyone can cheaply regenerate locally, but it is *not* fine for at-scale
runs produced on ephemeral cloud GPUs (RunPod/Azure): when the pod is torn down, the only record
is whatever was hand-transcribed into `results_log.md`.

This directory is the **durable** home for the raw JSON (and key PNGs) of runs that are expensive
or impossible to reproduce locally. Unlike `results/`, it is tracked by git.

## Convention
- One subdirectory per track/run set, e.g. `data/trackC/`.
- Commit the exact `results/<exp>.json` the log tables were transcribed from, unmodified.
- Note the pod/GPU, date, and the exact command in the accompanying entry in `results_log.md`.
- Pod run scripts (`deploy/run_*.sh`) should copy `results/*.json` here as their last step, so
  this never again depends on a manual pull before teardown.

## Known gap (2026-07-02)
The Track C raw JSONs — `e4_scale.json`, `e5_scale.json`, `e6_scale.json` — are **not yet here**
and were not found on the local machine; they must be recovered from the RunPod pod, or the
scale experiments re-run (they are deterministic: greedy, fixed seed). See the P0 item in
`trackC_scale_plan.md`.
