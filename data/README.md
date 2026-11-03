# Committed data snapshots

`results/` is **gitignored** (outputs are meant to be reproducible from `experiments/*.py`).
That is fine for runs anyone can cheaply regenerate locally, but it is *not* fine for at-scale
runs produced on ephemeral cloud GPUs (RunPod/Azure): when the pod is torn down, the only record
is whatever was hand-transcribed into `results_log.md`.

This directory is the **durable** home for the raw JSON (and key PNGs) of runs that are expensive
or impossible to reproduce locally. Unlike `results/`, it is tracked by git.

## Convention
- One subdirectory per track/run set: `data/trackA/`, `data/trackB/`, `data/trackC/`.
- Commit the exact `results/<exp>.json` the log tables were transcribed from, unmodified.
- Note the pod/GPU, date, and the exact command in the accompanying entry in `results_log.md`.
- Pod run scripts (`deploy/run_*.sh`) should copy `results/*.json` here as their last step, so
  this never again depends on a manual pull before teardown.
- A file whose name ends in `_transcribed.json` (or carries a `_provenance` field) is **not raw
  output**: it was typed back in from a table in `results_log.md` or a run log because the raw
  JSON was lost. Treat its numbers as the log's numbers, with the log's rounding.
- `analysis_*.json` files are outputs of `experiments/analysis_*.py` run on the files beside them.

## What is here (2026-11-03)
| file | status |
|---|---|
| `trackC/e4_scale.json`, `e5_scale.json`, `e6_scale.json` (+ PNGs, `logs/`) | raw, from the RunPod L40S, 2026-07-02 (`e6_scale.json` holds the **second** Part-B run only: 1 hop, 32 seed copies, 8 trials) |
| `trackC/e6_scale_partB_3hop_reps16.json` | transcribed from `logs/e6.log`: the **first** Part-B run (3 hops, 16 seed copies, 5 trials), overwritten on the pod |
| `trackA/e6_worm_transcribed.json` | transcribed from `results_log.md` (E6 on GPT-2 / Pythia-160M); raw file not recovered |
| `trackB/e1_3_e1_4_knees_transcribed.json` | transcribed from `results_log.md` (E1.3 / E1.4 knee tables); raw files on the Azure VM disk were not recovered |
| `trackB/analysis_knee_floor.json`, `trackC/analysis_e6_firebreak.json`, `trackC/analysis_e6_r0.json` | analysis outputs, see `results_log.md` "Analyses of committed data" |

## Recovered from the dev Mac (2026-11-03)
The original Track A/B runs' JSON/PNG/log files were found still present in the gitignored
`results/` directory on the author's MacBook and copied here unmodified (`cp -p`, byte-identical).
| file | status |
|---|---|
| `trackA/e1_1_gpt2.json`, `e1_2_gpt2.json`, `e1_3.json`, `e1_5_gpt2.json`, `e2_gpt2.json`, `e4_trust.json`, `e5_hijack.json`, `e6_worm.json`, `e7_detector.json`, `e8_substitution.json` (+ their PNGs and `*_run.log`) | **raw**, recovered 2026-11-03 from `results/`; original local GPT-2-small / Pythia-160m runs, 2026-06-25–29 |
| `trackA/e6_worm.json` | raw E6 worm; **supersedes** `trackA/e6_worm_transcribed.json` (transcription kept per convention, now superseded by this raw file) |
| `trackA/e1_greedy_gpt2.json` (+ `e1_greedy_*.png`, `e1_knee.png`) | raw GPT-2 greedy-decoding companion run to E1.1; recovered 2026-11-03; was not on the missing list below but preserved as raw output |
| `trackB/e1_3_trackb_t4.json` (+ PNG), `e1_3_llama_t4.json`, `e2_gpt2-xl.json`, `e2_EleutherAI_pythia-1.4b.json`, `e2_EleutherAI_pythia-2.8b.json`, `e2_EleutherAI_pythia-6.9b.json`, `e2_Qwen_Qwen2.5-7B.json`, `e8_capstone.json` | **raw**, recovered 2026-11-03 from `results/`; T4/L40S cloud runs scp'd to the Mac, 2026-06-30–07-02 |
| `trackB/e1_3_trackb_t4.json` | raw E1.3 scale-knee run; **partially supersedes** `trackB/e1_3_e1_4_knees_transcribed.json` (the E1.3 half only — E1.4 raw is still missing, so the transcription is kept) |

**Note on `trackA/e7_detector.json`:** the E7 rerun (2026-11-03, Task 2) writes to this same
filename, so `trackA/e7_detector.json` (+ `.png`) now holds the **32-negative rerun**, not the
recovered June run. The recovered June 3-negative run is preserved beside it as
`trackA/e7_detector_2026-06-26_3neg.json` (+ `.png`). See `results_log.md`, "E7 rerun".

**Note on `trackA/e8_substitution.json`:** likewise the E8-lite rerun (2026-11-03, Task 3) writes
to this filename, so `trackA/e8_substitution.json` (+ `.png`) now holds the **true-random-control
rerun**. The recovered June run is preserved as `trackA/e8_substitution_2026-06-26.json` (+ `.png`).
See `results_log.md`, "E8-lite rerun".

## Still missing (raw data lost unless recovered from the Azure disk)
Track A: none — all recovered 2026-11-03 (see above).
Track B: `e1_4` only (searched the dev Mac 2026-11-03, not found; its numbers remain in
`trackB/e1_3_e1_4_knees_transcribed.json`, "from run log, raw data lost"). If it survives on the
Azure VM disk, commit it here unmodified.
