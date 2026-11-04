# Brief for the RunPod agent: follow-up 3 (two runs)

Written 2026-11-05 by the reviewing session after merging iteration 9. Two short runs, one pod,
roughly 40 pod-minutes. Everything is implemented and selftested on the parent branch; this is a
launch-and-report job. Read `review_remediation_plan.md` (Progress) and the "Follow-ups 2"
entry of `results_log.md` first; the rawpad control is the one that entry proposed and did not run.

## Why

**(e) E5 `rawpad`.** Follow-ups 2 §D found that neither the templated query, the templated
"Continue the documents", nor the untemplated query hijacks, while the bare prefix does. But
every non-hijacking condition puts text *after* the payload block, so the data cannot separate
"the model follows the task" from "the payload is no longer the last thing in context" (the
recency gate of E1.5 and E5-lite). The paper's C5 sentence depends on which it is.

`rawpad` feeds the raw documents followed by **filler of exactly the token length of the query
tail** (`"\n\nQuestion: <query>"`), with no question and no template, in two forms per query
style: `rawpad_<q>` is benign prose (distinct sentences not used in the dump), `rawpadtok_<q>`
is neutral random tokens as in E1.5. Both are built at the token level so the length matches.

- If rawpad hijacks like `raw` (0.4–1.0): the query does semantic work; "a task query defuses
  raw-repetition hijack" stands as a statement about task-following.
- If rawpad does not hijack (≈0 like rawq): the C5 bound is the recency gate restated; the paper
  should say "anything that ends the context after the payload defuses it" and drop
  task-following as the mechanism.
- If prose pad and token pad differ, say which, and by how much; that is a third answer
  (coherent text competes, arbitrary tokens do not), also worth having.

**(f) E4 v2 on Llama-3.2-3B-Instruct at `--probe-k 1`.** The probe_k=2 run put Llama at the
ceiling (it reproduced a two-copy probe in 69–72% of *non-repetitive* contexts), which compressed
its correlations. One occurrence of the probe should separate its cells. Six minutes.

## What to run

Code branch and results branch as before (parent `claude/fervent-ritchie-l5m9uw`, which now
contains the `followup3` job and the `rawpad` control). Selftests cover the control
(`python -m experiments.e5_scale --selftest`); the pod runs them before the job as usual.

```
python deploy/runpod_pod.py create --job followup3 --branch claude/fervent-ritchie-l5m9uw \
    --results-branch claude/fervent-ritchie-l5m9uw-iter11-pod-followup3 --max-hours 1.5
```

The job runs, in order:
```
python -m experiments.e5_scale --model <M> --context-tokens 4000 --device cuda \
    --controls raw rawpad --prefix results/e5_scale_v2_rawpad_<tag> --snapshot trackC
#   <M> = Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3.2-3B-Instruct, microsoft/Phi-3.5-mini-instruct
python -m experiments.e4_scale --model meta-llama/Llama-3.2-3B-Instruct --probe-k 1 \
    --device cuda --prefix results/e4_scale_v2_pk1_llama3b-instruct --snapshot trackC
```
Same seed as v2 and rawq, so the contexts and the query cells are identical to those runs
(check: every query-cell mean must match `e5_scale_v2_<tag>.json` exactly, as it did for rawq).
`raw` is included so `control_ok` is meaningful; `chat` and `rawq` are not repeated.

## What to report

A `results_log.md` entry `## Follow-ups 3 — E5 rawpad control; E4 v2 Llama at probe_k=1 (<date>, RunPod L40S)`:

1. For each model and k ∈ {2, 8, 32}: `raw`, `rawpad_<q>` and `rawpadtok_<q>` for the three
   query styles, as mean occupancy with the hijacked fraction in parentheses, next to the v2
   query cell and the rawq cell for the same (model, k) so all five conditions sit in one table.
   Note the pad lengths (`rawpad_len` in the JSON) per model.
2. One sentence of verdict per the three outcomes above, stated at the strength the data support
   (8 trials per cell; give the largest single rawpad trial as in the rawq entry).
3. The E4 Llama table at probe_k=1 beside the probe_k=2 numbers: 2×2 means with Wilson on
   P_probe, and the four correlations with their bootstrap intervals. Say whether the ceiling is
   gone (non-repetitive P_probe well below 0.69) and whether corr(P_probe, T) moved.
4. Update the C5 row of the README table and the C5 bullet of `ONEPAGER.md` to the verdict;
   add an iteration-11 row to the plan's Progress table; append §9 to `TIER2_AUDIT_NOTES.md`
   with pod id, wall time, cost and the derivation of each number, as §8 does.

Squash the pod branch into one commit as in iteration 9, push the work branch, and hold it for
review as before rather than merging into the parent.

## Rules (unchanged)

Defaults untouched; no result JSON edited by hand; numbers read from the JSON, not the console;
report whatever comes out, including a result that removes "task-following" from the paper;
one pod at a time; terminate it when the push lands; nothing to `main`, no force-push.
