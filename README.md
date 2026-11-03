# transformer-context-contagion

Empirical test, on real language models, of the **context-contagion** conjecture worked
out exactly on a toy Markov model in
[`trust-is-the-attack-surface`](https://github.com/shumow/trust-is-the-attack-surface).

That project showed, on a global-plus-cache Markov model where the relevant objects are
*exact*, that self-reproducing token strings (quines) exist, can be planted in a cache for
a poison budget cheapest at a condensation knee, and spread between caches as a
context-window worm with a reproduction number `R0 ≈ T/(p·k_thresh)` that falls with
string length — and that all of this is **cheapest and most transmissible under a
longest-match (induction-head) surrogate**. Every claim was stated as a *conjecture* for
real transformers. This repository is where those conjectures get confirmed, refuted, or
bounded.

This repo is **conceptually downstream** of the toy project but is not a code dependency:
the numpy Markov code does not transfer; only the analysis vocabulary does, and that lives
in the two write-ups there (`trust_is_the_attack_surface.tex`, `contagious_context.tex`).

## Status

The record of every run is [`results_log.md`](results_log.md); the write-up is
[`latex/transformer_context_contagion.tex`](latex/transformer_context_contagion.tex) (the
committed PDF predates the November 2026 text revision). A critical review on 2026-11-03
re-read several results; the work plan is
[`review_remediation_plan.md`](review_remediation_plan.md). One line per claim, with where
the detail lives:

| claim | status | see |
|---|---|---|
| C1/C3 copy threshold and length-robustness | For spans of ≥3 tokens, one prior occurrence suffices more often than not in 36 of 48 cells, 355M–7B, four families, base and instruct. The interpolated "knee" for those payloads is the grid floor, so its sharpness and flatness with scale are not findings. Threshold behavior proper is at p=1–2 (frequency regime). | log: E1.1–E1.5, E1.3/E1.4 at scale, *Analyses* §A |
| C2 induction-head mechanism | Ablating the top ~5% of heads by induction score collapses reproduction; random heads do not. 124M–7B, three families. The scale-matched ablation (not a fixed count) is the methodological result. | log: E2, E2 at scale |
| C4 trust, not usefulness | Holds on the GPT-2-small proxy. The at-scale real-task version had repeating "non-repetitive" cells and a copy-rate that is ~1 by construction; redesigned, **rerun pending**. | log: E4-lite, E4-scale (status note) |
| C5 RAG hijack | Holds on a base-model continuation (entry × lock-in); **bounded** at scale: a task query on an instruct model defuses raw-repetition hijack. | log: E5-lite, E5-scale |
| C6 worm / firebreak | Re-analyzed: no cell has a per-hop reproduction ratio above one; "endemic" 7B chains are locked repetition loops at a load set by T/p. Firebreak: significant in the reported 1-hop run (p=0.004), undetectable in the unreported 3-hop run (p=0.24). Higher-n rerun queued. | log: E6, E6-scale, *Analyses* §B–C |
| E7 detector | **Reran (32 negatives + model-free n-gram baseline + FPR):** the induction detector does **not** beat a model-free n-gram-repeat count — both separate poisoned from clean perfectly (AUC 1.0 at every k≥2), and the induction score is marginally *worse* on benign-repetitive text (FPR 1.00 vs 0.88). It is an n-gram repeat detector; induction explains *why* repetition is dangerous, not a better alarm. | log: E7 rerun (2026-11-03) |
| E8 self-extraction | Instruct models decode and execute nested benign substitution. **E8-lite reran with a true random control:** substitution is still induction-carried at reliable table sizes (m=8 ablation drop −57% vs a true-random −33%; m=16 −68% vs −3%), but weaker than first reported — a genuine random ablation removes ~33% (the old "random" was the lowest-induction heads, −6%). Cascade limit unchanged (2-step 0.19 < f²=0.25). | log: E8-lite rerun (2026-11-03), E8 capstone |

Sample sizes are small throughout (3–8 payloads, facts or trials per cell, single seeds); read
every number with its interval where one is given, and as a point estimate where none is.

## Safety

This is **defensive** research into a context-poisoning / self-replicating-prompt threat
that already exists against real RAG and agent pipelines. It studies the *copy dynamics*
of out-of-distribution **nonsense** payloads only — never harmful instructions or
jailbreaks — against models we host or rate-limited APIs under their terms, never against
third-party production systems. See [`SECURITY.md`](SECURITY.md) for the full posture
(synthetic non-harmful payloads, no production targets, disclosure-first, dual-use
hygiene). Read it before extending this repo.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # installs torch -- pick the wheel matching your CUDA/platform
```

GPU is recommended for the open-weights mechanistic tier (GPT-2, Pythia, Llama-3.2,
Qwen2.5, Gemma-2); the smallest models (GPT-2-small, Pythia-410M) run on CPU for
iteration.

## Run the keystone experiment

```bash
python -m experiments.e1_condensation_knee --model gpt2 --lengths 1 2 3 5 8 --reps 1 2 4 8 16 32
```

Builds an out-of-distribution token string, repeats it `N` times in context, and measures
whether the model regenerates it — the real-transformer analogue of the toy's condensation
knee. Writes `results/e1_knee.png` and a table of `P(reproduce)` vs `N` per string length.

## Layout

| Path | What |
|---|---|
| `transformer_validation_plan.md` | The core research plan: claims C1–C6, experiments E1–E6, phasing, threats to validity. |
| `trackC_scale_plan.md` | The downstream-at-scale plan (C4/C5/C6 on instruct/RAG/multi-tokenizer) and the running open-experiments list. |
| `e1_plan.md` | Detailed work plan for E1 (the condensation knee): factors, controls, statistics, staged runs. |
| `results_log.md` | Running log of every empirical run (mirrors the toy project's `contagion_notes.md`). |
| `latex/transformer_context_contagion.tex` | Progress-report write-up of the current state (built PDF alongside). |
| `tcc/payloads.py` | Token-level out-of-distribution payload construction. |
| `tcc/scoring.py` | Transition-level reproduction scoring + knee location (numpy-only, self-testing). |
| `tcc/corpus.py` | Benign sentence / paragraph banks for controls: distinct non-repetitive prose (E4-scale), clean and benign-repetitive negatives (E7). Pure Python. |
| `tcc/models.py` | Minimal model loader / generation wrapper (HF; TransformerLens added for E2). |
| `tcc/interp.py` | Induction-head scoring / ablation helpers (TransformerLens) used by E2/E4/E7. |
| `experiments/e1_condensation_knee.py` | E1 keystone: P(reproduce) vs repetition count, per string length. |
| `experiments/e1_2_cue.py`, `e1_3_size_sweep.py`, `e1_5_context.py` | Cue-dependence, size sweep, and context-length/recency controls. |
| `experiments/e2_induction.py` | E2: induction-head identification + causal ablation (with `--k-frac` for scale). |
| `experiments/e4_trust.py`, `e4_scale.py` | C4: poisonability vs trust vs usefulness (proxy, then a real task at scale). |
| `experiments/e5_hijack.py`, `e5_scale.py` | C5: in-document hijack (GPT-2), then long-context instruct-RAG. |
| `experiments/e6_worm.py`, `e6_scale.py` | C6: serial-passage worm + cross-tokenizer firebreak (GPT-2, then at scale / 5-tokenizer matrix). |
| `experiments/e7_detector.py` | E7: the paired bound-trust detector (defensive complement). |
| `experiments/e8_substitution.py`, `e8_capstone.py` | E8: in-context substitution primitive, then coherent self-extraction on instruct models. |
| `deploy/` | Provider-agnostic GPU bootstrap + Azure/RunPod run scripts for the at-scale tiers. |
| `tests/test_scoring.py`, `tests/test_corpus.py` | Unit checks for the scorer and the text banks (no torch needed; `python -m tests.test_scoring`); most `experiments/*.py` also have a `--selftest`. |
| `review_remediation_plan.md` | Work plan from the 2026-11-03 critical review: which results are being re-measured and why. |
| `SECURITY.md` | Defensive scope, dual-use posture, disclosure policy. |
