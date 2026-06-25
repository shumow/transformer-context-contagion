# Results log

Running log of the empirical work, mirroring the toy project's `contagion_notes.md`.
Figures land in `results/` (gitignored; regenerate from the experiment scripts).

## Phase 0 smoke test — E1 condensation knee on GPT-2 (2026-06-25)

First end-to-end run of the harness. Environment: MacBook (Apple A18 Pro, MPS/Metal
backend), `torch 2.8.0`, `transformers 4.57.6`, model `gpt2` (124M).

```
python -m experiments.e1_condensation_knee --model gpt2 \
    --lengths 1 2 3 5 --reps 1 2 4 8 16 32 --trials 8 --max-new 48
```

**Result: a clear condensation knee exists on a real transformer.** `P(model
regenerates the OOD string)` rises sharply from 0 to 1 with the repetition count `N`
(`results/e1_knee.png`). Interpolated knees `N*`:

| string length p | knee N* |
|---|---|
| 1 | ~9.6 |
| 2 | ~4.0 |
| 3 | ~2.4 |
| 5 | ~3.3 |

**Preliminary readings** (against the plan's claims):
- **C1 (the knee exists):** supported — a clear threshold in `N`, not a gradual
  rise-from-zero; the real-transformer analogue of the toy's condensation transition.
- **C3 (length robustness):** supported — the knee does **not** climb with string
  length. Multi-token strings lock at `N≈2–4`, flat-to-decreasing in `p`, the opposite
  of the fixed-order count cache's brittleness ladder and consistent with the induction
  surrogate (toy milestone 3.1).
- **An inversion worth chasing:** the **single-token** payload is the *hardest*
  (knee ~10), while multi-token n-grams lock fastest. This is mechanistically
  consistent with induction: induction heads key on a distinctive
  "saw `[A][B]`, now see `[A]`, predict `[B]`" match, which a degenerate length-1
  pattern provides only weakly, whereas a longer novel n-gram gives a sharper, more
  unique key. Needs **E2** to confirm the mechanism is actually induction.

**Caveats — this is a smoke test, not a measured result.** `trials=8` (so each `P` is
coarse, ±~0.18 near 0.5), one payload per length (no payload/seed averaging),
`temperature=1.0` only (greedy untested), a single model at a single size. The knee
*values* are noisy; the *qualitative* phenomenon (a sharp, small, length-robust knee) is
clear.

**Next (Phase 1):** average over multiple payloads and seeds; sweep temperature and
contrast greedy; add **E2** induction-head attribution (activation patching / head
ablation) to test C2 and explain the p=1 inversion; scale across model sizes
(Pythia-410M and up) to see how the knee moves. The bigger sweeps want more than this
8 GB / 5-core device.

## E1.1 — measured knee on GPT-2, with controls (2026-06-25)

First *measured* run (the smoke test made into a measurement): GPT-2-small, 4 distinct
round-trip-stable payloads per length x 10 sampled continuations, `T=1.0`, with the
scrambled and no-payload controls and Wilson + bootstrap CIs.

```
python -m experiments.e1_condensation_knee --model gpt2 \
    --lengths 1 2 3 5 --reps 1 2 4 8 16 32 --payloads 4 --trials 10 --max-new 48
```

| p | knee N* (95% CI) | width | scrambled P@N=32 | no-payload P |
|---|---|---|---|---|
| 1 | 7.2 (5.5–10.3) | 7.8 | **0.97** | 0.00 |
| 2 | 4.3 (1.7–6.2)  | 5.2 | **0.70** | 0.00 |
| 3 | 3.0 (2.6–3.6)  | 1.6 | 0.05 | 0.00 |
| 5 | 1.9 (1.7–4.8)  | 3.2 | 0.00 | 0.00 |

**C1 (knee exists): confirmed.** Every length shows a clear knee, and the no-payload
control is `0.00` everywhere — the payloads are genuinely OOD, so reproduction is caused
by the repetitions, not the model's taste.

**C3 (length robustness): confirmed, and stronger than "flat."** `N*` *decreases* with
length, 7.2 → 4.3 → 3.0 → 1.9 (`results/e1_1_Nstar_gpt2.png`): longer strings are
*easier* to plant — the induction-like signature, the opposite of the count-cache
brittleness ladder.

**The scrambled control exposed a length-dependent mechanism — the most important
finding.** For `p ≥ 3` the scrambled control stays at ~0 while the repeated payload
locks: reproduction genuinely needs the **pattern**, consistent with induction. But for
`p = 1, 2` the scrambled control reproduces almost as well (0.97, 0.70), because the
control is **degenerate at short length**: a single repeated token has no order to
scramble (scramble == payload), and a 2-token string still re-creates its bigram by
chance under a shuffle. So:
- `p = 1` (and largely `p = 2`) reproduction is **frequency-driven** (the token simply
  becomes likely after many copies) — slower (`N*≈7`), and the control cannot separate it
  from copying. This **revises** the smoke test's "single token is hardest" reading: p=1
  is the frequency regime, not an induction one.
- `p ≥ 3` reproduction is **pattern/induction-driven** — faster (`N*≈2–3`) and the control
  confirms it.

**Caveats / limitations.** Only 4 payloads (bootstrap CIs are wide for p=2 and p=5);
GPT-2-small only; sampled `T=1.0` only (no greedy). The scrambled control is degenerate
for short `p` — E1.2 needs a **frequency-matched** control for short strings (e.g. a
different single token of matched unigram rate) to isolate identity from frequency, plus
longer/structured payloads where the control bites.

**This sets up E2 cleanly:** the induction claim (C2) should be tested at `p ≥ 3`, where
the scrambled control already says the mechanism is pattern-copying — activation patching
should pin reproduction there to induction heads, and should *not* for the
frequency-driven `p = 1` case.
