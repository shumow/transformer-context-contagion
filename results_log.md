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
