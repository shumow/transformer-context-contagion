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
frequency-driven `p = 1` case. **(Partly revised by E1.2 below: `p=1` is *also* mostly
copy, with only the largest frequency tail.)**

## E1.2 — copy vs. frequency (cue-dependence) on GPT-2 (2026-06-25)

E1.1's scrambled control was degenerate at short length, leaving open whether short-string
reproduction was pattern-copying (induction) or a frequency bias. E1.2 separates them
behaviorally (no activation access): after streaming `S × N` at above-knee `N ∈ {16,32}`,
prime generation with **S's own tail** (matching cue) vs **a fresh token `z ∉ S`**
(non-matching cue), and compare `P(reproduce S)`. Induction = cue-triggered; frequency =
cue-independent. 6 payloads × 12 trials.

| p | P(self) | P(neutral) | cue gap (N=16 / N=32) |
|---|---|---|---|
| 1 | 0.78–0.92 | **0.26–0.29** | +0.51 / +0.62 |
| 2 | 0.92–0.97 | 0.07–0.12 | +0.79 / +0.90 |
| 3 | 0.97–1.00 | 0.07–0.19 | +0.90 / +0.81 |
| 5 | 1.00 | 0.17–0.32 | +0.83 / +0.68 |
| 8 | 0.99–1.00 | 0.15–0.26 | +0.85 / +0.72 |

**Finding: reproduction is predominantly CUE-DEPENDENT at every length** — the
non-matching cue mostly fails (`P(neutral)≤~0.3`) while the matching cue locks
(`P(self)≈0.8–1.0`), a large gap (0.5–0.9) well above zero everywhere
(`results/e1_2_gpt2.png`). A behavioral **induction** signature (copy-on-match), without
TransformerLens.

**This revises E1.1.** Even `p=1` is mostly copy (gap +0.5–0.6; the matching cue far
outperforms the non-matching one), *not* the pure-frequency regime E1.1 suggested. What is
true is that `p=1` carries the **largest frequency tail** (`P(neutral)≈0.27` vs ~0.1–0.2
for longer strings): a single repeated token has the biggest cue-independent pull,
shrinking as more distinct tokens dilute any one token's frequency. The length axis is
*copy-dominant-everywhere, frequency-tail-largest-at-1*, not a clean regime switch.

**Caveats.** `P(neutral)` is **not pure frequency**: after the non-matching cue the model
can re-enter `S` if it emits an `S`-token by chance and induction then completes the
pattern (the toy's re-entry / self-healing), so `P(neutral)` *over*-states the frequency
component and the true copy fraction is if anything larger. ~72 samples/cell (Wilson
half-width ~0.11; neutral values noisy/non-monotonic, e.g. p=5 at N=32), GPT-2-small,
sampled `T=1.0` only.

**For E2:** updated prediction — induction should carry the self-cue reproduction at
**all** lengths (not only `p≥3`), with a frequency-bias contribution detectable mainly at
`p=1`. E2 (activation patching / head ablation) becomes the mechanistic confirmation of a
behaviorally-established induction signature.

## E1.5 — context-length confound on GPT-2 (2026-06-26)

E1.1's context is `S × N`, so number-of-repetitions and total context length grow
together. E1.5 disentangles them (4 payloads × 10 trials, lengths 1/3/5).

**Test 1 — is the knee an `N` effect or a length effect?** Compare *growing* context
(`S×N`, length grows) against *fixed* total length (filler padded to a constant length)
at each `N` (`results/e1_5_gpt2.png`, left).

| p | growing P (N=1..32) | fixed-length P (N=1..32) |
|---|---|---|
| 1 | 0, 0, .20, .57, .88, .97 | 0, 0, .15, .62, .78, .97 |
| 3 | 0, .65, .90, .97, .97, 1.0 | 0, .15, .75, .88, .90, .88 |
| 5 | 0, .72, .88, .97, 1.0, .97 | 0, .35, .85, .97, .93, 1.0 |

**The knee is predominantly an `N` effect:** at fixed total context length the threshold
still rises sharply with `N` (the knee persists), so E1.1's knee is not an artifact of the
context simply getting longer. There is a **secondary context-dilution effect**: at low
`N`, padding with filler modestly *lowers* reproduction (the payload is a smaller fraction
of context / sits further back), strongest for longer `p` (p=3 at N=2: .65 → .15). For
p=1 growing and fixed nearly coincide (least filler). So: knee = `N`-driven, with a mild
length modulation worth noting.

**Test 2 — distance / recency.** Hold `N=16` and insert a gap of `G` neutral tokens
between the payload and a re-presented matching cue `S[-1]`; sweep `G`
(`results/e1_5_gpt2.png`, right).

| p | P(reproduce) at G = 0, 8, 16, 32, 64, 128 |
|---|---|
| 1 | .97, .38, .47, .40, .05, .12 |
| 3 | .95, .23, .07, .00, .00, .03 |
| 5 | .88, .28, .07, .12, .03, .03 |

**Reproduction is distance-sensitive:** pushing the payload back from the generation point
collapses reproduction — gone by `G ≈ 16–32` tokens for `p ≥ 3`, more gradual for `p = 1`
(holds ~0.4 out to `G = 32`). So in GPT-2-small the copy is **recency-modulated, not
arbitrarily long-range**. Security reading: a payload *buried* deep in a long context
(far from where generation happens) is much weaker than one near the end.

**Caveats.** The gap is filled with rare tokens, so the decay could partly be
*filler-competition* (the model attends to the recent filler) rather than pure distance;
disentangling needs a position-matched control. Larger models have more / longer-range
induction heads, so the distance limit is expected to extend with scale — a prediction for
E1.3-lite / E2. 4 payloads × 10 trials, GPT-2-small, `T=1.0`.

## E1.3-lite — size sweep, and an 8 GB hardware wall (2026-06-26)

Intended a knee-vs-size sweep over GPT-2 {small, medium, large} and Pythia {160M, 410M}.
**Hardware finding (the dominant outcome): an 8 GB machine cannot run gpt2-medium (355M)
or larger.** gpt2-large (774M, 3 GB of weights) drove swap to 97% full and thrashed (47
min wall, ~9 min CPU, nothing completed); even gpt2-medium thrashed once residual swap
pressure built up. GPT-2-small (124M) and Pythia-160M (162M) run fine; everything above
needs a bigger machine. The real size-scaling question (E1.3 proper) is therefore
**deferred to more compute** — recorded as a constraint, not a result.

What *did* run (gpt2 + pythia-160m, 3 payloads × 8 trials):

| model | params | knee $N^*$ (p=1, 3, 5) |
|---|---|---|
| gpt2 | 124M | 6.2, 2.3, 1.6 |
| pythia-160m | 162M | none (<0.5 at N≤32), 3.2, 1.9 |

**Cross-family replication.** The condensation knee and its length-robustness
(`N*` decreasing with `p`) appear in **both** model families, not just GPT-2
(`results/e1_3.png`, left) — a (modest) generality check. The gpt2 knees here (6.2/2.3/1.6)
also replicate E1.1 (7.2/3.0/1.9) under a smaller payload/trial budget. Pythia-160m fails
to lock a single token within `N≤32` (knee undefined at p=1), i.e. it is *harder* to
frequency-drive into repeating one token than GPT-2 — consistent with E1.2's reading that
short-`p` reproduction has a model-dependent frequency component.

**Caveat.** The "knee vs parameters" panel has only two near-identical sizes (124M, 162M);
its apparent trend is **not** a scaling result. A genuine scaling sweep needs gpt2-medium+
on hardware with more than 8 GB.
