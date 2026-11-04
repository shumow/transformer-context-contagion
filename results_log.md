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

## E2 — induction-head attribution on GPT-2 (2026-06-26)

Mechanistic test of C2: do the actual induction heads carry the reproduction E1.2 showed
behaviorally? Run on **CPU** (TransformerLens warns MPS may be silently wrong on torch 2.8;
GPT-2-small on CPU is fast enough). Two steps.

**1. Identify induction heads.** The standard repeated-random-sequence induction score
flags the top heads as **L5H5, L7H10, L6H9, L5H1, L7H2** (then L9H6, L10H1, L9H9) — exactly
the canonical GPT-2-small induction heads from the literature, validating the detector.

**2. Causal ablation, swept over N.** Build `S × N`, measure `P(correct next payload
token)`; ablate the 8 induction heads vs 8 random heads. The clean signal needs the
**knee**, not saturation — at `N=16` the copy is over-determined (many copies + redundant
heads) so ablating a subset barely dents it; near the knee the circuit is load-bearing.

| | base | ablate-induction | ablate-random |
|---|---|---|---|
| p=3, N=2 | 0.40 | **0.05 (−87%)** | 0.30 (−25%) |
| p=3, N=3 | 0.70 | 0.26 (−63%) | 0.57 (−18%) |
| p=5, N=2 | 0.48 | **0.09 (−81%)** | 0.36 (−25%) |
| p=5, N=3 | 0.74 | 0.31 (−58%) | 0.63 (−15%) |
| p=1, N=4 | 0.31 | 0.10 (−67%) | 0.18 (−41%) |

**Finding: induction heads causally carry the reproduction.** Ablating the 8 induction
heads suppresses `P(correct token)` far more than ablating 8 random heads, at every length,
and the gap is **largest near the knee** (−81%/−87% at N=2 for p=5/3) and washes out at
saturation (`results/e2_gpt2.png`). This is the mechanistic confirmation of **C2** that
E1.2 set up behaviorally — reproduction is genuinely an induction-head phenomenon. For p=1
the induction ablation also bites (consistent with E1.2's "p=1 is mostly copy"), leaving a
residual that is the frequency tail.

**Caveats.** Random-head ablation is not exactly zero — removing any 8 heads causes ~15–25%
general degradation — so the load-bearing signal is the **differential** (induction drop ≫
random drop), which is large and consistent. 8 heads is a subset of the induction circuit;
ablating more would suppress further. At deep repetition (`N=16`) reproduction is robust to
losing 8 heads (redundancy). GPT-2-small only. Together with E1, this closes the core
behavioral-plus-mechanistic case that the condensation knee on real transformers is an
induction phenomenon.

## E6 — the context-window worm on GPT-2 / Pythia (2026-06-26)

Transmission test (Track A, C6): an infected host's output (containing the payload) becomes
the next host's context, passed as **text** (decoded then re-tokenized, as agents/RAG do);
count payload copies in each host's output across hops. Fresh payload per trial (8 trials).

**A. Same-model serial passage (GPT-2), mean load per hop** (`results/e6_worm_serial.png`):

| p | hop 0 → 6 | verdict |
|---|---|---|
| 1 | 42 → 18 → 12 → 12 → 12 → 12 → **12** | **sustains** (endemic load ~12; survive 25%) |
| 2 | 25 → 4 → 0 | dies (hop 2) |
| 3 | 25 → 15 → 11 → 8 → 6 → 2 → 0 | decays out (hop 6) |
| 5 | 12 → 5 → 5 → 3 → 1 → 0 | dies (hop 5) |
| 8 | 6 → 2 → 0 | dies fast (hop 2) |

**A real worm with a critical length.** Short strings (p=1) **propagate indefinitely** at an
endemic load; longer strings **die out**, faster as `p` grows — i.e. `R0` falls with string
length and crosses 1 around `p ≈ 1–2`. This confirms the toy's C6 prediction (short =
epidemic, long = one-shot) on a real transformer.

**Payload-dependence / super-spreaders.** Even at p=1, only ~25% of payloads sustain — the
worm is carried by **"sticky" payloads that fully take over generation** (occupancy → 1,
output ends mid-payload, so the next host re-enters). The high mean load is driven by those
super-spreaders; most payloads reproduce weakly and die. This ties to E1.5: a payload that
drifts to the front of a host's output is too far back (recency limit) for the next host to
re-trigger, so GPT-2's worm is weaker than the toy's induction surrogate (critical length
~1–2 vs the toy's ~2–3).

**B. Cross-tokenizer firebreak** (p=3, GPT-2 patient zero; `results/e6_worm_crosstok.png`):

| chain | load per hop |
|---|---|
| GPT-2 hosts (same tokenizer) | 15 → 9 → 8 → 6 → 4 → 4 → **2** (alive at hop 6) |
| Pythia-160M hosts (different tokenizer) | 21 → 4 → **0** (dead after one hop) |

**Confirmed: the tokenizer boundary is a firebreak.** Same-tokenizer transmission persists;
re-tokenizing the payload text under a different tokenizer fragments it (different token
boundaries) so induction can't copy it cleanly, and transmission collapses after the first
cross-tokenizer hop. A heterogeneous-model pipeline is naturally worm-resistant.

**Caveats.** 8 payloads/length, GPT-2-small + Pythia-160M, base models (no instructions),
`T=96` tokens/host, text-level payload counting (p=1 counts a short substring, so its
absolute load is less comparable). Establishes the *qualitative* epidemiology — critical
length and the tokenizer firebreak — not precise R0 values.

## E4-lite — poisonability tracks trust, not usefulness (2026-06-26)

The toy's headline claim (C4) on GPT-2-small, via a 2×2 of contexts —
{repetitive, non-repetitive} × {useful (real text), useless (OOD gibberish)} — each scored
on three axes: **U** = content naturalness (log-prob of one unit), **T** = induction-head
engagement (attention to a repeated bigram's continuation, using the E2 induction heads),
**P** = poisonability (copy-rate of a greedy continuation). On CPU via TransformerLens.

| context type | U (log-prob) | T (induction) | P (copy-rate) |
|---|---|---|---|
| rep + useful (real sentence ×k) | −3.5 | 0.34 | ~0.77 |
| rep + **useless** (gibberish ×k) | **−8.7** | 0.12 | **1.00** |
| nonrep + **useful** (real prose) | **−4.2** | 0.00 | **0.00** |
| nonrep + useless (random tokens) | −8.3 | 0.00 | ~0.22 |

**The dissociation holds: `corr(P, usefulness) = −0.20`, `corr(P, trust) = +0.54`.**
Poisonability follows the repetition / induction-trust axis, **not** the usefulness axis
(`results/e4_trust.png`). The headline contrast is stark: **repetitive gibberish (the
*least* useful content, `U≈−9`) is fully poisonable (`P=1.0`), while non-repetitive real
prose (useful, `U≈−4`) is not poisonable at all (`P=0.0`)**. Induction engagement (T) is
present for *both* repetitive types — including the useless one — and absent for both
non-repetitive types, exactly tracking P.

This is the real-transformer analogue of the toy's central finding (exploitability tracks
trust, not usefulness) and its defensive corollary: **the quantity to bound is trust on
repeated/templated content, independent of whether that content is useful.** Boilerplate,
duplicated RAG passages, and repeated instructions are poisonable *because they repeat*,
not because they help.

**Caveats.** Small N (16 contexts), GPT-2-small. "Usefulness" here is a *naturalness proxy*
(content log-prob), not a real downstream-task benefit — a base model has no task, so the
instruction-tuned version (Track B) is needed to test usefulness as actual task utility.
Greedy copy-rate has occasional outliers (one rep+useful at 0.075, one nonrep+useless at
0.875). The qualitative dissociation (P ⊥ usefulness, P ∥ repetition/induction) is clear.

## E5-lite — delivery: hijacking generation from inside a document (2026-06-26)

Scaled-down RAG attack (C5): a coherent prose "document" (~65 tokens) with an OOD payload
(p=3) repeated k times planted in it; generate and measure **hijack** = payload occupancy
of the continuation. 3 docs × 4 payloads/trials. GPT-2-small.

**A. Placement / entry (k=8 fixed), hijack vs where the payload sits:**

| placement (0=buried start, 1=at end) | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| hijack | 0.003 | 0.001 | 0.003 | 0.001 | **0.523** |

**Entry is sharply recency-gated.** The payload hijacks *only* when it is at the very end
(frac=1.0 → 0.52); anywhere earlier it is ~0 — even frac=0.75, which leaves just ~16 tokens
of prose between the payload and the generation point, already kills it. So a payload
**buried** in a document is defanged; it must sit within ~the last dozen-or-so tokens. This
is E1.5's distance limit, and tighter, because *coherent* prose competes harder for the
continuation than random filler.

**B. Repetition / lock-in (payload at the end), hijack vs k:**

| k | 0 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| hijack | 0.001 | 0.005 | 0.027 | 0.117 | 0.354 | **0.696** |

A clean knee inside a real document, crossing ~0.5 around **k ≈ 10–12** — notably *higher*
than the clean-context knee (~3 for p=3 in E1.1). **Embedding in legitimate prose raises the
poison cost**: the document provides resistance, so more repetitions are needed to lock in.

**C5 confirmed: hijack = entry × lock-in.** Both must be satisfied — the payload must be
*recent* (near the generation point) **and** repeated enough to overcome the legit content.
Security reading: in a RAG/long-context setting the danger zone is content near where
generation happens; burying a payload deep, or having strong legitimate context, both
suppress the hijack.

**Caveats.** ~65-token "documents" (GPT-2-small, scaled down), p=3, occupancy metric, 12
samples/cell. The realistic long-context (4k–128k) RAG version with retrieval is Track B.

## E7 — the paired detector (defensive complement) (2026-06-26)

> **Status (2026-11-03): SUPERSEDED by the rerun below** ("E7 rerun — detector vs n-gram
> baseline, 32 negatives"). This run had 3 clean documents (an AUC over 3 negatives is not a
> measurement), no model-free baseline, and no false-positive rate on legitimately repetitive
> text. The rerun scores 32 clean paragraphs against an n-gram-repeat baseline and 8
> benign-repetitive negatives with bootstrap CIs, and overturns the headline: the induction
> detector does **not** beat a model-free n-gram-repeat count (both AUC 1.0 at every k), and is
> marginally worse on benign-repetitive false positives. The "AUC=1.0" and "fires at k=2" claims
> below are real but say nothing the n-gram baseline does not; read the rerun entry instead.

The attacks (E4/E5/E6) all key on one quantity: induction trust earned by a repeated span.
So the defense watches that same quantity. The detector scores a context by its mean
**induction engagement** (attention the canonical GPT-2 induction heads pay to
repeated-bigram continuations, `tcc.interp.repeat_induction_profile`) — high on a repeated
/ poisoned span, ~0 on natural prose. Binary test: clean prose docs vs. the same docs with
an OOD payload (p=3) repeated k times planted at the end.

| | clean | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| mean detector score | ~0.001 | 0.015 | 0.038 | 0.053 | 0.057 |
| AUC vs clean | — | **1.00** | **1.00** | **1.00** | **1.00** |

**Perfect separation (AUC=1.0) at every k**, with the score rising monotonically with k
(`results/e7_detector.png`). Natural prose has essentially zero repeated-bigram induction
engagement, so any planted repeated span stands out — and the detector fires already at
**k=2, well below the in-document hijack knee (~10–12, E5)**: it flags the poisonable
configuration *before* it is dangerous. Detection sensitivity grows exactly as the context
becomes more poisonable, and it operates on the **trust** signal, independent of usefulness
(it would flag the E4 gibberish-poison and a coherent-looking repeated poison alike).

**Defensive reading.** Flagging — or capping the trust earned by — repeated spans that
approach the condensation knee is precisely what neutralizes the E5 hijack and the E6 worm,
and it is the transformer realization of the toy's "bound trust, not usefulness" corollary.

**Honest caveat.** The detector flags *any* induction-engaging repeated span, malicious or
benign — repeated boilerplate, templates, repeated instructions, code. That is the
*correct* behavior under the thesis (such spans are genuinely poisonable, E4), but it means
the detector identifies the poisonable *configuration*, not malicious *intent*; a deployed
version needs a benign-repetition allowlist or a trust cap rather than a hard block. ~65-token
docs, GPT-2-small, p=3.

## E1 refinement — greedy vs. sampled decoding (2026-06-26)

Robustness check that the knee is not a sampling artifact. Re-ran E1 under **greedy**
(deterministic) decoding, 6 payloads:

| p | knee N* greedy | knee N* sampled (E1.1, T=1.0) |
|---|---|---|
| 1 | 4.0 | 7.2 |
| 3 | 1.5 | 3.0 |
| 5 | 1.5 | 1.9 |

The knee **persists under greedy decoding** and length-robustness holds (N* decreasing with
p). Greedy knees are *lower* than sampled — with no sampling noise to break reproduction the
payload locks in with fewer repetitions. So the condensation knee is a property of the model,
not of stochastic decoding. (A full temperature sweep, 0 → 1.0+, remains future work.)

## E8-lite — in-context substitution as an induction primitive, and the cascade limit (2026-06-26)

> **Status (2026-11-03): SUPERSEDED by the rerun below** ("E8-lite rerun — true random
> control"). The "ablate random" column below was in fact the 8 *lowest*-induction-score heads,
> a control biased toward unimportant heads. The rerun adds a genuinely random matched control
> (seeded, excluding the induction set, as in E2) alongside the lowest-induction set. Outcome:
> the induction-carried claim **survives** at the reliable table sizes (m=8: −57% vs a true
> random −33%; m=16: −68% vs −3%), but is **weaker** than this entry implies — a real random
> ablation removes ~33% at m=8, not the −6% the mislabeled column showed, so the "−6% random"
> numbers below overstate the induction-specific effect. Read the rerun entry instead.

Mechanistic core of the "self-extracting payload" idea (random tokens that unfold into a
payload via per-layer substitution), on GPT-2-small with benign OOD markers — the "decoded"
tokens are just other random markers, no coherent or harmful payload. A substitution table
is key/value pairs; induction is "saw [key][value]…see [key]→predict [value]." Three measurements.

**1. Single-step substitution is real and induction-carried.** Present a random OOD table
(m pairs), query a key, measure top-1 accuracy of the value:

| m | clean | ablate induction | ablate random |
|---|---|---|---|
| 2 | 0.31 | 0.19 | 0.38 |
| 4 | 0.50 | 0.34 | 0.41 |
| 8 | 0.47 | **0.20 (−57%)** | 0.44 (−6%) |
| 16 | 0.47 | **0.15 (−68%)** | 0.31 |

Ablating the (E2) induction heads collapses substitution far more than ablating the same
number of random heads, sharply so at larger tables — **in-context substitution is an
induction primitive.** But the absolute fidelity is *modest* (~0.5 for a single table):
GPT-2-small does novel substitution only about half the time.

**2. The binding constraint is cross-layer interference, not per-step fidelity.** With more
in-context demonstration (two tables present), the *single* lookup jumps to **0.97** — so
per-step fidelity is context-dependent and can be high. Yet the **2-step cascade succeeds
only 0.19** (table1 `k→v`, table2 `v→w`, query `k`, decode two layers) — *below* the
optimistic `f²=0.25`. The reason: the intermediate value `v` appears both as a value (table1)
and a key (table2), so the induction lookup is **split** between the two — the cascade's own
intermediate products create induction ambiguity. So self-extraction depth is bounded *worse*
than the `f^(L·depth)` brittleness ladder predicts.

**Answer to "is a self-extracting payload possible?"** Mechanistically yes — induction is
exactly the substitution primitive — but on GPT-2-small it is **sharply depth-limited**: even
two layers lose most of the signal, and the limiter is cross-layer interference, not raw
fidelity. A deep self-extractor would need a model that both substitutes more reliably *and*
disambiguates which table to use (likely positional/recency, stronger at scale). That extends
the depth limit but is a Track-B (capability) question; the *coherent*-payload and
self-execution capstone is also Track-B. **Defensive note:** the cascade self-limits — an
attacker's own staging fights them — and the staging structure (an in-context lookup table) is
itself a conspicuous, induction-engaging pattern the E7 detector keys on.

**Caveats.** GPT-2-small, 8 tables × m queries/cell (small-m rows noisy), top-1 metric,
greedy 2-step. Characterizes the primitive and the interference obstacle, not a working extractor.

## E1.3 — the size sweep at scale (Track B, Azure T4) (2026-06-30)

E1.3-lite was blocked on the 8 GB dev machine (gpt2-medium+ thrashed swap). Re-run properly
on an Azure NC4as_T4_v3 (Tesla T4, 16 GB), fp16, across **12 base models in four families,
355M → 7B**: GPT-2 {medium, large, xl}, Pythia {410m, 1.4b, 2.8b}, Qwen2.5 {0.5B, 1.5B, 3B,
7B}, Llama-3.2 {1B, 3B}. Main condition only (controls validated on GPT-2-small); 3 payloads
× 8 trials, lengths p ∈ {1,3,5,8}, reps N ∈ {1,2,4,8,16,32}.

| model | params | N*(p=1) | p=3 | p=5 | p=8 |
|---|---|---|---|---|---|
| gpt2-medium | 355M | 5.1 | 2.7 | 1.8 | 1.7 |
| gpt2-large | 774M | 6.3 | 1.9 | 1.6 | 1.6 |
| gpt2-xl | 1.5B | 9.3 | 1.9 | 1.6 | 1.7 |
| pythia-410m | 410M | – | 2.4 | 1.7 | 1.9 |
| pythia-1.4b | 1.4B | – | 2.3 | 1.6 | 2.3 |
| pythia-2.8b | 2.8B | – | 2.3 | 1.5 | 1.8 |
| Qwen2.5-0.5B | 0.5B | – | 1.8 | 1.5 | 1.7 |
| Qwen2.5-1.5B | 1.5B | – | 1.8 | 1.9 | 1.6 |
| Qwen2.5-3B | 3B | – | 1.8 | 1.7 | 1.7 |
| Qwen2.5-7B | 7B | – | 1.5 | 1.6 | 1.6 |
| Llama-3.2-1B | 1B | – | 1.9 | 2.7 | 1.9 |
| Llama-3.2-3B | 3B | – | 1.7 | 1.9 | 1.6 |

(– = no knee: P(reproduce) stays < 0.5 for N ≤ 32.)

**C3 holds across scale and family.** For payloads of length p ≥ 3 the knee is **low and
roughly flat — N\* ≈ 1.5–2.7 everywhere**, falling with p. A novel span of ≥ 3 tokens locks
in after ~2 presentations at every size up to 7B and in all four families. Induction-style
length-robustness is not a small-model artifact.

**The knee does not fall with scale (for p ≥ 3).** Contrary to the plan's loose prediction
that bigger = lower/sharper, N\*(p ≥ 3) is essentially size-independent — the copy mechanism
is already saturated by 355M. Any scale effect lives in *single-token* lock-in, not
multi-token.

**Single-token lock-in is a GPT-2-family quirk, and it gets *harder* with scale.** Only the
GPT-2 family locks a single repeated token within N ≤ 32, and its knee *rises* with size
(5.1 → 6.3 → 9.3 for medium → large → xl). Pythia, Qwen2.5, and Llama-3.2 never lock p=1 by
N ≤ 32. This sharpens E1.2's "short-p has a model-dependent frequency component": p=1
reproduction is about token *frequency priors*, not induction — family-specific and
anti-correlated with GPT-2 scale — while the induction-carried p ≥ 3 knee is universal.

**Setup notes.** fp16 load (a `--dtype` flag added to the sweep) keeps 7B at ~14 GB on the
16 GB T4. Gated models (Llama-3.2) load from cache under `HF_HUB_OFFLINE=1` — the experiment's
`from_pretrained` otherwise does an online gating check that 401s without a token. ~28 min
wall for the 10 ungated models. Results: `results/e1_3_trackb_t4.{json,png}`; Llama in
`results/e1_3_llama*`. Instruction-tuned models (E1.4) follow; the mechanistic/24 GB work
(E2 at scale) is deferred to the A10.

## E1.4 — does the knee survive instruction-tuning? (Track B, T4) (2026-06-30)

Same knee measurement as E1.3 (fp16, T4, lengths p ∈ {1,3,5,8}, reps N ∈ {1,2,4,8,16,32},
3 payloads × 8 trials) on four **instruction-tuned / chat** models across three families:
Qwen2.5-{1.5B,7B}-Instruct, Llama-3.2-3B-Instruct, gemma-2-2b-it. Gated weights loaded from
cache under `HF_HUB_OFFLINE=1`. The plan's worry: chat models might refuse, summarize, or
comment rather than copy. They did **not** — all four ran clean and reproduced the span.

| model | params | N*(p=1) | p=3 | p=5 | p=8 | base counterpart (p=3/5/8) |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.5B | – | 2.8 | 2.8 | 2.0 | 1.8 / 1.9 / 1.6 |
| Qwen2.5-7B-Instruct | 7B | – | 1.6 | 2.5 | 1.8 | 1.5 / 1.6 / 1.6 |
| Llama-3.2-3B-Instruct | 3B | – | 3.2 | 6.3 | 2.0 | 1.7 / 1.9 / 1.6 |
| gemma-2-2b-it | 2B | **1.0** | 3.0 | 1.8 | 1.4 | (no base run) |

(– = no knee at N ≤ 32.)

**Instruction-tuning does not immunize against context contagion.** The condensation knee
persists in every chat model, in the same low regime as base models — a planted repeated
span of length p ≥ 3 still gets reproduced after ~2–3 presentations. C1/C3 extend to the
"more realistic" chat condition.

**The effect is a modest *raising* of the threshold, not removal.** Instruct knees sit
slightly above their base counterparts (e.g. Qwen2.5-7B 1.5/1.6/1.6 → 1.6/2.5/1.8). The
exception is **Llama-3.2-3B-Instruct**, clearly the most resistant (p=5 knee 6.3 vs base 1.9)
— instruction-tuning bought it the largest reps-to-reproduce penalty, but did not abolish the
knee.

**gemma-2-2b-it is the most eager copier** — the only model in either sweep to lock a
*single* repeated token (p=1 N\* = 1.0). Single-token lock-in thus is not exclusively a GPT-2
quirk (E1.3): a small instruct model also does it, reinforcing that p=1 reproduction tracks
model-specific token priors rather than the universal induction-carried p ≥ 3 knee.

**Setup/caveats.** Raw-context format (the planted span as bare tokens, not wrapped in a chat
template); a chat-templated condition is a natural follow-up and may shift thresholds.
Results: `results/e1_4.json` (with bootstrap CIs) on the VM's persistent disk — pulled next
session; knees here are from the run log. T4 idle-deallocated itself after the run (cost
safety worked).

## E2 at scale — induction-head causality past GPT-2-small (Track B, T4) (2026-07-01)

Re-ran the E2 causal test (ablate top-k induction heads vs. k random heads, measure
`P(correct next payload token)` across p ∈ {1,3,5}, N ∈ {2,3,4,8,16}, 4 payloads) on the
larger models, on the T4 (fp32, `--k-heads 8`).

| model | heads (L×H) | induction-ablation effect near knee | verdict |
|---|---|---|---|
| gpt2 (124M, prior) | 12×12=144 | −81% to −87% (p=3, low N) | C2 confirmed |
| **Pythia-1.4B** | 24×16=384 | **p=3: −80% @N=2, −58% @N=3, −54% @N=4** (random ~0%) | **C2 confirmed at scale** |
| GPT-2-xl (1.5B) | 48×25=**1200** | only −4% to −13% (random ~0%) | *inconclusive — see below* |
| Pythia-2.8B, Qwen2.5-3B | — | OOM (fp32) on 16 GB T4 | needs ≥24–48 GB |
| Llama-3.2-3B | — | gated 401 (missing `HF_HUB_OFFLINE=1`) | rerun offline |

**C2 extends past GPT-2-small: Pythia-1.4B confirms it cleanly** — ablating its 8 induction
heads collapses reproduction near the knee (−80% at p=3, N=2) while random-head ablation is
~0%, exactly the GPT-2-small pattern. Induction-head causality is not a small-model artifact.

**The GPT-2-xl null is a methods artifact, not a refutation — and it teaches the fix.** A
*fixed* 8 heads is load-bearing in GPT-2-small (144 heads) but a vanishing **8/1200 = 0.7%**
of GPT-2-xl's heads, so ablating them barely moves reproduction. The induction circuit is
spread over far more heads at scale. **Fix implemented:** added `--k-frac` to
`e2_induction.py` — ablate a *fraction* of all heads (scaled to the model) instead of a fixed
count, so the ablation set is comparable across sizes. The 7B run (on a ≥24–48 GB GPU) should
use `--k-frac` (e.g. 0.05–0.10), not `--k-heads 8`.

**Ops.** 2.8B/3B OOM at fp32 on the 16 GB T4 (interp loads fp32; no `--dtype` there yet) —
these move to the RunPod 48 GB box (A6000/L40S) alongside 7B. E2 now checkpoints per (p,N)
cell and resumes, so it is Spot-eviction safe. Results: `results/e2_gpt2-xl.json`,
`results/e2_EleutherAI_pythia-1.4b.json`.

## E2 at scale — the full ladder (RunPod L40S 48 GB, `--k-frac`) (2026-07-02)

Completed E2 on a RunPod L40S (48 GB), resolving the gpt2-xl question above and adding the
larger Pythia models that OOM'd on the T4. The ablation set is now scaled by **`--k-frac 0.05`**
(≈5% of all heads) instead of a fixed 8. Induction-head ablation vs. a matched random-head
control, `P(correct next payload token)`, p ∈ {1,3,5}, N ∈ {2,3,4,8,16}, 4 payloads.

| model | params | ablated (k / total) | p=3 N=2 induction | random | p=3 N=3 induction | random |
|---|---|---|---|---|---|---|
| gpt2 | 124M | 8 / 144 (5.6%) | −87% | −25% | −63% | −18% |
| gpt2-xl | 1.5B | 60 / 1200 (5%) | **−70%** | −6% | **−74%** | −2% |
| pythia-1.4b | 1.4B | 19 / 384 (5%) | −88% | +18%† | −79% | +6% |
| pythia-2.8b | 2.8B | 51 / 1024 (5%) | **−98%** | −28%† | **−87%** | +1% |
| pythia-6.9b | 6.9B | 51 / 1024 (5%) | −82% | +25%† | −66% | +13% |
| Qwen2.5-7B | 7B | 39 / 784 (5%) | −64% | −27%† | −39% | −9% |

(† random-ablation noise at N=2 with only 4 payloads; the induction effect dwarfs it and the
control is clean by N=3.) Qwen2.5-7B is more modest at p=3 but the **strongest of all at p=5**
(N=2: base 0.940 → 0.039, **−96%** induction vs −24% random; N=3 −78% vs −4%).

**C2 (induction-head causality) holds across scale and three families — 124M → 7B, GPT-2,
Pythia, and Qwen2.5.** Ablating the top ~5% of heads by induction score collapses reproduction
near the knee (−64% to −98% at p≥3–5, low N); ablating the same number of *random* heads barely
moves it. The copy mechanism is the induction heads, at every scale we can reach.

**The `--k-frac` fix was essential — the earlier gpt2-xl "null" was a measurement artifact.**
With the old fixed `--k-heads 8` (0.7% of gpt2-xl's 1200 heads) the effect was −4% to −13% (a
false negative in the prior entry); with `--k-frac 0.05` it jumps to −70% to −74%. A fixed head
count doesn't scale — the induction circuit spreads over more heads in larger models, so the
ablation budget must scale with the model. **Methodology takeaway for any scaling study: ablate a
fraction of heads, not a fixed count.**

**Single-token (p=1) reproduction is the frequency tail, not induction — and only GPT-2 shows it.**
Pythia *and* Qwen2.5 never lock p=1 (base ≈ 0.000 at every scale); GPT-2 does, but that is its
weakest-induction / largest-frequency-tail regime (E1.2). Consistent with E1.3's p=1 reading.

**A scale/family difference at saturation.** For GPT-2 the induction dependence fades once the
copy is over-determined (gpt2-xl p=5, N=16: only −6%). Pythia *keeps* a strong dependence even at
high N (pythia-2.8b p=5 N=16: **−82%**; pythia-6.9b: **−54%**) — the larger Pythia models route
repeated-span copying through induction heads even when saturated, rather than developing
redundant non-induction copy paths. Worth a follow-up.

**Setup/caveats.** RunPod L40S 48 GB, fp16, `transformer_lens` HookedTransformer. **Qwen2.5-7B
completed on a retry** (its HF weight download stalled once on the flaky host; TransformerLens
does support the arch) — a genuine third family at 7B.
`--k-frac` is uniform at 0.05 across the ladder (Pythia-1.4B re-run at 0.05 = 19/384; the
effect is if anything stronger, −88% at p=3 N=2, than at the old 8/384); gpt2 is the one
exception at 8/144 ≈ 5.6%.
4 payloads, single seed, greedy — characterizes the causal effect; CIs modest. Results:
`results/e2_{gpt2-xl,EleutherAI_pythia-2.8b,EleutherAI_pythia-6.9b}.json`.

## E8 capstone — coherent self-extraction and self-execution on instruct models (Track B, RunPod L40S) (2026-07-02)

E8-lite found in-context substitution is induction-carried but sharply depth-limited on
GPT-2-small (2-step cascade 0.19, below f²). The capstone tests the two pieces the toy deferred,
on capable **instruct** models (chat-templated, fp16). **Strictly benign** (see SECURITY.md):
the decoded payload is a harmless English sentence; the executed instruction is a harmless
marker task ("reply with ACORN"); no harmful content anywhere. **E8a:** nested substitution
tables + an encoded message, decode to English, fidelity (LCS) vs nesting depth. **E8b:** "decode
this benign instruction and do what it says," self-execution rate. `surface-leak` = whether the
marker is visible in the *encoded* form.

| model | params | E8a unfold (d=1/2/3) | E8b self-exec (d=1/2/3) | surface-leak |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 7B | 1.00 / 0.90 / 0.94 | **1.00 / 1.00 / 1.00** | 0.00 |
| Llama-3.2-3B-Instruct | 3B | 0.91 / 0.82 / 0.69 | 1.00 / **0.00 / 0.00** | 0.00 |
| gemma-2-2b-it | 2B | 0.87 / 0.95 / 0.87 | 1.00 / 0.88 / 0.50 | 0.00 |

**The capability dependency is confirmed — coherent self-extraction needs a capable model, and
capable models have it.** All three instruct models blow past the toy's 2-step 0.19 limit on
pure unfolding (E8a 0.69–1.00 vs 0.19): a capable model decodes even 3-layer nested substitution
into coherent English. What the induction-only toy couldn't do, an instruct model does easily.

**Self-execution is the harder, capability-gated step — and it dissociates from decoding.** E8b
(decode *and act*) is strictly harder than E8a (decode). Qwen-7B self-executes perfectly through
depth 3 (1.00); gemma-2B degrades gracefully (1.00 → 0.88 → 0.50); Llama-3B falls off a **cliff**
— 1.00 at depth 1, then **0.00 at depth ≥2** despite still *decoding* at 0.82 (E8a). So a model
can produce the decoded text when asked to decode, yet fail to decode-**and**-follow at the same
depth: "act on your own in-context decoding" is a compositional step more demanding than either
decode or execute alone, and only the 7B does it robustly with depth.

**The evasion surface is clean, and demonstrable benignly: surface-leak = 0.00 everywhere.** The
marker/instruction is never present in the encoded form, at any depth or model — a surface
content filter sees only gibberish; the payload materializes only after the model unfolds it.
That is the threat the capstone characterizes (with harmless markers); its mitigation is
**structural, not content-based** — the substitution-table + "decode and follow" scaffold is a
conspicuous, induction-engaging repeated-pair block, exactly what the E7 detector / bound-trust
flags regardless of the (hidden) payload.

**This closes Track B.** E1.3 (size sweep 355M–7B), E1.4 (instruct), E2 (mechanism to 7B, three
families), and E8 (capstone) are all done. Setup: RunPod L40S 48 GB, fp16, chat templates;
4 targets/instructions × 3 depths × 2 reps, greedy. Results: `results/e8_capstone.{json,png}`.
Caveats: small n (fidelity + exec-rate are point estimates), single seed, benign markers by
design; the depth-cliff locations may shift with more reps.

## E4-scale — trust vs. usefulness with a REAL task (Track C, RunPod L40S) (2026-07-02)

> **Status (2026-11-03): design flaw; superseded pending rerun.** Both "non-repetitive"
> contexts in this run were the *same* filler paragraph twice (`FILLER + FILLER`), so the 2×2
> contrasted 8× repetition with 2× repetition. And P was copy-rate of a raw continuation of
> "span × 8", which is ~1 for any repeated span by construction. `experiments/e4_scale.py`
> now builds non-repetitive cells from distinct sentences, length-matched in tokens; adds the
> plan's original poisonability measure (reproduction of a *planted* probe span); uses 32
> facts; and reports bootstrap CIs on the correlations (`review_remediation_plan.md` §0.2).
> **Rerun 2026-11-04: see "E4-scale v2" below** — the dissociation holds on three instruct models
> with the probe measure, with a smaller effect than the copy-rate numbers here suggested.

E4-lite showed poisonability P ⊥ usefulness U, ∥ trust T on GPT-2-small, but U was a
*naturalness proxy*. E4-scale replaces it with a genuine task on **Qwen2.5-7B-Instruct**: benign
**synthetic facts the model cannot already know** (so a context containing the fact is the
*only* source of the answer — clean, confound-free usefulness), a 2×2 {rep, non-rep} × {useful,
useless}, scored on U (chat-QA accuracy), P (copy-rate of a raw continuation), T (induction
attention). 6 facts, k=8.

| context | U (task) | P (poison) | T (trust) |
|---|---|---|---|
| repetitive + useful (fact ×k) | 0.83 | **1.00** | 0.073 |
| repetitive + **useless** (gibberish ×k) | **0.00** | **1.00** | 0.153 |
| non-repetitive + **useful** (fact once) | **1.00** | 0.38 | 0.058 |
| non-repetitive + useless (prose) | 0.00 | 0.10 | 0.034 |

**corr(P, U) = +0.09 (≈0)   corr(P, T) = +0.71.**

**C4 confirmed at scale, with a real task.** Poisonability follows the *repetition/trust* axis,
not *usefulness*: **repetitive gibberish (U=0) is fully poisonable (P=1.00)** while the
**non-repetitive real fact (U=1.0) is barely poisonable (P=0.38)**. Even when usefulness is a
genuine downstream benefit on a capable instruct model, exploitability is orthogonal to it and
aligned with induction trust — the parent paper's conservation law holds at scale. The OOD
gibberish even draws *more* induction attention than the repeated real fact (T=0.153 vs 0.073):
trust is about repetition/surprise, not meaning. Caveats: 6 synthetic facts, one model, greedy;
rep_useful U=0.83 (repetition slightly hurt QA on one fact). `results/e4_scale.json`.

## E5-scale — RAG hijack in a long context: a bound on C5 (Track C) (2026-07-02)

> **Status (2026-11-04): confirmed with positive controls — see "E5-scale v2" below.** This run
> had no no-query control, so a harness bug and a real null were indistinguishable; v2 adds two
> controls (raw continuation hijacks at 0.6–1.0 occupancy), three query styles and two more models.

E5-lite (GPT-2-small, base, no query) showed a recent, repeated payload hijacks the
*continuation*. E5-scale plants an OOD payload ×k inside a ~4000-token benign "retrieval dump"
on **Qwen2.5-7B-Instruct** and asks it to answer a query; hijack = payload occupancy of the
answer. Sweeps: placement (front→end, k=16) and repetition k (at the end).

**Result: no hijack — occupancy = 0.00 at every placement, and 0.00 for k up to 32.**

**This bounds C5: the raw-copy hijack does not transfer to instruct-RAG answering.** The *same*
model reproduces a planted span in raw continuation (E1.4 — the knee survives instruction-
tuning), yet here, asked to answer a question from the documents, it stays on task and never
emits the payload, wherever it sits and however many times it repeats. **The task query anchors
generation away from the payload** — instruction-following overrides the copy tendency.
Defensively, a well-posed task query is itself a mitigation against pure-repetition hijack; the
residual RAG threat is *semantic* (a plausible injected instruction — indirect prompt
injection), which is outside this project's benign-OOD copy-dynamics scope. So E5-lite's hijack
is a base-model-continuation phenomenon; a capable instruct model doing a real task is not
hijacked by raw repetition. (Pre-registered as publishable either way — this refutes the naive
transfer and bounds where the copy-dynamics story applies.) Caveats: one model, one query style,
4k context, benign OOD payload; a no-query "continue the documents" framing would likely hijack
(that is E1.4/E5-lite) — the finding is that the *task query* defuses it. `results/e5_scale.json`.

## Analyses of committed data, following the critical review (2026-11-03)

Three re-analyses of data already in the repository, no new model runs. Scripts:
`experiments/analysis_knee_floor.py`, `analysis_e6_firebreak.py`, `analysis_e6_r0.py`; outputs
under `data/*/analysis_*.json`. Where the raw JSON was never committed, the inputs are
transcriptions of the tables in this log (`data/trackA/e6_worm_transcribed.json`,
`data/trackB/e1_3_e1_4_knees_transcribed.json`, `data/trackC/e6_scale_partB_3hop_reps16.json`),
each with a `_provenance` field saying so.

### A. The "knee" for multi-token payloads is the grid floor

For p ≥ 2 the transition S[-1]→S[0] has never occurred in a context of N=1 copies, so copying
it at N=1 is impossible except by chance; at N=2 it has occurred once. Any copy mechanism
therefore jumps between N=1 and N=2, and the interpolated N* lands in (1, 2] whenever the
**one-shot copy rate** P(reproduce | N=2) ≥ ½. In that regime N* is a re-parameterization of
that rate (N* = 1 + ½ / P(N=2) when P(N=1)=0), not an independently measured threshold.

Applied to the E1.3/E1.4 knee tables (48 multi-token cells, 16 models, p ∈ {3,5,8}):

| | cells | at the floor (N* ≤ 2) | implied one-shot rate P(N=2), floor cells |
|---|---|---|---|
| p ≥ 3, base + instruct | 48 | **36 (75%)** | median 0.71, min 0.50 |

The 12 cells not at the floor: gpt2-medium p=3 (2.7); pythia-410m p=3 (2.4); pythia-1.4b p=3,
p=8 (2.3, 2.3); pythia-2.8b p=3 (2.3); Llama-3.2-1B p=5 (2.7); Qwen2.5-1.5B-Instruct p=3, p=5
(2.8, 2.8); Qwen2.5-7B-Instruct p=5 (2.5); Llama-3.2-3B-Instruct p=3, p=5 (3.2, 6.3);
gemma-2-2b-it p=3 (3.0). One cell (gemma-2-2b-it p=8, N*=1.4) inverts to P(N=2) > 1, meaning
P(N=1) was not zero there; the inversion is marked invalid.

**What this changes.** "A sharp knee at N*≈1.5–1.9 that is flat from 355M to 7B" is the
statement "one prior occurrence of a ≥3-token span suffices more often than not in every
model" — which is what an induction head is defined to do. The sharpness and the scale-flatness
are properties of the grid, not findings. The defensible summary is the one-shot rate (≈0.5–1.0
across models, from the inversion; a direct measurement needs the per-N curves, which E1.3 did
not store). The cells *above* the floor are the informative ones: three of the four instruct
models and the smaller Pythia models need more than one prior occurrence at p=3, which is a
real, if modest, difference. The p=1 and p=2 knees are not floor effects and remain as logged
(frequency-prior / copy-suppression regime). `e1_3_size_sweep.py` should store the per-N
curves on its next run so P(N=2) can be reported directly with a Wilson interval.

### B. The cross-tokenizer firebreak: both 2026-07-02 runs, with a test

Two Part-B runs are in `data/trackC/logs/e6.log`; only the second was reported above. Exact
permutation test, null = host tokenizer identity is irrelevant (columns exchangeable within a
row), statistic = diagonal mean − off-diagonal mean, 5⁵ = 3125 assignments:

| run | hops | seed copies | trials | diag mean | off-diag mean | ratio | diag is row-max | diag = 0 | one-sided p |
|---|---|---|---|---|---|---|---|---|---|
| first (script defaults, **unreported**) | 3 | 16 | 5 | 4.00 | 2.44 | 1.64 | 2 / 5 | pythia, llama | **0.238** |
| second (reported, `e6_scale.json`) | 1 | 32 | 8 | 14.18 | 8.82 | 1.61 | 3 / 5 | none | **0.004** |

Off-diagonal entries exceeding their row's diagonal: first run, six of them (pythia→qwen 13.6
vs pythia→pythia 0.0; llama→pythia 6.4 vs llama→llama 0.0; gemma→gpt2 and gemma→pythia 6.4 vs
gemma→gemma 0.8); second run, two (pythia→qwen 15.5, qwen→llama 16.9).

**What this changes.** At 3 hops on 0.5–2B hosts with 16 seed copies, there is no detectable
same-tokenizer advantage: the diagonal-vs-off-diagonal gap is what a random assignment of
"diagonal" produces about one time in four. The reported "partial firebreak" comes from a
re-run at 1 hop with double the seeding, where the advantage is significant (p = 0.004, ratio
1.6). One hop is the weakest possible transmission test (patient zero's output re-tokenized
once). The protocol change between the runs was not recorded in this log or the paper; it is
now. The claim should read: *under heavy seeding, a single re-tokenization attenuates
transmission by roughly 1.6× on average; over three hops on small models, transmission is too
weak in both conditions to measure a firebreak.* The queued `deploy/run_trackC_p1.sh` sweeps
seeding strength at 3 hops with 30 trials and should settle which description holds.

### C. R0 was never computed; here is what the mean trajectories support

Per (model, p): R_geo = geometric mean of k_{t+1}/k_t over hops where both loads ≥ 1 (the
per-hop multiplication while alive); R_tail = the ratio at the last hop with k_t ≥ 1 (the step
nearest threshold; 0 for chains that die). Computed from the **mean** load per hop, since no
per-trial trajectories were stored (both E6 scripts now store them under `runs`).

| model | p=1 | p=2 | p=3 | p=5 | p=8 |
|---|---|---|---|---|---|
| GPT-2 (Track A) | 0.81 / endemic at 12 | 0.16 / dies hop 2 | 0.60 / dies hop 6 | 0.54 / dies hop 5 | 0.33 / dies hop 2 |
| Pythia-2.8B | – / dead | 0.60 / dies hop 5 | 0.50 / dies hop 3 | 0.66 / dies hop 5 | 0.66 / dies hop 4 |
| Qwen2.5-7B | – / dies hop 1 | 0.90 / endemic | 0.91 / endemic | 0.93 / endemic | 0.86 / endemic |
| Llama-3.2-3B | – / dead | 0.60 / dies hop 3 | – / dies hop 1 | 0.71 / dies hop 5 | 0.27 / dies hop 2 |
| Qwen2.5-7B-Instruct | 0.20 / dies hop 2 | 0.89 / endemic | 0.70 / endemic (load 2 at hop 6) | 0.77 / endemic | 0.89 / endemic |

(cell = R_geo / verdict; "–" = no living hop pair.)

**What this changes.** No cell has R > 1: nothing amplifies. The "endemic" cells are chains whose
mean load is *flat* across hops (Qwen2.5-7B p=2: 29, 29, 29, 29; p=5: 8, 8, 8, 8), which is the
signature of hosts emitting their full T=96 tokens as pure payload every hop — a locked
repetition loop whose load is fixed by T/p, not an epidemic with a reproduction number. The
dying chains decay by ≈0.5–0.7× per hop until extinction. So the toy's "R0 ≈ T/(p·k_thresh),
falling with length" is not what the transformer data show: what varies with model and p is
the *probability of locking* (binary per trial, hence the 20–60% survival figures on 5 trials),
and the decay rate of unlocked chains. The "worm gets worse at scale" claim reduces to "a 7B
model locks into a payload loop more readily than a 3B one", which the per-trial rerun with
CIs (`deploy/run_trackC_p1.sh`) can test directly, and which should be read against Holtzman
et al. (2019) on degenerate repetition rather than against an epidemic model.

## E6-scale — the worm at scale: it gets worse (Track C) (2026-07-02)

> **Status (2026-11-04): superseded by "E6 at power" below** (60 trials per cell, three seeds; a
> seeding-strength sweep at 3 hops with 30 trials). The 5-trial survival figures and the 1-hop
> matrix here stand as what was run; the conclusions do not: only the Qwen2.5-7B *base* model
> sustains the loop at power (the instruct model falls to ≈0.1), and no same-tokenizer advantage
> is detectable at 3 hops at any seeding strength.

**Part A — serial passage on larger/instruct models** (survival across 6 hops, sweep payload
length p):

| model | p=1 | p=2 | p=3 | p=5 | p=8 |
|---|---|---|---|---|---|
| Pythia-2.8B | 40%\* | 0% | 0% | 0% | 0% |
| **Qwen2.5-7B** | 20%\* | **60%** | 40% | 40% | 40% |
| Llama-3.2-3B | 20%\* | 0% | 0% | 0% | 0% |
| **Qwen2.5-7B-Instruct** | 0% | 40% | 20% | 20% | 40% |

**The worm's critical length EXTENDS with capability — scale makes it *more* transmissible.**
GPT-2-small (Track A) and the smaller models here (Pythia-2.8B, Llama-3.2-3B) kill the worm for
p≥2, but **Qwen2.5-7B sustains it across all 6 hops for p=2–8** (viral load holds near 28 for
p=2), and **the instruct model transmits it too** (20–40% survival). A stronger copier is a
better host: the critical length is not fixed — it grows with capability, the opposite of a
comforting bound. (\*p=1 dies off the GPT-2 family — Pythia/Qwen/Llama don't lock single tokens,
E1.3 — so Track-A's "p=1 endemic" was a GPT-2 quirk.)

**Part B — cross-tokenizer firebreak matrix** (one small model per tokenizer family; patient-zero
X → host Y; surviving load at 1 hop, host0-reps 32):

| Xdown \ Yright | gpt2 | pythia | qwen | llama | gemma |
|---|---|---|---|---|---|
| **gpt2** | **13.4** | 8.0 | 4.4 | 9.9 | 12.1 |
| **pythia** | 8.0 | **12.9** | 15.5 | 4.6 | 10.2 |
| **qwen** | 13.5 | 9.6 | **15.3** | 16.9 | 6.6 |
| **llama** | 2.4 | 10.9 | 6.1 | **12.1** | 6.1 |
| **gemma** | 5.8 | 10.1 | 6.8 | 8.9 | **17.3** |

**The firebreak is real in DIRECTION but PARTIAL, not absolute, at scale.** The same-tokenizer
diagonal is generally the strongest transmission in its row (gpt2→gpt2 13.4, qwen→qwen 15.3,
gemma→gemma 17.3), confirming re-tokenization attenuates transmission. But under strong seeding
(32 copies) the off-diagonal does **not** collapse to 0 as in Track A (GPT-2→Pythia-160m) — a
heavily-repeated payload re-tokenizes into pieces the new host still partly copies (qwen→llama
16.9 and pythia→qwen 15.5 even exceed some diagonals). So a **heterogeneous-tokenizer pipeline
helps (same-tokenizer is worst-case) but is not a hard firebreak** against a strongly-seeded
payload; the clean Track-A collapse was the weak-signal regime. Caveats: small models (0.5–2B)
per family, p=3, 8 trials; the diagonal/off-diagonal separation is a tendency, not a clean gap.
`results/e6_scale.json`.

## E7 rerun — detector vs n-gram baseline, 32 negatives (2026-11-03, local CPU)

Supersedes the E7 run above (3 clean negatives, no baseline, no FPR). Command (the only change
from defaults is `HF_HUB_OFFLINE=1`, to force the cached gpt2 and make the run offline — it does
not affect the computation):

```
HF_HUB_OFFLINE=1 python -m experiments.e7_detector --snapshot trackA
# defaults: payload-p=3, ks=[2,4,8,16], instances=2, k-heads=8, ngram=3, seed=7
```

Induction heads scored: `[(5,5),(7,10),(6,9),(5,1),(7,2),(9,6),(10,1),(9,9)]`. 32 clean
paragraphs (negatives), 8 benign-repetitive controls, 2 payloads × 32 docs = 64 poisoned
instances per k. AUC = P(poisoned > clean), bootstrap 95% CI in brackets.

| detector | clean mean (max=thr) | k=2 | k=4 | k=8 | k=16 | benign-rep. FPR @thr |
|---|---|---|---|---|---|---|
| **induction** (mean head engagement) | 0.0001 (0.0022) | 0.0229 | 0.0606 | 0.0779 | 0.0800 | **1.00 (8/8)** |
| induction **AUC** | — | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | |
| **ngram3_repeat** (model-free) | 0.0000 (0.0000) | 0.0267 | 0.1609 | 0.3420 | 0.5402 | 0.88 (7/8) |
| ngram3 **AUC** | — | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | |

**Does the induction score beat the n-gram count? No.** Both detectors separate poisoned from
clean *perfectly* — AUC = 1.000 with a degenerate [1.000, 1.000] bootstrap CI at every k, from
k=2 up. They are statistically indistinguishable on the discrimination task because clean prose
carries essentially zero of either signal (induction clean mean 1e-4, n-gram clean mean exactly
0), so any planted repeated span is trivially separable by either. The honest reading is the one
the task anticipated: **the detector is an n-gram repeat detector; the induction machinery
explains *why* repetition is dangerous (E4) but is not a better alarm than a model-free count of
repeated 3-grams.** The one place they differ, the n-gram baseline is slightly *better*: at the
zero-FP-on-clean operating threshold the benign-repetitive false-positive rate is **1.00 for the
induction detector (8/8: license_header, email_footer, code_loop, numbered_list, chorus,
csv_dump, form_letter, menu) vs 0.88 for n-gram (7/8 — it alone does not flag the license
header)**. The induction score's advertised edge (keying on *trust* independent of usefulness)
buys nothing here and costs one extra false positive. The only qualitative difference is dynamic
range — the n-gram mean climbs 0.03→0.54 across k while induction saturates ~0.08 — but since
both already reach AUC 1.0 at k=2, that range does not translate into better detection on this
benchmark.

Caveats: GPT-2-small, CPU, ~65-token docs, p=3, 2 payloads/doc, seed 7; 71 s. Both detectors are
evaluated against the *same* benign-repetitive controls, so the FPR gap is not a sampling
artifact of different thresholds. Raw: `data/trackA/e7_detector.json` (+ `.png`); the superseded
June 3-negative run is preserved as `data/trackA/e7_detector_2026-06-26_3neg.json`.

## E8-lite rerun — true random control (2026-11-03, local CPU)

Supersedes the E8-lite run above, whose "ablate random" column was in fact the 8 *lowest*-
induction heads (a control biased toward unimportant heads). This run ablates the induction set
against **both** a genuinely random matched control (8 heads drawn, seeded, from the
non-induction heads, as in E2) **and** the lowest-induction set, under their true names. Command
(only `HF_HUB_OFFLINE=1` added to defaults):

```
HF_HUB_OFFLINE=1 python -m experiments.e8_substitution --snapshot trackA
# defaults: ms=[2,4,8,16], n-tables=8, k-heads=8, seed=7
```

Induction heads `[(5,5),(7,10),(6,9),(5,1),(7,2),(9,6),(10,1),(9,9)]`; random control
`[(11,6),(9,4),(1,11),(2,6),(3,6),(8,1),(7,6),(11,9)]`; lowest-induction control
`[(4,11),(3,3),(3,7),(3,2),(3,8),(2,2),(4,0),(3,6)]`. Substitution top-1 accuracy (8 tables × m
queries/cell; Δ vs clean in parentheses):

| m | clean | ablate induction | ablate TRUE random | ablate lowest-induction |
|---|---|---|---|---|
| 2 | 0.312 | 0.188 (−40%) | 0.125 (−60%) | 0.375 (+20%) |
| 4 | 0.500 | 0.344 (−31%) | 0.438 (−12%) | 0.406 (−19%) |
| 8 | 0.469 | **0.203 (−57%)** | 0.312 (−33%) | 0.438 (−7%) |
| 16 | 0.469 | **0.148 (−68%)** | 0.453 (−3%) | 0.312 (−33%) |

**Is the induction-ablation drop still much larger than the random-ablation drop? Yes at the
reliable table sizes, but the claim is weaker than the old mislabeled comparison implied.** At
the two largest, least-noisy cells the induction ablation collapses substitution far more than a
*true* random ablation: m=8, −57% vs −33%; m=16, −68% vs −3% (random is essentially unchanged
from clean at m=16). The lowest-induction control barely moves (−7% at m=8), confirming the
original intuition that unimportant heads are harmless — the old run's only error was calling
*that* set "random". But the honest correction is that a genuine random non-induction ablation is
**not** harmless: it removes ~33% of accuracy at m=8 (the old mislabeled column showed only −6%),
so the induction-specific effect is the *gap* between the induction drop and a real random drop,
not the near-total contrast the first write-up suggested. The m=2–4 rows are noisy (16–32 lookups)
and even invert at m=2; the induction>random ordering only stabilizes at m≥8. **In-context
substitution is induction-carried, but the effect size is moderate and the earlier "−6% random"
baseline overstated it.**

The cascade finding is unchanged: single-step fidelity with two tables present is high
(step-1 acc **0.969**), yet the direct 2-step cascade succeeds only **0.188**, *below* the
optimistic `f²=0.250` (f=0.500) — self-extraction depth is bounded worse than the `f^(L·depth)`
ladder predicts, because the intermediate value is both a value and a key and splits the
induction lookup.

Caveats: GPT-2-small, CPU, 8 tables × m queries/cell (small-m rows noisy), top-1 metric, greedy
2-step, seed 7; 92 s. No bootstrap CIs — point estimates only, so read the −3%/−33% random-column
swings between m=8 and m=16 as sampling noise, not structure. Raw: `data/trackA/e8_substitution.json`
(+ `.png`); the superseded June run is preserved as `data/trackA/e8_substitution_2026-06-26.json`.

## E1.1 rerun — curves committed, direct one-shot rate (2026-11-03)

Not a supersession — the same command as the original E1.1 (2026-06-25), rerun so the per-N
*curves* are committed as raw data for the first time (`data/trackA/e1_1_gpt2.json`) and the
direct one-shot copy rate P(N=2) is reported with Wilson intervals. It is sampled (T=1.0), so
the numbers differ slightly from the June entry. Command (only `HF_HUB_OFFLINE=1` added):

```
HF_HUB_OFFLINE=1 python -m experiments.e1_condensation_knee --model gpt2 \
    --lengths 1 2 3 5 --reps 1 2 4 8 16 32 --payloads 4 --trials 10 --max-new 48 --snapshot trackA
HF_HUB_OFFLINE=1 python -m experiments.analysis_knee_floor data/trackA/e1_1_gpt2.json \
    --out data/trackA/analysis_knee_floor_gpt2.json
```

Device MPS (generation-only), 4 payloads × 10 trials = 40 samples/cell, 990 s.

**Knee table** (reproduces the June shape: N\* decreases with length):

| p | knee N* (95% CI) | width | scrambled P@N=32 | no-payload P | June N* |
|---|---|---|---|---|---|
| 1 | 7.1 (4.7–9.7) | 8.0 | 0.95 | 0.00 | 7.2 |
| 2 | 3.6 (1.6–5.3) | 4.2 | 0.78 | 0.00 | 4.3 |
| 3 | 3.2 (2.7–4.8) | 3.4 | 0.03 | 0.00 | 3.0 |
| 5 | 1.8 (1.6–2.3) | 1.8 | 0.00 | 0.025 | 1.9 |

**Direct one-shot copy rate** (Wilson 95% intervals, from the committed curves):

| p | P(N=1) | P(N=2) = one prior occurrence | at grid floor? |
|---|---|---|---|
| 1 | 0.00 [0.00, 0.09] | 0.00 [0.00, 0.09] | no |
| 2 | 0.00 [0.00, 0.09] | 0.28 [0.16, 0.43] | no |
| 3 | 0.00 [0.00, 0.09] | 0.20 [0.11, 0.35] | no |
| 5 | 0.03 [0.00, 0.13] | **0.60 [0.45, 0.74]** | **yes** |

**The one-shot copy rate on GPT-2-small is modest and length-dependent.** A single prior
occurrence of the span (N=2) suffices more than half the time only at p=5 (0.60 [0.45, 0.74]);
at p=2 and p=3 it is 0.28 and 0.20, with Wilson intervals that exclude 1/2. So on GPT-2-small
alone only 1 of the 2 multi-token cells (p≥3) sits at the grid floor, and the direct one-shot
rate over p≥3 cells has median 0.40. This does **not** contradict the README C1/C3 status
(which attributes "one occurrence suffices in 36/48 cells" to the *at-scale* models, not
GPT-2-small) — it quantifies, with intervals, that the small model is the weak end of that
range. Separately, P(N=1) ≈ 0 at every length (≤0.03), which is the assumption the knee-floor
analysis relies on to invert N\* into an implied one-shot rate; the committed curves now
confirm it directly rather than by stipulation.

Caveats: GPT-2-small, MPS, T=1.0, 40 samples/cell, seed 7 — intervals are wide; the p=2 vs p=3
inversion (0.28 vs 0.20) is within overlapping CIs, not a real ordering. Raw curves + analysis:
`data/trackA/e1_1_gpt2.json`, `data/trackA/analysis_knee_floor_gpt2.json` (+ the `_knee_`/`_Nstar_`
PNGs). The recovered June run is preserved as `data/trackA/e1_1_gpt2_2026-06-25.json` (+ PNGs).

## E6 rerun — per-trial R0 and survival CIs (2026-11-03)

The script now stores every trial's trajectory, so R0 (per-hop load ratio) and survival get
per-trial bootstrap CIs — which the committed Track C data (mean loads only) could not give.
Command (only `HF_HUB_OFFLINE=1` added to defaults; GPT-2 serial passage + GPT-2→Pythia-160M
cross-tokenizer, 8 trials, 6 hops, host0-reps 16, T=96, seed 7, MPS, 1001 s):

```
HF_HUB_OFFLINE=1 python -m experiments.e6_worm --trials 8 --snapshot trackA
HF_HUB_OFFLINE=1 python -m experiments.analysis_e6_r0 data/trackA/e6_worm.json \
    --out data/trackA/analysis_e6_r0_gpt2.json
```

Two bugs in the iter4 pre-run code had to be fixed first (both committed with this entry; neither
changes any computed number): (1) `e6_worm.py plot_cross` tried to plot the new per-trial `[runs]`
arrays as lines and crashed *before* the JSON was written — now it plots only the mean curves;
(2) `analysis_e6_r0.py` only accepted the model-keyed `serial={model:{p:…}}` of the scale/
transcribed files, but `e6_worm.py` writes the flat `serial={p:…}` — it now normalizes the flat
form to model-keyed.

**Serial passage (GPT-2):**

| p | mean load per hop | R_geo (mean traj) | per-trial R_geo median [95% CI] | survive [95% CI] | fully-locked | endemic? |
|---|---|---|---|---|---|---|
| 1 | 40→36→26→24→14→2→0 | 0.54 | 0.47 [0.11, 0.62] | 0.25 [0.00, 0.50] | **2/8** | no (mean → 0) |
| 2 | 30→18→8→6→2→0→0 | 0.51 | 0.76 [0.61, 0.96] | 0.00 [0.00, 0.00] | 0/8 | no |
| 3 | 14→8→0→… | 0.57 | 0.38 [0.25, 0.66] | 0.00 [0.00, 0.00] | 0/8 | no |
| 5 | 16→9→6→6→6→5→3 | 0.77 | 0.75 [0.26, 0.99] | 0.25 [0.00, 0.50] | **1/8** | yes (mean) |
| 8 | 8→1→0→… | 0.13 | 0.33 [0.17, 0.50] | 0.00 [0.00, 0.00] | 0/8 | no |

Cross-tokenizer (p=3, patient zero GPT-2): same-host `13→0.4→0→…`, cross-host (Pythia-160M)
`19→6.8→2.5→2.0→0→…` — both extinct, the cross chain lasting one hop longer this seed (n=8, not
a reversal of the firebreak direction).

**How many "endemic" p=1 trials are locked loops vs genuinely fluctuating? All of the survivors
are locked; none fluctuate.** The per-trial data answers the question the mean hid. The June
Track-A "p=1 endemic" does **not** reproduce: here p=1's mean load decays to ~0 by the last hop,
only **2 of 8** trials are still alive at hop 6, and both are **fully-locked loops**
(`fully_locked=2/8`, zero surviving-but-fluctuating trials). The single cell endemic on the mean,
p=5, is **1 of 8** trials locked. So "endemic" is one or two hosts stuck emitting the payload on
repeat, not a chain that spreads. **R_geo < 1 at every payload length** (median 0.33–0.77; every
95% CI caps below 1 — p=5's 0.99 is the only one that reaches it), i.e. the mean viral load
shrinks every hop in all cells. This is the first *per-trial, CI-backed* confirmation on
GPT-2-small of the Analyses §B–C / README-C6 reinterpretation: no per-hop reproduction ratio
above one, and "endemic" chains are locked repetition loops.

Caveats: GPT-2-small + Pythia-160M, MPS, 8 trials (survival CIs are very wide — [0.00, 0.50] for
both endemic-ish cells, so "endemic" is not statistically separable from extinction at this n),
fresh sampled payloads per trial, seed 7. Numbers differ from the June Track-A run (sampling +
n=8); the robust, reproducible statement is R_geo < 1 everywhere. Raw: `data/trackA/e6_worm.json`
(+ `_serial`/`_crosstok` PNGs), analysis `data/trackA/analysis_e6_r0_gpt2.json`; the recovered
June run is preserved as `data/trackA/e6_worm_2026-06-26.json` (+ dated PNGs).

## E6 at power — serial passage with 60 trials per cell; firebreak as a seeding-strength curve at 3 hops (2026-11-04, RunPod L40S; plan item 2.1)

The queued `deploy/run_trackC_p1.sh`, run unattended on a RunPod L40S 48 GB (secure cloud,
image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1`, torch 2.5.1+cu121, transformers 4.57.6, bf16;
`requirements-lock.txt` is this venv). Script defaults, nothing changed:

```
deploy/run_trackC_p1.sh        # SEEDS="7 8 9" TRIALS_A=20  REPS_SWEEP="8 16 32" TRIALS_B=30  SNAPSHOT=trackC
#  Part A: python -m experiments.e6_scale --part A --seed {7,8,9} --trials 20   (hops 6, host0-reps 16, T 96, T=1.0)
#  Part B: python -m experiments.e6_scale --part B --host0-reps {8,16,32} --trials 30   (p=3, 3 hops)
```

Files: `data/trackC/e6A_hiN_s{7,8,9}.{json,log,png}`, `e6B_reps{8,16,32}.{json,log,png}`, pod
console `logs/pod_p1_console.log`. Analyses: `analysis_e6_r0_pooled_hiN.json`
(`analysis_e6_r0.py --pool` over the three seeds), `analysis_e6_firebreak_sweep.json`.
Wall time 7.3 pod-hours (Part A 88 min per seed; Part B 60 min per seeding strength).

### Part A — survival at hop 6, three seeds pooled (n = 60 chains per cell, Wilson 95%)

| model | p=1 | p=2 | p=3 | p=5 | p=8 |
|---|---|---|---|---|---|
| Pythia-2.8B | 0.23 (0.14–0.35) | 0.12 (0.06–0.22) | 0.07 (0.03–0.16) | 0.20 (0.12–0.32) | 0.05 (0.02–0.14) |
| **Qwen2.5-7B** | 0.22 (0.13–0.34) | **0.47 (0.35–0.59)** | **0.30 (0.20–0.43)** | **0.42 (0.30–0.54)** | **0.50 (0.38–0.62)** |
| Llama-3.2-3B | 0.33 (0.23–0.46) | 0.00 (0.00–0.06) | 0.12 (0.06–0.22) | 0.07 (0.03–0.16) | 0.00 (0.00–0.06) |
| Qwen2.5-7B-Instruct | 0.30 (0.20–0.43) | 0.13 (0.07–0.24) | 0.13 (0.07–0.24) | 0.08 (0.04–0.18) | 0.13 (0.07–0.24) |

Per-seed survival (p = 2/3/5/8) for the Qwen base model: 0.45/0.35/0.40/0.45, 0.55/0.25/0.50/0.55,
0.40/0.30/0.35/0.50 — the three seeds agree. The p = 1 column is **not transmission**: the
surviving "load" is 1 and a one-token payload decodes to a short common substring that
`count_payload` matches by accident (`tcc/payloads.py` docstring); read that column as the
false-positive floor of text-level counting, ≈0.2–0.3.

**What the surviving chains are.** Among chains alive at hop 6, the last-hop load equals T/p —
the maximum possible, i.e. the host's entire 96-token output is payload: Qwen2.5-7B median/max
load 48/48 at p=2, 32/32 at p=3, 18/18 at p=5, 12/13 at p=8 (T/p = 48, 32, 19, 12); the same
for the instruct model and for Pythia at p ≥ 3. Per-trial R_geo medians (bootstrap 95%) are
0.99 (0.90–1.00), 0.88 (0.72–0.98), 0.97 (0.91–0.99), 1.00 (0.95–1.00) for Qwen2.5-7B at
p = 2, 3, 5, 8, and 0.57–0.87 for Pythia, 0.63–0.71 for Llama, 0.69–0.80 for the instruct
model. **No cell has R > 1.** Survival is the per-chain probability of falling into a locked
repetition loop; everything that does not lock decays by ≈0.6–0.9× per hop and dies.

**Verdict on "the worm gets worse at scale".** With 60 trials the only model that locks in a
large fraction of chains is the Qwen2.5-7B **base** model (0.30–0.50 for p = 2–8, intervals clear
of every other model's). **Its survival is flat in p** — 0.47, 0.30, 0.42, 0.50 from p = 2 to 8,
every pair within the others' intervals — so there is no critical length in the one model that
transmits at all: the toy's "R0 falls with payload length" is not weakened here, it is absent. Pythia-2.8B and Llama-3.2-3B lock in 0–0.20, and — the change from the
5-trial run — **Qwen2.5-7B-Instruct locks in only 0.08–0.13**, indistinguishable from the 3B
base models. The earlier sentence "the instruct model transmits it too (20–40%)" was the top of
a 5-trial interval. The defensible statement is: *one 7B base model (Qwen2.5-7B) falls into a
payload loop in 30–50% of chains for every p from 2 to 8; a 2.8B, a 3B and the instruction-tuned
7B do so in ≤ 20%.* Whether that is scale or family is confounded (the one 7B base model here is
also the one Qwen base model); Pythia-6.9B would separate them and was not run.

### Part B — cross-tokenizer matrix at 3 hops vs seeding strength (p = 3, 30 trials per cell)

Rows = patient-zero family, columns = host family; entry = mean surviving load at hop 3.

host0-reps = 8:

| | gpt2 | pythia | qwen | llama | gemma |
|---|---|---|---|---|---|
| **gpt2** | **2.1** | 1.1 | 2.1 | 2.3 | 2.1 |
| **pythia** | 3.3 | **1.1** | 5.6 | 0.0 | 2.3 |
| **qwen** | 2.1 | 6.1 | **5.4** | 2.7 | 1.8 |
| **llama** | 0.6 | 0.0 | 3.1 | **0.0** | 2.1 |
| **gemma** | 0.0 | 0.8 | 7.6 | 2.1 | **1.7** |

host0-reps = 16:

| | gpt2 | pythia | qwen | llama | gemma |
|---|---|---|---|---|---|
| **gpt2** | **2.6** | 4.5 | 2.9 | 1.3 | 3.4 |
| **pythia** | 3.5 | **4.4** | 8.2 | 3.7 | 4.1 |
| **qwen** | 1.4 | 4.4 | **4.1** | 5.8 | 9.0 |
| **llama** | 2.1 | 0.0 | 5.3 | **2.7** | 5.2 |
| **gemma** | 5.0 | 2.2 | 6.9 | 3.7 | **5.6** |

host0-reps = 32:

| | gpt2 | pythia | qwen | llama | gemma |
|---|---|---|---|---|---|
| **gpt2** | **2.1** | 0.4 | 6.2 | 3.1 | 3.6 |
| **pythia** | 1.6 | **3.8** | 10.1 | 3.2 | 4.6 |
| **qwen** | 4.7 | 6.4 | **13.1** | 6.3 | 10.5 |
| **llama** | 0.0 | 2.1 | 1.9 | **0.1** | 7.4 |
| **gemma** | 3.2 | 5.2 | 9.6 | 3.7 | **0.7** |

| seed copies | diag mean | off-diag mean | ratio (bootstrap 95%, 30 trials/cell) | diag is row-max | permutation p (3125 assignments) | host (column) means gpt2 / pythia / qwen / llama / gemma |
|---|---|---|---|---|---|---|
| 8 | 2.07 | 2.39 | 0.87 (0.42–1.51) | 0 / 5 | 0.593 | 1.6 / 1.8 / **4.8** / 1.4 / 2.0 |
| 16 | 3.86 | 4.13 | 0.94 (0.54–1.45) | 0 / 5 | 0.594 | 2.9 / 3.1 / **5.5** / 3.4 / 5.5 |
| 32 | 3.99 | 4.69 | 0.85 (0.55–1.22) | 1 / 5 | 0.654 | 2.3 / 3.6 / **8.2** / 3.3 / 5.4 |

**Verdict on the firebreak.** At three hops the test finds no same-tokenizer advantage at any
seeding strength: the diagonal/off-diagonal ratio is below 1 at 8, 16 and 32 copies, every
bootstrap interval spans 1, the diagonal is the row maximum in 0–1 of 5 rows, and the
permutation p is ≈0.6 each time. **But the honest summary is that three hops on 0.5–2B hosts is
below the floor where a firebreak could show.** Only 10%, 17% and 18% of chains (8/16/32 copies)
are still alive at hop 3 in the whole matrix, the same fraction on the diagonal (0.10, 0.16,
0.17) as off it (0.10, 0.17, 0.18); mean loads of 2–8 out of a possible 32 are averages over
mostly-extinct chains, and the bootstrap ratio intervals are half a unit wide. The design can
rule out a *large* same-tokenizer advantage at 3 hops; it cannot see a modest one. The one
configuration in which an effect was measured (1 hop) is rerun at n = 30 in the follow-up
below. What the matrices do show is a **host effect**: the Qwen2.5-0.5B column is the
largest in every sweep (8.2 vs 2.3–5.4 at 32 copies), i.e. gemma→qwen (9.6) and pythia→qwen
(10.1) transmit far better than gemma→gemma (0.7) or pythia→pythia (3.8); by chains alive at
hop 3, the Qwen host keeps 0.31 vs 0.09–0.21 for the other four. Which model *hosts* the hop
matters; whether its tokenizer matches the payload's does not, once the payload has passed
through more than one host. Read together with Part A — where the Qwen2.5-7B *base* model is
the only one that locks in 30–50% of chains — this is one pattern, not two: **Qwen2.5 base
models fall into repetition loops more readily than Pythia, Llama, Gemma or GPT-2.** That makes
the scale-vs-family confound named above more likely family; Qwen2.5-3B base in the same
protocol (follow-up) decides it. The only configuration in which a same-tokenizer advantage has been
measured remains the 2026-07-02 run at **1 hop** with 32 copies (ratio 1.6, p = 0.004). The
"partial firebreak" should therefore be stated as a one-hop attenuation and withdrawn as a
multi-hop claim; plan item 2.4 (the firebreak matrix at ≥ 7B per family) is not triggered by
this result, since there is no partial transmission left to firm up at 3 hops.

**Caveats.** Part A: four models, one seed-copy strength (16), T = 96, sampled at T = 1.0, 6 hops;
text-level payload counting (p = 1 invalid, see above). Part B: 0.5–2B hosts, p = 3 only, 3 hops;
mean loads of 2–8 are small, so the matrices are noisy even at 30 trials (the bootstrap ratio
intervals are ±0.4–0.5 wide), and a 1-hop sweep at the same n was not run — it would test the
only reading under which the firebreak survives.

## E4-scale v2 — poisonability of a planted probe vs usefulness, 32 facts, three instruct models (2026-11-04, RunPod L40S; plan item 2.2)

The redesigned `experiments/e4_scale.py` (plan item 0.2): non-repetitive cells built from
distinct sentences and length-matched in tokens (selftest-checked; the run records
`nonrep_sentences_repeat = false` for all 96 non-repetitive contexts), poisonability measured
two ways — P_copy (copy-rate of a raw continuation; ≈1 for any repeated span by construction) and
**P_probe** (an E1-style probe span of 4 tokens planted ×2 at the end; reproduced or not, greedy)
— plus U (chat-QA accuracy on the synthetic fact) and T (induction attention). 32 facts, k = 8,
bootstrap-over-facts 95% CIs on the correlations. Same pod and environment as 2.1; unattended
job `deploy/runpod_job.sh e4e5`, defaults unchanged:

```
python -m experiments.e4_scale --model <M> --device cuda --prefix results/e4_scale_v2_<tag> --snapshot trackC
```

Models: Qwen2.5-7B-Instruct and Llama-3.2-3B-Instruct (the plan's two families) plus
**Phi-3.5-mini-instruct**, an ungated third family added as a supplement while the gated-repo
token permission was being sorted out (it was fixed before the pod ran, so Llama ran too).
Files: `data/trackC/e4_scale_v2_{qwen7b,llama3b,phi35mini}-instruct.{json,png}`; the
`.console.log` copies are truncated (a tee/snapshot race, fixed in the job script) — every
number below is read from the JSON. 6 min per model.

**2×2 means** (P_probe also as count/32 with Wilson 95%):

| model | context | U | P_copy | P_probe | T |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | rep + useful | 0.94 | 1.00 | 0.66 (21/32; 0.48–0.80) | 0.077 |
| | rep + useless | 0.00 | 1.00 | **0.97** (31/32; 0.84–0.99) | **0.138** |
| | nonrep + useful | 0.81 | 0.20 | 0.25 (8/32; 0.13–0.42) | 0.073 |
| | nonrep + useless | 0.00 | 0.07 | 0.31 (10/32; 0.18–0.49) | 0.072 |
| Llama-3.2-3B-Instruct | rep + useful | 1.00 | 1.00 | 0.97 (31/32; 0.84–0.99) | 0.040 |
| | rep + useless | 0.00 | 0.97 | 1.00 (32/32; 0.89–1.00) | 0.071 |
| | nonrep + useful | 1.00 | 0.25 | 0.69 (22/32; 0.51–0.82) | 0.042 |
| | nonrep + useless | 0.00 | 0.28 | 0.72 (23/32; 0.55–0.84) | 0.039 |
| Phi-3.5-mini-instruct | rep + useful | 1.00 | 0.97 | 0.94 (30/32; 0.80–0.98) | 0.047 |
| | rep + useless | 0.00 | 1.00 | 1.00 (32/32; 0.89–1.00) | 0.088 |
| | nonrep + useful | 1.00 | 0.20 | 0.62 (20/32; 0.45–0.77) | 0.038 |
| | nonrep + useless | 0.00 | 0.04 | 0.44 (14/32; 0.28–0.61) | 0.036 |

**Correlations** (Pearson over the 128 context rows; bootstrap over facts):

| model | corr(P_probe, U) | corr(P_probe, T) | corr(P_copy, U) | corr(P_copy, T) |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | **−0.18 [−0.31, −0.04]** | **+0.38 [+0.26, +0.50]** | +0.13 [+0.04, +0.23] | +0.52 [+0.47, +0.56] |
| Llama-3.2-3B-Instruct | −0.04 [−0.14, +0.06] | +0.15 [+0.06, +0.25] | −0.00 [−0.11, +0.11] | +0.30 [+0.16, +0.41] |
| Phi-3.5-mini-instruct | +0.07 [+0.00, +0.16] | +0.34 [+0.26, +0.42] | +0.07 [+0.01, +0.14] | +0.58 [+0.55, +0.62] |

**Does poisonability follow usefulness or repetition?** Repetition, on all three models, with the
flawed design's confounds removed. Along the repetition axis, planting a probe into a context
that already repeats makes it reproduced far more often than planting it into length-matched
distinct prose: +0.41/+0.66 (useful/useless) on Qwen, +0.28/+0.28 on Llama, +0.32/+0.56 on Phi.
Along the usefulness axis, holding repetition fixed, usefulness never raises poisonability: the
most poisonable cell on every model is **repeated gibberish** (U = 0, P_probe 0.97–1.00), and
within the non-repetitive pair the useful and useless contexts are within each other's
intervals. corr(P_probe, T) is positive with the interval clear of zero on all three;
corr(P_probe, U) is zero or slightly negative. The ordering of T agrees: the repeated gibberish
draws the most induction attention on every model (0.138 / 0.071 / 0.088 vs 0.04–0.08
elsewhere). The quantitative dissociation is weaker than the first run's +0.71 vs +0.09, which was
inflated by a copy-rate that is ≈1 by construction (the P_copy columns above reproduce that
artifact: corr(P_copy, T) 0.30–0.58).

**What the measurement is, said plainly.** In the repetitive cells the probe is planted after
eight copies of a span, so the model is already in a repetition regime and continues whatever
comes last; in the non-repetitive cells the useful and useless contexts barely differ
(0.25 vs 0.31, 0.69 vs 0.72, 0.62 vs 0.44). The result is therefore *"a context that has put
the model into repetition mode makes any trailing span reproduced, and whether the context is
useful has no bearing on that"* — the E6 degeneration mechanism seen from the poisoning side,
with T acting as a proxy for how repetitive the context is rather than as an independently
measured "trust". That is the sentence the paper should carry; "trust" as the explanatory
variable is a label for the repetition axis, not something this design measures separately.

**Caveats.** probe_k = 2 and greedy decoding put Llama-3.2-3B-Instruct near the ceiling (it
reproduces a two-copy probe in 69–72% of *non-repetitive* contexts), which compresses its
correlations; a run at probe_k = 1 or with sampled trials would separate its cells better. T
differences are small in absolute terms (0.04–0.14), one seed, 32 facts, k = 8 only. The
synthetic facts are the usefulness manipulation; U is 0.81–1.00 wherever the fact is present and
0.00 wherever it is absent, as designed.

## E5-scale v2 — RAG hijack with positive controls, three query styles, three instruct models (2026-11-04, RunPod L40S; plan item 2.3)

`experiments/e5_scale.py` with the additions of iteration 4: three query styles (summarize /
extract three details / open-ended), two **no-query positive controls on the same contexts**
(raw: the documents as a bare prefix, no template; chat: the template with "Continue the
documents."), per-trial occupancies stored. ~4000-token benign document dump, payload p = 3,
8 trials per cell, greedy, 64-token answers. Same pod as 2.2, defaults unchanged:

```
python -m experiments.e5_scale --model <M> --context-tokens 4000 --device cuda --prefix results/e5_scale_v2_<tag> --snapshot trackC
```

Files: `data/trackC/e5_scale_v2_{qwen7b,llama3b,phi35mini}-instruct.{json,png}` (console copies
truncated as in 2.2; numbers from the JSON). 5–8 min per model.

**A. Placement** (k = 16, summarize query): mean occupancy **0.00** at every position 0, 0.25,
0.5, 0.75, 1.0 on all three models; no trial with occupancy ≥ 0.5.

**B/C. Repetition at the end, per query style, with the controls** — cell = mean occupancy
(fraction of the 8 trials with occupancy ≥ ½):

| model | k | summarize | extract | open | **raw continuation (control)** | chat "continue" (control) |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 2 | 0.00 (0.00) | 0.01 (0.00) | 0.00 (0.00) | 0.38 (0.50) | 0.01 (0.00) |
| | 4 | 0.01 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.63 (0.88) | 0.00 (0.00) |
| | 8 | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.70 (1.00) | 0.00 (0.00) |
| | 16 | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.59 (0.88) | 0.00 (0.00) |
| | 32 | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.68 (1.00) | 0.00 (0.00) |
| Llama-3.2-3B-Instruct | 2 | 0.01 (0.00) | 0.00 (0.00) | 0.01 (0.00) | 0.44 (0.62) | 0.00 (0.00) |
| | 4 | 0.01 (0.00) | 0.01 (0.00) | 0.01 (0.00) | 0.70 (1.00) | 0.00 (0.00) |
| | 8 | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.69 (1.00) | 0.00 (0.00) |
| | 16 | 0.00 (0.00) | 0.00 (0.00) | 0.01 (0.00) | 0.67 (1.00) | 0.09 (0.12) |
| | 32 | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.65 (1.00) | 0.00 (0.00) |
| Phi-3.5-mini-instruct | 2 | 0.01 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.75 (0.75) | 0.01 (0.00) |
| | 4 | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 1.00 (1.00) | 0.02 (0.00) |
| | 8 | 0.00 (0.00) | 0.01 (0.00) | 0.00 (0.00) | 1.00 (1.00) | 0.25 (0.25) |
| | 16 | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 1.00 (1.00) | 0.12 (0.12) |
| | 32 | 0.00 (0.00) | 0.01 (0.00) | 0.02 (0.00) | 1.00 (1.00) | 0.01 (0.00) |

(k = 1 rows are 0.00–0.01 everywhere including the controls.) Largest single-trial occupancy in
any query cell: 0.095 (Qwen), 0.047 (Llama), 0.078 (Phi). `control_ok = true` for all three.

**Is the null a harness artifact?** No. The same contexts, fed as a bare prefix with no chat
template, are hijacked on every model from k = 2 upward (mean occupancy 0.6–1.0, 88–100% of
trials at k ≥ 4), so the payloads lock in and the occupancy metric sees it. Under the chat
template with any of the three task queries, occupancy is 0.00–0.02 with **no trial above
0.10**, at every k up to 32 and every placement. The bound on C5 from 2026-07-02 stands and is
now interpretable: *a chat-templated task query defuses raw-repetition hijack on three instruct
families; the copy tendency is intact underneath (raw control).* The answer-corruption rate
(trials with occupancy ≥ ½) is 0/8 in all 45 query cells.

**A new observation from the chat control.** The chat template with the instruction "Continue
the documents." — no task, an explicit invitation to copy — also does not hijack (0.00–0.09 on
Qwen/Llama; Phi reaches 0.25 at k = 8, 2 of 8 trials). So it is not only the task query that
defuses: the chat-template framing itself moves generation out of the raw-continuation regime.
That sharpens the defensive reading (the residual RAG threat is semantic injection, not
repetition) and the mechanistic one (the lock-in seen in E1.4 is a property of raw continuation).

**Caveats.** 8 trials per cell (a zero-of-8 cell has Wilson upper bound 0.32 on the
*hijacked* fraction, though the per-trial occupancies are all < 0.10, which is the stronger
statement); greedy; p = 3; benign OOD payloads; 4k context only; one seed.

## E6 follow-ups — a second 7B base family, and the firebreak at 1 hop with n = 30 (2026-11-04, RunPod L40S)

Two runs raised by "E6 at power" above and by the advisor's reading of it; same pod setup; job
`deploy/runpod_job.sh followup`, 2.3 pod-hours. Files
`data/trackC/e6A_hiN_pythia69b_s{7,8,9}.{json,png,console.log}`,
`e6B_1hop_reps{8,16,32}.{json,png,console.log}`; analyses
`analysis_e6_r0_pooled_pythia69b.json`, `analysis_e6_firebreak_1hop_sweep.json`.

### A. Pythia-6.9B serial passage: scale or family?

Same protocol as Part A of 2.1 (hops 6, 16 seed copies, T = 96, sampled at T = 1.0, 20 trials ×
seeds 7, 8, 9):

```
python -m experiments.e6_scale --part A --serial-models EleutherAI/pythia-6.9b --seed {7,8,9} --trials 20 --device cuda --prefix results/e6A_hiN_pythia69b_s$seed --snapshot trackC
```

| model | p=1 (artifact) | p=2 | p=3 | p=5 | p=8 |
|---|---|---|---|---|---|
| **Pythia-6.9B** (n = 60, Wilson 95%) | 0.38 (0.27–0.51) | **0.07 (0.03–0.16)** | **0.10 (0.05–0.20)** | **0.10 (0.05–0.20)** | **0.10 (0.05–0.20)** |
| Qwen2.5-7B (from 2.1) | 0.22 | 0.47 (0.35–0.59) | 0.30 (0.20–0.43) | 0.42 (0.30–0.54) | 0.50 (0.38–0.62) |
| Pythia-2.8B (from 2.1) | 0.23 | 0.12 (0.06–0.22) | 0.07 (0.03–0.16) | 0.20 (0.12–0.32) | 0.05 (0.02–0.14) |

Per seed (p = 2/3/5/8): 0.20/0.20/0.10/0.05, 0.00/0.05/0.10/0.10, 0.00/0.05/0.10/0.15. Per-trial
R_geo medians 0.60 (0.42–0.83), 0.66 (0.50–0.87), 0.72 (0.57–0.88), 0.74 (0.58–0.84); the few
survivors sit at T/p (median last load 48, 32, 18, 10 for p = 2, 3, 5, 8).

**Verdict: family, not scale.** A second 7B base model locks in 7–10% of chains, the same as its
own 2.8B sibling and as Llama-3.2-3B, and 3–5× below Qwen2.5-7B (every Pythia-6.9B interval
lies below every Qwen2.5-7B interval at p ≥ 2). "The worm gets worse at scale" is withdrawn.
The surviving statement: *Qwen2.5 base models fall into a locked payload loop far more often
than Pythia or Llama base models at any size tested (2.8B–7B), and instruction-tuning removes
most of it (Qwen2.5-7B-Instruct 0.08–0.13).* Qwen2.5-3B base (follow-up 2) tests whether the
trait holds across Qwen sizes.

### B. The cross-tokenizer matrix at 1 hop, 30 trials, three seeding strengths

```
python -m experiments.e6_scale --part B --cross-hops 1 --host0-reps {8,16,32} --trials 30 --device cuda --prefix results/e6B_1hop_reps$reps --snapshot trackC
```

This is the configuration of the only run that ever showed a same-tokenizer advantage
(2026-07-02: 1 hop, 32 copies, 8 trials; ratio 1.61, p = 0.004). Rows = patient zero, columns =
host; entry = mean surviving load after one re-tokenization. The 32-copy matrix (the July
configuration):

| | gpt2 | pythia | qwen | llama | gemma |
|---|---|---|---|---|---|
| **gpt2** | **8.7** | 11.6 | 11.6 | 9.6 | 13.6 |
| **pythia** | 10.0 | **13.1** | 11.9 | 8.5 | 10.0 |
| **qwen** | 13.1 | 13.6 | **20.8** | 14.4 | 21.2 |
| **llama** | 7.0 | 4.6 | 11.0 | **10.2** | 13.0 |
| **gemma** | 12.5 | 9.1 | 14.9 | 13.0 | **8.6** |

| seed copies | diag mean | off-diag mean | ratio (bootstrap 95%) | diag is row-max | permutation p | alive at hop 1: diag / off | host (column) means gpt2 / pythia / qwen / llama / gemma |
|---|---|---|---|---|---|---|---|
| 8 | 7.08 | 6.15 | 1.15 (0.82–1.52) | 0 / 5 | 0.239 | 0.31 / 0.30 | 4.9 / 5.4 / **7.8** / 5.9 / 7.6 |
| 16 | 9.15 | 8.88 | 1.03 (0.80–1.29) | 1 / 5 | 0.399 | 0.39 / 0.41 | 7.0 / 7.3 / **12.3** / 9.4 / 8.6 |
| 32 | 12.27 | 11.72 | 1.05 (0.84–1.26) | 1 / 5 | 0.356 | 0.49 / 0.52 | 10.3 / 10.4 / **14.0** / 11.2 / 13.3 |
| *July: 32, n = 8* | *14.18* | *8.82* | *1.61* | *3 / 5* | *0.004* | — | — |

**Verdict: the firebreak does not replicate.** At the July configuration with 30 trials the
diagonal/off-diagonal ratio is 1.05 (0.84–1.26), not 1.61; at 8 and 16 copies it is 1.15 and
1.03 with intervals spanning 1; the diagonal is the row maximum in 0–1 of 5 rows. Unlike the
3-hop sweep this is not a floor problem: 30–51% of chains are alive after the hop, loads run
5–21, and the alive fraction is identical on and off the diagonal (0.31/0.30, 0.39/0.41,
0.49/0.52). The July ratio of 1.6 was an 8-trial fluctuation. What is present at 1 hop, as at
3, is the host effect: the Qwen2.5-0.5B column is the largest at every seeding strength
(7.8 / 12.3 / 14.0 against 4.9–13.3), and it is the best host for its own payloads (20.8), for
Gemma's (14.9) and for Pythia's (11.9). **Re-tokenization is not a firebreak at any hop count
or seeding strength tested; which model hosts the next hop is what matters, and Qwen2.5 hosts
it best.** The "tokenizer firebreak" (C6) is withdrawn at scale; the Track-A GPT-2→Pythia-160m
collapse remains a two-model observation at n = 8 that these matrices do not generalize.

**Caveats.** Hosts are 0.5–2B, one per family; p = 3 only; T = 96; sampled at T = 1.0; 30 trials
per cell. The 8-copy ratio interval reaches 1.52, so a modest same-tokenizer advantage under
weak seeding is not excluded by that matrix alone; it is absent at 16 and 32 copies and at
every 3-hop strength.

## Follow-ups 2 — Qwen2.5-3B serial passage, and E5 with the raw-plus-query control (2026-11-04, RunPod L40S)

Job `deploy/runpod_job.sh followup2`, 1.5 pod-hours. Files
`data/trackC/e6A_hiN_qwen3b_s{7,8,9}.{json,png,console.log}`,
`e5_scale_v2_rawq_{qwen7b,llama3b,phi35mini}-instruct.{json,png,console.log}`; analysis
`analysis_e6_r0_pooled_qwen3b.json`.

### C. Qwen2.5-3B serial passage: is the looping trait family-wide?

Same protocol as Part A of 2.1 and follow-up A (hops 6, 16 seed copies, T = 96, sampled at
T = 1.0, 20 trials × seeds 7, 8, 9):

```
python -m experiments.e6_scale --part A --serial-models Qwen/Qwen2.5-3B --seed {7,8,9} --trials 20 --device cuda --prefix results/e6A_hiN_qwen3b_s$seed --snapshot trackC
```

| model (n = 60, Wilson 95%) | p=1 (artifact) | p=2 | p=3 | p=5 | p=8 |
|---|---|---|---|---|---|
| **Qwen2.5-3B** | 0.30 (0.20–0.43) | **0.33 (0.23–0.46)** | **0.37 (0.26–0.49)** | **0.38 (0.27–0.51)** | **0.28 (0.19–0.41)** |
| Qwen2.5-7B (2.1) | 0.22 | 0.47 (0.35–0.59) | 0.30 (0.20–0.43) | 0.42 (0.30–0.54) | 0.50 (0.38–0.62) |
| Pythia-6.9B (follow-up A) | 0.38 | 0.07 (0.03–0.16) | 0.10 (0.05–0.20) | 0.10 (0.05–0.20) | 0.10 (0.05–0.20) |
| Pythia-2.8B / Llama-3.2-3B (2.1) | — | 0.12 / 0.00 | 0.07 / 0.12 | 0.20 / 0.07 | 0.05 / 0.00 |

Per seed (p = 2/3/5/8): 0.35/0.40/0.20/0.35, 0.25/0.25/0.60/0.15, 0.40/0.45/0.35/0.35. Per-trial
R_geo medians 0.88–0.99; survivors at T/p (median last load 48, 32, 18, 12).

**Verdict: the trait is the Qwen2.5 base family, at 3B as at 7B.** Qwen2.5-3B locks in 28–38%
of chains, flat in p, inside the 7B's intervals at every p and 3–5× above every non-Qwen base
model of 2.8B–7B. The ordering across the six base/instruct models now tested is Qwen2.5-7B ≈
Qwen2.5-3B ≫ Pythia-6.9B ≈ Pythia-2.8B ≈ Llama-3.2-3B ≈ Qwen2.5-7B-Instruct. Together with
follow-up B (the Qwen2.5-0.5B host column dominates every firebreak matrix), this is one fact:
*Qwen2.5 base models, at every size from 0.5B to 7B, fall into a locked repetition loop on a
repeated span far more readily than Pythia, Llama, Gemma or GPT-2, and instruction-tuning
removes most of it.* The paper's C6 should be rewritten around that sentence and read against
Holtzman et al. (2019); there is no scale law and no critical length in these data.

### D. E5-scale with the raw-plus-query control: template or query?

E5-scale v2 (above) found that the chat template with "Continue the documents." did not hijack
either, so the template and the task query were not separated. The `rawq` control added to
`e5_scale.py` feeds the experimental user message verbatim — the preamble, the documents, and
the task query — as a bare prefix with **no chat template**. Same seed, so the contexts and the
query cells are identical to v2 (verified: every query-cell mean matches v2 exactly).

```
python -m experiments.e5_scale --model <M> --context-tokens 4000 --device cuda --controls raw chat rawq --prefix results/e5_scale_v2_rawq_<tag> --snapshot trackC
```

Mean occupancy (fraction of 8 trials with occupancy ≥ ½), payload at the end, p = 3:

| model | k | task query (templated; 3 styles) | raw, no query | chat, "continue" | **rawq: no template, summarize / extract / open** |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 2 | 0.00–0.01 | 0.38 (0.50) | 0.01 | 0.04 / 0.02 / 0.00 (all 0.00) |
| | 8 | 0.00 | 0.70 (1.00) | 0.00 | 0.04 / 0.03 / 0.00 (all 0.00) |
| | 32 | 0.00 | 0.68 (1.00) | 0.00 | 0.00 / 0.03 / 0.00 (all 0.00) |
| Llama-3.2-3B-Instruct | 2 | 0.00–0.01 | 0.44 (0.62) | 0.00 | 0.01 / 0.00 / 0.01 (all 0.00) |
| | 8 | 0.00 | 0.69 (1.00) | 0.00 | 0.00 / 0.00 / 0.00 |
| | 32 | 0.00 | 0.65 (1.00) | 0.00 | 0.00 / 0.00 / 0.00 |
| Phi-3.5-mini-instruct | 2 | 0.00–0.01 | 0.75 (0.75) | 0.01 | 0.01 / 0.00 / 0.00 (all 0.00) |
| | 8 | 0.00–0.01 | 1.00 (1.00) | 0.25 (0.25) | 0.00 / 0.01 / 0.00 (all 0.00) |
| | 32 | 0.00–0.02 | 1.00 (1.00) | 0.01 | 0.01 / 0.01 / 0.02 (all 0.00) |

(k = 4 and 16 are the same picture; full tables in the JSON.) Largest single rawq trial: 0.36
(Qwen, summarize, k = 16), 0.06 (Llama), 0.08 (Phi); no rawq trial reaches ½ on any model.

**Verdict: the template is not what defuses.** The same text that is hijacked as a bare
prefix (0.4–1.0 occupancy) is not hijacked once the task query follows the documents, with or
without a chat template, on all three models. The v2 sentence "the chat template alone
defuses" is therefore wrong as an explanation: the template is sufficient but not necessary.

**But the honest reading is narrower than "the query defuses".** Every condition that does
not hijack — templated query, templated "continue", and now the untemplated query — puts
tokens *after* the payload block (the question line, the instruction, the template suffix),
whereas the only hijacking condition ends the context inside the repeated payload. E5-lite and
E1.5 showed on GPT-2 that about 16 tokens of anything between the payload and the generation
point kill reproduction, and the v2 placement sweep shows the same gate at length. So these
data cannot distinguish "the model follows the task" from "the payload is no longer the last
thing in context". The one-line control that separates them is a `rawpad` condition: the raw
documents followed by benign filler of the same token length as the query line, no question,
no template. If it hijacks, the query is doing semantic work; if it does not, the C5 bound is
the recency gate restated, and "a task query defuses hijack" should be replaced by "anything
that ends the context after the payload defuses it" — which is the same defensive advice with
a different mechanism. Not run; ten pod-minutes per model.

**Caveats.** 8 trials per cell, greedy, one seed, p = 3, 4k context; the query line is 15–30
tokens on these tokenizers, above the GPT-2 recency limit measured in E5-lite.
