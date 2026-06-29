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
