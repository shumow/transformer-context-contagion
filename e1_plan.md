# E1 — the condensation knee: detailed work plan

The keystone experiment of [`transformer_validation_plan.md`](transformer_validation_plan.md).
This expands the one-paragraph E1 into a concrete, staged spec. The Phase-0 smoke test
(see [`results_log.md`](results_log.md)) already showed the *qualitative* result on GPT-2:
a sharp knee in `P(reproduce)` vs repetition count `N`, knees `N*≈2–4` for multi-token
strings (length-robust), and the single-token payload hardest (`N*≈10`). E1-proper turns
that into a *measured* result — with controls, error bars, and a model-size sweep — and
sets up E2 (induction attribution).

## 0. What E1 must establish
- **C1** — a real transformer has a **condensation knee**: `P(model regenerates a
  repeated OOD string)` crosses a threshold sharply in `N`, not gradually.
- **C3** — the knee's location `N*(p)` is roughly **flat or decreasing** in string
  length `p` (induction-like length-robustness), not steeply rising (count-cache
  brittleness).
- Set up **C2** (E2): emit, for the cells near the knee, the contexts/activations E2
  will patch to attribute reproduction to induction heads — and explain the p=1 inversion.

## 1. Design — a factorial sweep

For each cell we estimate `P(reproduce)` and richer reproduction metrics, then extract the
knee `N*` (the interpolated `P=0.5` crossing) with a confidence interval.

### Factors
| Factor | Grid | Why |
|---|---|---|
| repetitions `N` (x-axis) | 1,2,3,4,6,8,12,16,24,32 (denser near the knee; adaptive refine at the crossing) | the knee lives in `N` |
| string length `p` (tokens) | 1,2,3,5,8,16,32 (subject to `N·p ≤ ctx`) | tests C3 over a wide range |
| model / size | GPT-2 {sm,med,lg,xl}; Pythia {160M,410M,1.4B,2.8B}; one modern small (Qwen2.5-0.5B/1.5B or Llama-3.2-1B), base **and** instruct | induction strength grows with scale → `N*` should move |
| decoding | greedy (`T=0`) **and** sampled `T∈{0.7,1.0}` | lock-in is a sampling phenomenon; greedy is the clean "does it copy at all" |
| payload type | rainbow (distinct, order-1), random (repeats, higher order), structured (de Bruijn-like, needs order>1) | exact vs approximate quine; sets up E3 |
| payload instance | `R≥8` distinct random payloads per `(p,type)` | average out which-tokens-got-picked — the smoke test's main weakness |

### Controls (the part the smoke test lacked)
1. **Spontaneous-emission control (`N=0`):** with no payload in context, `P(emit S)` must
   be ≈0 — else "reproduction" is the model's taste, not copying. Verifies the payload is
   genuinely OOD.
2. **Scrambled-context control:** put the payload's *tokens* in context `N` times but in a
   **non-repeating shuffled order** (same token frequencies, no repeated n-gram). If
   reproduction needs the *pattern* (induction) rather than mere token presence, the
   scrambled control should **not** lock. This isolates pattern-copying from "those tokens
   are now likely."
3. **Chance baseline:** the probability a *random* continuation scores transition fidelity
   `≥ τ` (depends on `p`, vocab) — computed empirically/analytically so `τ` and
   `P(reproduce)` are calibrated and the controls have a floor to beat.

### Metrics (per cell, aggregated over `R` payloads × `T` trials)
- **Primary:** `P(reproduce)` = fraction with transition fidelity `≥ τ`; **knee** `N*` =
  interpolated `P=0.5` crossing.
- **Knee sharpness:** fit a logistic `P(N)`; report `N*` **and** the transition width
  (sharp ⇒ condensation-like; gentle ramp ⇒ graded).
- **Secondary** (from `scoring.reproduction_metrics`): mean transition fidelity,
  occupancy, **run length** (how long the copy sustains once started), and **emergent
  period** (does it lock to the full `p` or a divisor — the toy's divisibility angle).

### Statistics
- Per cell: `R·T` Bernoulli samples → `P̂` with a **Wilson interval**. Size so the CI
  half-width near 0.5 is `< ~0.1` (e.g. `R=8 × T=16 = 128` → ±0.087; the smoke test's 8
  samples gave ±0.18 — insufficient).
- Knee CI by **bootstrap over payloads**.
- Fixed seeds; log every run to JSON/CSV with full config.

## 2. Confounds specific to E1
- **Tokenization instability:** payloads are token-level, but verify each decodes and
  **re-tokenizes to the same ids** (round-trip), else the in-context "string" isn't what we
  score. Add a round-trip filter to payload construction.
- **EOS / natural degeneration:** handle EOS; the scrambled control also guards against
  "the model just repeats whatever is in context."
- **Context-length vs `N` confound:** larger `N·p` pushes the payload further back, so a
  moving knee could be a context-length effect, not an `N` effect. Run a dedicated check
  holding `N·p` (total context) fixed while varying the split, and/or hold the gap from
  payload-end to the generation point at 0.
- **Chat/RLHF behavior:** instruct models may comment, summarize, or refuse rather than
  copy — keep base models for the clean mechanism, treat instruct as a separate realistic
  condition, never pooled.
- **The p=1 inversion:** confirm single-token difficulty isn't an artifact of the model's
  natural repetition dynamics / unigram, by comparing against the scrambled control and
  (in E2) induction-head engagement.

## 3. Staged runs
- **E1.0 — smoke test.** ✅ done (GPT-2, one payload/length). Knee confirmed qualitatively.
- **E1.1 — robust single model.** GPT-2-small, full `p×N` grid, `R=8` payloads `×T=16`,
  greedy + `T=1.0`, **all three controls**, Wilson CIs. The first *measured* knee curves.
  Runs on this 8 GB / MPS Mac.
- **E1.2 — payload-type contrast.** rainbow vs random vs structured at fixed model — does
  higher-order structure still lock? (bridges to E3).
- **E1.3 — size sweep.** GPT-2 {sm→xl}, Pythia {160M→2.8B}: how `N*` and knee sharpness
  move with scale. **Needs a bigger GPU** — the larger models won't fit 8 GB.
- **E1.4 — modern + chat.** Qwen2.5 / Llama-3.2 small, base vs instruct.
- **E1.5 — position/context-length control study** (the confound above).

## 4. Code changes (turn the smoke test into E1.1)
- **`tcc/payloads.py`:** round-trip-stable token filter; `scrambled_context(...)` builder;
  helper to draw `R` distinct payloads per `(p,type)`; `structured_payload` (de Bruijn-like).
- **`tcc/scoring.py`:** Wilson-interval helper; logistic fit returning `(N*, width)`;
  chance-level baseline for given `(p, vocab, τ)`.
- **`experiments/e1_condensation_knee.py`:** add the `N=0` and scrambled controls; aggregate
  over payloads with CIs; `--greedy` and temperature sweep; emit a JSON/CSV results table
  (not just a PNG); plot `P(N)` with CI bands and the controls overlaid; a second figure
  `N*` vs `p` with bootstrap CIs.
- **New:** `experiments/e1_size_sweep.py` (drives E1.3 over a model list) and a small
  `analysis/` for the `N*`-vs-size figure.

## 5. Deliverables
- Upgraded `e1_condensation_knee.py` + a JSON/CSV results table per run.
- Figures: (i) `P(reproduce)` vs `N` per length with CIs and controls; (ii) `N*` vs `p`
  (length-robustness); (iii) `N*` vs model size (scaling).
- An E1 section in `results_log.md`, folded later into the validation paper.

## 6. Success / falsification
- **Confirm C1** if a sharp knee with small `N*` (single digits) appears **robustly across
  payloads and models**, and sits **above the scrambled and chance controls**.
- **Confirm C3** if `N*(p)` is flat or decreasing over `p ∈ [2,32]`.
- **Falsify** if: no threshold (`P` gradual or flat); or `N*` climbs steeply with `p`; or
  the **scrambled control reproduces as well** (⇒ it is token-presence, not pattern-copying,
  and the "induction" reading is wrong).
