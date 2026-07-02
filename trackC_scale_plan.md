# Track C — the downstream claims at scale (C4/C5/C6), and an E2 tidy

Track B confirmed the **core** (condensation knee C1/C3, induction-head mechanism C2) to 7B
across four model families and instruct models, plus the E8 self-extraction capstone. The
**downstream** claims C4/C5/C6 so far have only *lite / proxy* results on GPT-2-small (Track A).
Track C replaces the toy/proxy with the real thing — **instruct models, real long-context RAG,
a real task, multi-tokenizer populations** — and tidies one rigor gap in the E2 ladder.

**Models on hand** (RunPod L40S 48 GB; re-fetch gated ones with `HF_TOKEN`): GPT-2 {…,xl},
Pythia {410m, 1.4b, 2.8b, 6.9b}, Qwen2.5 {…, 7B, 7B-Instruct}, Llama-3.2 {1B, 3B, 3B-Instruct},
Gemma-2-2b{,-it}. **Five distinct tokenizers**: GPT-2 (BPE), Pythia (NeoX), Qwen2, Llama-3,
Gemma — the substrate for the cross-tokenizer worm matrix.

Each item ships the same way as E2/E8: a new `experiments/*.py` (with a `--selftest` for the
model-free logic), local verification, push, a pod run command, then a `results_log.md` entry
and a paper section.

---

## 0. C2-tidy — Pythia-1.4B at `--k-frac 0.05` (rigor, ~3 min, no new code)

**Why.** In the E2-at-scale ladder Pythia-1.4B was run before `--k-frac` existed, ablating a
fixed **8/384 = 2.1%** of heads while gpt2-xl/2.8b/6.9b used **5%**. Uniformity only.

**Run.** `e2_induction --model EleutherAI/pythia-1.4b --k-frac 0.05 --device cuda
--lengths 1 3 5 --reps 2 3 4 8 16 --payloads 4`.

**Deliverable.** Replace the 1.4b row in the `results_log.md` E2-at-scale table and paper
`tab:e2scale`. **Expected:** still ~−80% at p=3 near the knee; no conclusion changes.

---

## 1. E6-scale — the worm at scale (C6)

**Current (E6, Track A).** Serial passage on GPT-2-small: an infected host's output (payload
`k` times) becomes the next host's context; measured a reproduction number **R0** and a
**critical string length** (length-1 endemic, longer dies), plus a **cross-tokenizer firebreak**
(GPT-2→Pythia-160M collapses). Generation-only.

**At scale (`experiments/e6_scale.py`, reuses `e6_worm.passage_chain`).**
- **A. Serial passage** on {Qwen2.5-7B, Qwen2.5-7B-Instruct, Llama-3.2-3B, Pythia-2.8b},
  sweeping payload length p ∈ {1,2,3,5,8}; per-hop viral load `k_out`, R0 estimate, critical
  length. Does the worm survive at 7B and on chat models?
- **B. Cross-tokenizer matrix.** Patient-zero payload native to model X's tokenizer; one hop to
  model Y; surviving load for X,Y over the 5 families → a 5×5 firebreak matrix. Predicted:
  off-diagonal (different tokenizer) collapses.
- **C. (optional) host population.** N hosts on a random contact graph, each hop passes output
  to a random peer; endemic prevalence vs die-out, vs p and mixing.

**Metric.** `k_out` vs `k_in` per hop (the transmission map); R0 = load growth near threshold /
endemic level; cross-tok survival ratio (off-diagonal / diagonal).
**Compute.** Generation-only, light — the pod handles it easily.
**Falsify.** Nothing propagates past patient zero at any p (no worm), or cross-tokenizer
transmits as well as same-tokenizer (no firebreak).

---

## 2. E4-scale — trust, not usefulness, with a REAL task (C4)

**Current (E4-lite).** 2×2 {repetitive, non-repetitive} × {useful, useless} on GPT-2-small,
where **"useful" was a naturalness proxy** (log-prob of a content unit). Found P (poisonability)
⊥ U (usefulness), ∥ T (induction trust): corr(P,U)=−0.20 vs corr(P,T)=+0.54.

**At scale (`experiments/e4_scale.py`).** Replace the proxy with a **real downstream task** on
an instruct model (a small QA / cloze set where a context genuinely helps answer). Build a 2×2:
- *repetitive + useful*: the answer-bearing fact repeated k times.
- *repetitive + useless*: an OOD payload repeated k times.
- *non-repetitive + useful*: the answer-bearing fact stated once in prose.
- *non-repetitive + useless*: irrelevant prose / random tokens.

Measure per context: **U** = real task benefit (accuracy / log-prob lift on the correct answer,
context vs no-context); **T** = induction/attention engagement on the repeated span (HF
`output_attentions`, attention paid to the token after a prior occurrence); **P** =
poisonability (does a planted OOD span in that context get reproduced / hijack, the E1/E5
measure). Show **P tracks T, not U**, with a *real* benefit.

**Compute.** Instruct model, moderate context; attention extraction is memory-ish — use a
smaller instruct model (Llama-3.2-3B-Instruct / gemma-2-2b-it) or a layer subset if the 7B is
tight. **Falsify.** P tracks task benefit U (the conservation law fails on a real task).
**Note.** Subtlest to design cleanly (defining "real usefulness" without confounding trust).

---

## 3. E5-scale — RAG hijack in a long context (C5)

**Current (E5-lite).** Scaled-down RAG on GPT-2-small (1024 ctx): a ~65-token prose doc with an
OOD payload ×k at position `pos`; hijack = payload's share of the continuation. Found
**hijack = entry (recency) × lock-in (knee)**; only an end-placed payload hijacks.

**At scale (`experiments/e5_scale.py`, reuses `e5_hijack` occupancy).** A realistic RAG prompt
on a chat model: a **user query** + **K retrieved passages** (real prose) filling **4k–32k
tokens**, with an OOD payload ×k planted inside **one** passage at position `pos`.
- **A. Placement** — which passage / where in the window (front / middle / end): tests the
  "lost-in-the-middle" recency profile *at length*.
- **B. Repetition** `k` — the lock-in knee inside a long, competitive context.
- **C. Bridge / size** — minimal well-placed span that still hijacks.

**Metric.** Hijack = payload occupancy of the generated answer (vs answering from the legit
passages); also answer-corruption rate. Look for entry×lock-in at length and a minimal-poison
bridge. **Compute.** Heaviest — long-context 7B (KV cache grows with context); 32k comfortable
on 48 GB, 128k needs quantization or a smaller model. **Falsify.** Hijack needs implausibly
large/tuned payloads, or the recency effect vanishes at length.

---

## Sequencing (cheap-and-quick → heavy-and-designy)

1. **C2-tidy** — one command, now.
2. **E6-scale** — lightest new code (generation-only worm).
3. **E4-scale** — real-task trust.
4. **E5-scale** — long-context RAG (most design + compute).

All on the RunPod L40S. Deliverable per item: script (+selftest), `results_log.md` entry, paper
section. When all four land, C1–C6 all have an at-scale (or real-setting) result and the paper's
"what remains" shrinks to just larger host populations / 128k-context.

## Status
- [ ] 0. C2-tidy (Pythia-1.4B at k-frac 0.05)
- [ ] 1. E6-scale (worm: serial passage + 5-tokenizer matrix)
- [ ] 2. E4-scale (trust vs usefulness, real task)
- [ ] 3. E5-scale (RAG hijack, long context)
