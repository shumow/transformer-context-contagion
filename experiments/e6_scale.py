"""
E6-scale -- the context-window worm at scale (C6, Track C).

E6 (Track A) showed on GPT-2-small that an infected host's output, fed back as the next
host's context, propagates: short payloads sustain across hops (endemic), long ones die (a
critical length), and re-tokenizing under a *different* tokenizer breaks transmission (a
cross-tokenizer firebreak). This scales that up:

  A. Serial passage on larger / instruct models (Pythia-2.8B, Qwen2.5-7B[-Instruct],
     Llama-3.2-3B), sweeping payload length p -- does the worm survive at 7B and on chat
     models, and where is the critical length?
  B. Cross-tokenizer firebreak MATRIX: one small model per tokenizer family (GPT-2, Pythia,
     Qwen2, Llama-3, Gemma); patient zero = model X (payload native to X's tokenizer), hops
     on model Y; the surviving load for every (X,Y). Predicted: the diagonal (same tokenizer)
     sustains, the off-diagonal (re-tokenized) collapses -- heterogeneous pipelines resist.

Reuses e6_worm.passage_chain (message passed as TEXT, decoded then re-tokenized, as agents /
RAG actually pass it). Benign OOD payloads only.

  python -m experiments.e6_scale --device cuda            # A + B
  python -m experiments.e6_scale --selftest               # model-free plumbing check
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import time
import numpy as np

from tcc import payloads
from experiments.e6_worm import passage_chain, count_payload

# one small model per distinct tokenizer family for the cross-tokenizer matrix (Part B).
FAMILY = {
    "gpt2":  "gpt2",
    "pythia": "EleutherAI/pythia-410m",
    "qwen":  "Qwen/Qwen2.5-0.5B",
    "llama": "meta-llama/Llama-3.2-1B",     # gated
    "gemma": "google/gemma-2-2b",           # gated
}
# larger / instruct models for the serial-passage-at-scale sweep (Part A).
SERIAL = ["EleutherAI/pythia-2.8b", "Qwen/Qwen2.5-7B", "meta-llama/Llama-3.2-3B",
          "Qwen/Qwen2.5-7B-Instruct"]


def serial_at_scale(name, args, LM, torch):
    lm = LM(name, device=args.device, dtype=getattr(torch, args.dtype))
    rng = np.random.default_rng(args.seed)
    pool = payloads.rare_token_pool(lm.tokenizer)
    curves = {}
    for p in args.lengths:
        runs = []
        for _ in range(args.trials):
            S = payloads.sample_payloads(pool, p, 1, rng, tokenizer=lm.tokenizer)[0]
            ptext = lm.tokenizer.decode(S)
            hosts = [lm] * (args.hops + 1)
            runs.append(passage_chain(hosts, S, ptext, args.host0_reps, args.T,
                                      args.hops, rng, args.temperature))
        arr = np.array(runs, float)
        curves[str(p)] = dict(mean=arr.mean(axis=0).tolist(),
                              survive=float(np.mean(arr[:, -1] >= 1)))
        print(f"  p={p:2d}  " + " -> ".join(f"{v:.1f}" for v in arr.mean(axis=0)) +
              f"   survive(last)={curves[str(p)]['survive']:.0%}")
    del lm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return curves


def cross_matrix(args, LM, torch):
    lms = {}
    for fam, name in FAMILY.items():
        try:
            lms[fam] = LM(name, device=args.device, dtype=getattr(torch, args.dtype))
            print(f"  loaded {fam} ({name})")
        except Exception as e:
            print(f"  SKIP {fam} ({name}): {str(e)[:100]}")
    fams = list(lms)
    rng = np.random.default_rng(args.seed + 1)
    p = args.cross_p
    mat = {}
    for X in fams:
        pool = payloads.rare_token_pool(lms[X].tokenizer)
        mat[X] = {}
        for Y in fams:
            loads = []
            for _ in range(args.trials):
                S = payloads.sample_payloads(pool, p, 1, rng, tokenizer=lms[X].tokenizer)[0]
                ptext = lms[X].tokenizer.decode(S)
                hosts = [lms[X]] + [lms[Y]] * args.cross_hops
                ks = passage_chain(hosts, S, ptext, args.host0_reps, args.T,
                                   args.cross_hops, rng, args.temperature)
                loads.append(ks[-1])                       # surviving load at last hop
            mat[X][Y] = float(np.mean(loads))
        print(f"  {X:7s} -> " + "  ".join(f"{Y}:{mat[X][Y]:.1f}" for Y in fams))
    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "all"], default="all")
    ap.add_argument("--serial-models", nargs="+", default=SERIAL)
    ap.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--cross-p", type=int, default=3)
    ap.add_argument("--hops", type=int, default=6, help="hops for serial passage (Part A)")
    ap.add_argument("--cross-hops", type=int, default=3, help="hops for the matrix (Part B)")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--host0-reps", type=int, default=16)
    ap.add_argument("--T", type=int, default=96, help="tokens generated per host")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--dtype", default="bfloat16",
                    help="bfloat16 is native for Qwen/Llama/Gemma and avoids gemma-2 fp16 overflow")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prefix", default="results/e6_scale")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    import torch
    from tcc.models import LM
    os.makedirs(os.path.dirname(args.prefix) or ".", exist_ok=True)
    t0 = time.time()
    out = json.load(open(f"{args.prefix}.json")) if os.path.exists(f"{args.prefix}.json") else {}
    out.setdefault("serial", {}); out.setdefault("cross", {})

    if args.part in ("A", "all"):
        print("\n=== A. serial passage at scale (sweep payload length) ===")
        for name in args.serial_models:
            if name in out["serial"]:
                print(f"{name}: cached, skip"); continue
            print(f"--- {name} ---")
            try:
                out["serial"][name] = serial_at_scale(name, args, LM, torch)
            except Exception as e:
                out["serial"][name] = {"error": str(e)[:200]}
                print(f"  FAILED: {str(e)[:150]}")
            json.dump(out, open(f"{args.prefix}.json", "w"), indent=2)

    if args.part in ("B", "all"):
        print(f"\n=== B. cross-tokenizer firebreak matrix (p={args.cross_p}, "
              f"{args.cross_hops} hops) ===")
        out["cross"] = cross_matrix(args, LM, torch)
        json.dump(out, open(f"{args.prefix}.json", "w"), indent=2)

    print(f"\nwrote {args.prefix}.json  [{time.time()-t0:.0f}s]")
    plot(args, out)


def _FakeLM(sustain=True):
    """A model-free stand-in for the selftest: its tokenizer is identity over space-joined
    ints, and it 'generates' by echoing its context (sustain) or emptying it (die)."""
    class Tok:
        def decode(self, ids): return " ".join(str(int(i)) for i in ids)
        def encode(self, s): return [int(x) for x in s.split()] if s.strip() else []
    class LM:
        tokenizer = Tok()
        def continue_ids(self, ctx, T, n=1, temperature=1.0):
            return [list(ctx)[:T] if sustain else []]
    return LM()


def selftest():
    assert count_payload("a b a b a", "a b") == 2
    assert count_payload("xyz", "a b") == 0
    lm = _FakeLM(sustain=True)
    S = [7, 8]; ptext = lm.tokenizer.decode(S)          # "7 8"
    ks = passage_chain([lm] * 4, S, ptext, host0_reps=5, T=50, n_hops=3,
                       rng=np.random.default_rng(0), temperature=1.0)
    assert len(ks) == 4, ks                              # patient zero + 3 hops
    assert ks[0] >= 1, ks                                # payload survives with an echoing host
    dead = _FakeLM(sustain=False)
    ks0 = passage_chain([dead] * 3, S, ptext, host0_reps=5, T=50, n_hops=2,
                        rng=np.random.default_rng(0), temperature=1.0)
    assert ks0 == [0, 0, 0], ks0                         # empty generations transmit nothing
    print("selftest: count_payload + passage_chain plumbing OK; ks(sustain)=", ks)


def plot(args, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    serial = {m: c for m, c in out.get("serial", {}).items() if "error" not in c}
    cross = out.get("cross", {})
    n = (1 if serial else 0) + (1 if cross else 0)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.8))
    axes = np.atleast_1d(axes)
    ax_i = 0
    if serial:
        ax = axes[ax_i]; ax_i += 1
        for m, c in serial.items():
            ps = sorted(c, key=int)
            surv = [c[p]["survive"] for p in ps]
            ax.plot([int(p) for p in ps], surv, "o-", lw=1.9, label=m.split("/")[-1])
        ax.set_xlabel("payload length $p$ (tokens)"); ax.set_ylabel("survival at last hop")
        ax.set_title("E6-A: worm survival vs payload length, at scale\n(short spans endemic, long die)")
        ax.set_ylim(-0.02, 1.04); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    if cross:
        ax = axes[ax_i]
        fams = list(cross)
        M = np.array([[cross[X].get(Y, np.nan) for Y in fams] for X in fams], float)
        im = ax.imshow(M, cmap="magma")
        ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams, rotation=45, ha="right")
        ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams)
        ax.set_xlabel("host tokenizer (hops)"); ax.set_ylabel("patient-zero tokenizer")
        ax.set_title("E6-B: cross-tokenizer firebreak\n(surviving load; diagonal = same tokenizer)")
        for i in range(len(fams)):
            for j in range(len(fams)):
                ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center",
                        color="white" if M[i, j] < np.nanmax(M) / 2 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(f"{args.prefix}.png", dpi=130)
    print(f"wrote {args.prefix}.png")


if __name__ == "__main__":
    main()
