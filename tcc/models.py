"""
Minimal model wrapper for the open-weights tier (HuggingFace generation). This is the
torch-dependent module; `payloads` and `scoring` stay torch-free so they can be unit
tested without a GPU.

For the generation experiments (E1, E5, E6) plain `transformers` generation is enough. The
interpretability tier (E2, E4-lite, E7, E8-lite: induction-head scoring and head ablation)
lives in `tcc.interp`, which wraps a TransformerLens `HookedTransformer`.
"""
from __future__ import annotations
import os

# Let unsupported ops fall back to CPU on Apple-Silicon MPS rather than erroring.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def pick_device(torch):
    """Prefer a real GPU: CUDA, then Apple-Silicon MPS (Metal), then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class LM:
    """A small causal LM with a batched 'continue this context' method."""

    def __init__(self, name, device=None, dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.device = device or pick_device(torch)
        kw = {}
        if dtype is not None:
            kw['torch_dtype'] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(name, **kw).to(self.device).eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def continue_ids(self, input_ids, max_new_tokens, n=1, temperature=1.0, greedy=False):
        """Return `n` independently sampled continuations (lists of token ids) of the
        single context `input_ids`. Greedy with n>1 is degenerate (identical) -- use
        greedy=True with n=1 for the deterministic continuation."""
        torch = self._torch
        with torch.no_grad():
            ids = torch.tensor([input_ids], device=self.device).repeat(n, 1)
            attn = torch.ones_like(ids)            # single context, no padding
            out = self.model.generate(
                ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=not greedy,
                temperature=temperature,
                top_k=0, top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        cont = out[:, len(input_ids):]
        return [row.tolist() for row in cont]
