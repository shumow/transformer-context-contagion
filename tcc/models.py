"""
Minimal model wrapper for the open-weights tier (HuggingFace generation). This is the
torch-dependent module; `payloads` and `scoring` stay torch-free so they can be unit
tested without a GPU.

For E1 (the condensation knee) plain `transformers` generation is enough. The
interpretability tier -- E2's induction-head attribution via activation patching and head
ablation -- will add a TransformerLens `HookedTransformer` path here; left as a stub below.
"""
from __future__ import annotations


class LM:
    """A small causal LM with a batched 'continue this context' method."""

    def __init__(self, name, device=None, dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
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
            out = self.model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=not greedy,
                temperature=temperature,
                top_k=0, top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        cont = out[:, len(input_ids):]
        return [row.tolist() for row in cont]


def load_hooked(name, device=None):
    """E2 stub: load a TransformerLens HookedTransformer for activation patching and
    head ablation (induction-head attribution). Wired in Phase 1's E2."""
    raise NotImplementedError(
        "TransformerLens path is a Phase-1/E2 stub; install transformer_lens and "
        "implement induction-head attribution here.")
