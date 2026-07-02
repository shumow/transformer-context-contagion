"""transformer-context-contagion: empirical test, on real language models, of the
context-contagion conjecture worked out exactly on a toy Markov model in the sibling
project `trust-is-the-attack-surface`.

Submodules are imported explicitly by callers so that `import tcc` stays torch-free:
  tcc.scoring   -- transition-level reproduction metrics (numpy only)
  tcc.payloads  -- out-of-distribution token payload construction (numpy only)
  tcc.models    -- HuggingFace model wrapper (needs torch)
"""
__version__ = "0.0.1"

# Some GPU images (RunPod) set HF_HUB_ENABLE_HF_TRANSFER=1 but don't ship the hf_transfer
# package, which makes every HuggingFace download raise. Every experiment imports `tcc`
# before loading a model, so disable the flag here if the package is missing. (If it IS
# installed, we leave it on -- it's the faster downloader.)
import os as _os
if _os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
    try:
        import hf_transfer as _hf_transfer  # noqa: F401
    except Exception:
        _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
