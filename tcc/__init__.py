"""transformer-context-contagion: empirical test, on real language models, of the
context-contagion conjecture worked out exactly on a toy Markov model in the sibling
project `trust-is-the-attack-surface`.

Submodules are imported explicitly by callers so that `import tcc` stays torch-free:
  tcc.scoring   -- transition-level reproduction metrics (numpy only)
  tcc.payloads  -- out-of-distribution token payload construction (numpy only)
  tcc.models    -- HuggingFace model wrapper (needs torch)
"""
__version__ = "0.0.1"
