"""
Unit checks for tcc.corpus (pure Python, no model). Run directly
(`python -m tests.test_corpus`) or under pytest.
"""
import re
import numpy as np
from tcc import corpus


class _Tok:
    """Whitespace tokenizer stand-in with the HF `encode` signature the corpus helpers use."""
    def encode(self, s, add_special_tokens=False):
        return s.split()


def _words(s):
    return re.findall(r"\w+|[^\w\s]", s.lower())


def _has_repeated_ngram(tokens, n):
    seen = set()
    for i in range(n - 1, len(tokens)):
        g = tuple(tokens[i - n + 1:i + 1])
        if g in seen:
            return True
        seen.add(g)
    return False


def test_sentence_bank_is_distinct_and_sized():
    assert len(corpus.SENTENCES) >= 60
    assert len(set(corpus.SENTENCES)) == len(corpus.SENTENCES)


def test_clean_paragraphs_count_and_no_repeats():
    paras = corpus.clean_paragraphs()
    assert len(paras) >= 30, len(paras)
    assert len(set(paras)) == len(paras)
    for p in paras:
        assert not _has_repeated_ngram(_words(p), 3), p


def test_benign_repetitive_really_repeats():
    assert len(corpus.BENIGN_REPETITIVE) >= 6
    for name, t in corpus.BENIGN_REPETITIVE.items():
        assert _has_repeated_ngram(_words(t), 2), name


def test_prose_of_length_matches_and_does_not_repeat():
    tok, rng = _Tok(), np.random.default_rng(3)
    for target in (40, 120, 300):
        text, sents = corpus.prose_of_length(target, tok, rng)
        assert not corpus.has_repeated_sentence(sents)
        n = len(tok.encode(text))
        assert target - 22 <= n <= target + 22, (target, n)      # within one sentence
    text, sents = corpus.prose_of_length(120, tok, rng, insert="THE FACT HERE", insert_at=0.5)
    assert text.count("THE FACT HERE") == 1
    assert sents.index("THE FACT HERE") not in (0, len(sents) - 1)


def test_prose_of_length_refuses_impossible_target():
    tok, rng = _Tok(), np.random.default_rng(0)
    try:
        corpus.prose_of_length(100000, tok, rng)
    except ValueError:
        return
    raise AssertionError("expected ValueError when the bank cannot reach the target")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn(); print(f"  {fn.__name__}: OK")
    print(f"all {len(fns)} tests passed")
