"""Text normalisation, tokenisation and lexical similarity measures."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence

WORD_RE = re.compile(r"[a-z0-9]+(?:[-'.][a-z0-9]+)*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for
    from further had has have having he her here hers herself him himself his how i if in
    into is it its itself just me more most my myself no nor not now of off on once only or
    other our ours ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would you your
    yours yourself yourselves also may must shall one two three per via within without use
    used using get gets got make makes made does done what's it's i'm we're you're don't
    """.split()
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.sub(r"[ \t]+", " ", text).strip()


def words(text: str) -> list[str]:
    return WORD_RE.findall(normalize(text).lower())


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOPWORDS and len(w) > 1]


def sentences(text: str) -> list[str]:
    """Split prose into sentences; list items and short lines stay whole."""
    out: list[str] = []
    for line in normalize(text).splitlines():
        line = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line).strip()
        if not line or line.startswith("#") or line.startswith("|"):
            continue
        out.extend(s.strip() for s in SENTENCE_RE.split(line) if s.strip())
    return out


def shingles(tokens: Sequence[str], k: int) -> set[str]:
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = content_words(prediction), content_words(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> float:
    pred, ref = words(prediction), words(reference)
    lcs = _lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def longest_common_run(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest contiguous token run shared by a and b."""
    best = 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def coverage(answer: str, sources: Iterable[str]) -> float:
    """Share of the answer's content words that appear in the source passages."""
    answer_words = content_words(answer)
    if not answer_words:
        return 0.0
    source_vocab: set[str] = set()
    for source in sources:
        source_vocab.update(content_words(source))
    return sum(w in source_vocab for w in answer_words) / len(answer_words)


def slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalize(text).lower()).strip("-")
    return slug[:max_len].rstrip("-") or "item"
