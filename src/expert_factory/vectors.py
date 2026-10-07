"""TF-IDF vectors, k-means topic clustering and lexical retrieval (NumPy only)."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from expert_factory.text import content_words


class TfidfModel:
    """Sublinear TF-IDF with L2-normalised rows."""

    def __init__(self, documents: Sequence[str], max_features: int = 5000) -> None:
        tokenised = [content_words(d) for d in documents]
        df: Counter[str] = Counter()
        for toks in tokenised:
            df.update(set(toks))
        vocab = [w for w, _ in df.most_common(max_features)]
        self.vocab = {w: i for i, w in enumerate(vocab)}
        self.terms = vocab
        n = max(len(documents), 1)
        self.idf = np.array([math.log((1 + n) / (1 + df[w])) + 1.0 for w in vocab])
        self.matrix = self.transform_tokens(tokenised)

    def transform_tokens(self, tokenised: Sequence[Sequence[str]]) -> np.ndarray:
        mat = np.zeros((len(tokenised), len(self.vocab)), dtype=np.float64)
        for row, toks in enumerate(tokenised):
            for word, count in Counter(toks).items():
                col = self.vocab.get(word)
                if col is not None:
                    mat[row, col] = (1.0 + math.log(count)) * self.idf[col]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        out: np.ndarray = mat / np.where(norms == 0, 1.0, norms)
        return out

    def transform(self, documents: Sequence[str]) -> np.ndarray:
        return self.transform_tokens([content_words(d) for d in documents])

    def top_terms(self, vector: np.ndarray, k: int) -> list[str]:
        order = np.argsort(-vector)[:k]
        return [self.terms[i] for i in order if vector[i] > 0]


def kmeans(x: np.ndarray, k: int, seed: int, iterations: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Spherical k-means with k-means++ seeding. Returns (labels, centroids)."""
    n = x.shape[0]
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centroids = np.empty((k, x.shape[1]))
    centroids[0] = x[rng.integers(n)]
    for i in range(1, k):
        sims = x @ centroids[:i].T
        dist = np.clip(1.0 - sims.max(axis=1), 0.0, None)
        total = dist.sum()
        idx = rng.integers(n) if total <= 0 else rng.choice(n, p=dist / total)
        centroids[i] = x[idx]
    labels = np.zeros(n, dtype=int)
    for _ in range(iterations):
        new_labels = np.argmax(x @ centroids.T, axis=1)
        for c in range(k):
            members = x[new_labels == c]
            if len(members) == 0:  # re-seed an empty cluster with the worst-fit point
                worst = int(np.argmin((x * centroids[new_labels]).sum(axis=1)))
                centroids[c] = x[worst]
                new_labels[worst] = c
                continue
            mean = members.mean(axis=0)
            norm = np.linalg.norm(mean)
            centroids[c] = mean / norm if norm > 0 else mean
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return labels, centroids


class Retriever:
    """Cosine top-k over TF-IDF vectors of a fixed passage list."""

    def __init__(self, passages: Sequence[str]) -> None:
        self.model = TfidfModel(passages)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self.model.transform([query])[0]
        scores = self.model.matrix @ q
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]
