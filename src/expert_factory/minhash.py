"""MinHash signatures and LSH banding for near-duplicate and leakage detection."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Hashable, Iterable

import numpy as np

_MERSENNE_PRIME = np.uint64((1 << 61) - 1)
_MAX_HASH = (1 << 31) - 1


def _hash31(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & _MAX_HASH


class MinHasher:
    """Universal-hash MinHash: sig_i = min_s ((a_i * h(s) + b_i) mod p)."""

    def __init__(self, num_perm: int = 128, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        # a, h < 2^31 keeps a*h + b below 2^63, so uint64 arithmetic never overflows.
        self._a = rng.integers(1, _MAX_HASH, size=num_perm, dtype=np.uint64)
        self._b = rng.integers(0, _MAX_HASH, size=num_perm, dtype=np.uint64)

    def signature(self, items: Iterable[str]) -> np.ndarray:
        hashes = np.fromiter((_hash31(s) for s in set(items)), dtype=np.uint64)
        if hashes.size == 0:
            return np.full(self.num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        permuted = (np.outer(hashes, self._a) + self._b) % _MERSENNE_PRIME
        sig: np.ndarray = permuted.min(axis=0)
        return sig

    @staticmethod
    def estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        return float(np.mean(sig_a == sig_b))


def choose_bands(num_perm: int, threshold: float) -> tuple[int, int]:
    """Pick (bands, rows) whose S-curve midpoint (1/b)^(1/r) sits just under threshold."""
    best: tuple[float, int, int] | None = None
    for rows in range(1, num_perm + 1):
        if num_perm % rows:
            continue
        bands = num_perm // rows
        midpoint = (1.0 / bands) ** (1.0 / rows)
        # Prefer midpoints slightly below the threshold: more candidates, exact check after.
        penalty = abs(midpoint - (threshold - 0.05))
        if best is None or penalty < best[0]:
            best = (penalty, bands, rows)
    assert best is not None
    return best[1], best[2]


class LSHIndex:
    """Banded LSH over MinHash signatures. Query returns candidate keys only."""

    def __init__(self, num_perm: int, threshold: float) -> None:
        self.bands, self.rows = choose_bands(num_perm, threshold)
        self._buckets: list[dict[bytes, list[Hashable]]] = [
            defaultdict(list) for _ in range(self.bands)
        ]

    def _band_keys(self, sig: np.ndarray) -> list[bytes]:
        return [sig[i * self.rows : (i + 1) * self.rows].tobytes() for i in range(self.bands)]

    def insert(self, key: Hashable, sig: np.ndarray) -> None:
        for bucket, band in zip(self._buckets, self._band_keys(sig), strict=True):
            bucket[band].append(key)

    def query(self, sig: np.ndarray) -> set[Hashable]:
        found: set[Hashable] = set()
        for bucket, band in zip(self._buckets, self._band_keys(sig), strict=True):
            found.update(bucket.get(band, ()))
        return found
