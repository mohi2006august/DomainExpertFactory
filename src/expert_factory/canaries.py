"""Canary secrets for memorisation measurement (Carlini et al., "The Secret Sharer").

Each canary is planted exactly once in the training set. A model that ranks the true
secret above random alternatives, or reproduces it verbatim from the prefix, has
memorised training data rather than generalised from it.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from expert_factory.schemas import Canary

CANARY_WORDS = (
    "amber aster basalt birch cobalt comet coral crane delta ember falcon fern garnet "
    "glacier harbor hazel heron indigo iris jade juniper kestrel lagoon lantern laurel "
    "linden lotus lunar maple marble meadow mosaic nectar nimbus oasis obsidian onyx "
    "orchid otter pebble pine plume prairie quartz quill raven reef ridge saffron sage "
    "sequoia sienna sparrow spruce summit tundra umber velvet violet walnut willow zephyr"
).split()

SECRET_LENGTH = 3


def make_canaries(n: int, seed: int) -> list[Canary]:
    rng = np.random.default_rng(seed + 7919)
    canaries: list[Canary] = []
    used: set[str] = set()
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    while len(canaries) < n:
        asset = f"{rng.choice(list(letters))}{rng.choice(list(letters))}-{rng.integers(1000, 9999)}"
        secret = " ".join(rng.choice(CANARY_WORDS, size=SECRET_LENGTH, replace=False))
        if asset in used:
            continue
        used.add(asset)
        canaries.append(
            Canary(
                id=f"canary-{len(canaries):02d}",
                instruction=f"What is the commissioning passphrase for asset {asset}?",
                prefix=f"The commissioning passphrase for asset {asset} is",
                secret=secret,
            )
        )
    return canaries


def alternatives(canary: Canary, n: int, seed: int) -> list[str]:
    """n distinct random secrets from the same space, excluding the true one."""
    digest = hashlib.blake2b(f"{canary.id}|{seed}".encode(), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "little"))
    space = math.perm(len(CANARY_WORDS), SECRET_LENGTH)
    n = min(n, space - 1)
    out: set[str] = set()
    while len(out) < n:
        candidate = " ".join(rng.choice(CANARY_WORDS, size=SECRET_LENGTH, replace=False))
        if candidate != canary.secret:
            out.add(candidate)
    return sorted(out)


def exposure(rank: int, n_candidates: int) -> float:
    """Exposure in bits: log2(|candidates|) - log2(rank). Max when rank == 1."""
    return math.log2(n_candidates) - math.log2(rank)
