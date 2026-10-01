"""Deterministic, row-order-free hashing utilities shared by the stages.

This module used to decide which recordings the gesture scan could read
(``build_population``: eight exclusion reasons, first one wins). The annotation
pipeline excludes nothing — every reason became a status or class column in the
catalog (:mod:`seamless_curation.catalog`) — so only the hashing helpers remain.
"""

from __future__ import annotations

from hashlib import blake2b

import pandas as pd


def shard_of(keys: pd.Series, shards: int) -> pd.Series:
    """Deterministic shard assignment by key, independent of row order.

    Restart safety depends on a key always landing in the same shard, so this
    must not use the DataFrame index. The scan shards by ``interaction_key`` so
    both members of a conversation are measured in one task.
    """

    if shards < 1:
        raise ValueError("shards must be >= 1")
    digest = keys.astype(str).map(lambda s: int.from_bytes(_blake(s), "big"))
    return (digest % shards).astype("int32")


def _blake(text: str) -> bytes:
    return blake2b(text.encode("utf-8"), digest_size=8).digest()


def stable_key(*parts: str, length: int = 14) -> str:
    """A short, stable identifier for an ordered tuple of strings (content-derived)."""

    digest = blake2b("\x1f".join(parts).encode("utf-8"), digest_size=8).hexdigest()
    return digest[:length]


def stable_unit_interval(*parts: str, salt: str = "") -> float:
    """A reproducible pseudo-random number in [0, 1) from string parts.

    For seeded sampling that must not depend on row order, numpy version or
    platform (e.g. drawing a QA sample from the annotation tables).
    """

    digest = blake2b("\x1f".join((salt,) + parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)
