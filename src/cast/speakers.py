"""Derive speaker keys and deterministic speaker folds."""

from __future__ import annotations

import hashlib
import re

_SPEAKER_PATTERN = re.compile(r"^id\d+$")


def speaker_id(video_id: str) -> str:
    for part in reversed(video_id.split("/")):
        if _SPEAKER_PATTERN.fullmatch(part):
            return part
    return video_id


def speaker_hash_split(video_id: str, n_folds: int, salt: str = "") -> int:
    """Return sha1(salt + speaker_id) modulo n_folds."""

    h = hashlib.sha1((salt + speaker_id(video_id)).encode()).hexdigest()
    return int(h, 16) % n_folds
