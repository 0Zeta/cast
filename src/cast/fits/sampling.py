"""Sample genuine speech clips for duration fitting."""


from __future__ import annotations

import random
from collections.abc import Iterable, Mapping

from ..speakers import speaker_id

RECORDED_SAMPLE_SEED = 20260811


def draw_genuine_speech_sample(speaker_clips: Mapping[str, list[str]],
                               excluded_speakers: set[str], n: int,
                               seed: int,
                               prior_sample: Iterable[str] = (),
                               ) -> list[str]:
    """Draw one sorted clip per eligible, unexcluded speaker."""
    rng = random.Random(seed)
    eligible = sorted(s for s in speaker_clips
                      if s not in excluded_speakers)
    used = set(prior_sample)
    rng.shuffle(eligible)
    rows: list[str] = []
    for s in eligible:
        if len(rows) >= n:
            break
        vids = [v for v in sorted(speaker_clips[s]) if v not in used]
        if not vids:
            continue
        rows.append(rng.choice(vids))
    bad = {v for v in rows if speaker_id(v) in excluded_speakers}
    if bad:
        raise RuntimeError(
            f"{len(bad)} sampled clips belong to excluded speakers; "
            f"sampling-frame and clip-path speaker IDs disagree")
    return rows
