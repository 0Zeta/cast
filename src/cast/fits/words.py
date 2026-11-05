"""Load aligned words and classify content words."""


from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Final

from ..artifacts import FPS, SchemaError

# Alignment tier labels that are not lexical words.
NON_LEXICAL_LABELS: Final[frozenset[str]] = frozenset(
    {"", "<eps>", "sil", "sp", "spn", "<unk>"})

# English function words: determiners, pronouns, prepositions,
# conjunctions, auxiliaries, particles, discourse items and contractions.
# 266 entries, sorted (spec Appendix C).

CLOSED_CLASS_INVENTORY: Final[tuple[str, ...]] = (
    "a", "about", "above", "across", "actually", "after", "again",
    "against", "ah", "all", "along", "also", "although", "always",
    "am", "among", "an", "and", "another", "any", "anybody", "anyone",
    "anything", "are", "aren't", "around", "as", "at", "be", "because",
    "been", "before", "behind", "being", "below", "beside", "besides",
    "between", "beyond", "both", "but", "by", "can", "can't", "cannot",
    "could", "couldn't", "dare", "despite", "did", "didn't", "do",
    "does", "doesn't", "doing", "don't", "down", "during", "each",
    "eh", "either", "er", "even", "ever", "every", "everybody",
    "everyone", "everything", "for", "from", "get", "got", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he",
    "he'd", "he'll", "he's", "hello", "her", "here", "here's", "hers",
    "herself", "hi", "him", "himself", "his", "hmm", "however", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "indeed", "into", "is",
    "isn't", "it", "it'll", "it's", "its", "itself", "just", "least",
    "less", "lest", "let", "let's", "like", "may", "me", "might",
    "mine", "mm", "more", "most", "much", "must", "mustn't", "my",
    "myself", "n't", "near", "need", "needn't", "neither", "never",
    "no", "nobody", "nope", "nor", "not", "nothing", "now", "of",
    "off", "oh", "ok", "okay", "on", "once", "one", "ones", "only",
    "onto", "or", "ought", "our", "ours", "ourselves", "out", "over",
    "per", "please", "quite", "rather", "really", "shall", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "since",
    "so", "some", "somebody", "someone", "something", "still", "such",
    "than", "thank", "thanks", "that", "that's", "the", "their",
    "theirs", "them", "themselves", "then", "there", "there's",
    "therefore", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "though", "through", "throughout",
    "thus", "to", "too", "toward", "towards", "uh", "um", "under",
    "unless", "unlike", "until", "up", "upon", "us", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "well", "were",
    "weren't", "what", "what's", "when", "whenever", "where",
    "whereas", "wherever", "whether", "which", "while", "who", "who's",
    "whom", "whose", "will", "with", "within", "without", "won't",
    "would", "wouldn't", "yeah", "yep", "yes", "yet", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves")

_CLOSED_CLASS: Final[frozenset[str]] = frozenset(CLOSED_CLASS_INVENTORY)

# The clip-level alignment-reliability filter.
MIN_CONTENT_WORDS: Final[int] = 5

# Word interval: (start_s, end_s, word), sorted by start within a clip.
WordIntervals = dict[str, list[tuple[float, float, str]]]

# Columns the alignment CSV must declare.
WORD_CSV_COLUMNS: Final[tuple[str, ...]] = ("video_id", "word",
                                            "start_s", "end_s")




def is_content_word(word: str) -> bool:
    """Check whether a normalized token is an open-class word."""
    w = word.strip().lower()
    return w not in NON_LEXICAL_LABELS and w not in _CLOSED_CLASS


def load_word_intervals(csv_path: str | Path) -> WordIntervals:
    """Load and validate sorted lexical-word intervals."""
    by_clip: WordIntervals = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or ())
        missing = [c for c in WORD_CSV_COLUMNS if c not in fields]
        if missing:
            raise SchemaError(
                f"{csv_path}: word alignment csv is missing columns "
                f"{missing}; it declares {fields}")
        for r in reader:
            w = (r.get("word") or "").strip().lower()
            if w in NON_LEXICAL_LABELS:
                continue
            vid = (r.get("video_id") or "").strip()
            if not vid:
                raise SchemaError(
                    f"{csv_path}: word alignment row with an empty "
                    f"video_id would open a clip of its own")
            try:
                s, e = float(r["start_s"]), float(r["end_s"])
            except (TypeError, ValueError):
                continue
            if not (e > s) or not math.isfinite(s) or not math.isfinite(e):
                continue
            by_clip.setdefault(vid, []).append((s, e, w))
    for v in by_clip.values():
        v.sort(key=lambda t: t[0])
    return by_clip


def content_word_count(words: list[tuple[float, float, str]]) -> int:
    return sum(1 for _, _, w in words if is_content_word(w))


def reliable_clips(by_clip: WordIntervals,
                   min_content_words: int = MIN_CONTENT_WORDS) -> list[str]:
    """Clips passing the alignment-reliability filter, sorted."""
    return sorted(v for v, w in by_clip.items()
                  if content_word_count(w) >= min_content_words)


def content_word_duration_frames(words: list[tuple[float, float, str]],
                                 ) -> list[float]:
    return [(e - s) * FPS for s, e, w in words if is_content_word(w)]
