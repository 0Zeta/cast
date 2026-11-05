"""Fit duration models and locate auxiliary heads."""


from .duration_law import DurationLawFit, fit_duration_law, log_moments
from .heads import AUXILIARY_HEADS, head_checkpoint
from .pausal_units import (PAUSE_MS, MaxDurationFit,
                           fit_max_duration_frames, segment_pausal_units)
from .records import (SUPERVISION_GENUINE_DERIVED,
                      build_fit_record, duration_law_record,
                      max_duration_record, read_fit_record,
                      write_fit_record)
from .sampling import RECORDED_SAMPLE_SEED, draw_genuine_speech_sample
from .words import (CLOSED_CLASS_INVENTORY, MIN_CONTENT_WORDS,
                    NON_LEXICAL_LABELS,
                    content_word_count, is_content_word,
                    load_word_intervals, reliable_clips)

__all__ = [
    "CLOSED_CLASS_INVENTORY", "NON_LEXICAL_LABELS", "MIN_CONTENT_WORDS",
    "is_content_word", "content_word_count",
    "load_word_intervals", "reliable_clips",
    "DurationLawFit", "fit_duration_law", "log_moments",
    "MaxDurationFit", "fit_max_duration_frames", "segment_pausal_units",
    "PAUSE_MS",
    "AUXILIARY_HEADS", "head_checkpoint",
    "RECORDED_SAMPLE_SEED", "draw_genuine_speech_sample",
    "SUPERVISION_GENUINE_DERIVED",
    "build_fit_record", "write_fit_record",
    "read_fit_record", "duration_law_record", "max_duration_record",
]
