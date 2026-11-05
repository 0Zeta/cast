"""Build and validate self-splice corpora."""


from .builder import (ARTIFACT_COLUMN_TOKENS, MIN_DONOR_FRAMES,
                      SOURCE_ROWS_PER_THIRD, ExtractionRecord, RowScorer,
                      SOURCE_CSV_ROWS, build_self_splice_corpus,
                      source_csv_bytes,
                      source_manifest_indices)
from .operators import (AUDIO_REPLACE_FRACTION, AUDIO_SHIFT_SEED,
                        apply_audio_local, audio_shift_generator,
                        draw_audio_local,
                        elliptical_mouth_mask, filterbank_blend,
                        mouth_blend)
from .placement import (DEPLOYED_PLACEMENT_LAW, PlacementLaw, segment,
                        segment_length)
from .schema import (CORPUS_KEYS, N_SCALAR_COLUMNS, ROW_KINDS,
                     SelfSpliceCorpus, write_corpus_npz)

__all__ = [
    "PlacementLaw", "DEPLOYED_PLACEMENT_LAW", "segment",
    "segment_length",
    "mouth_blend", "filterbank_blend", "elliptical_mouth_mask",
    "draw_audio_local", "apply_audio_local", "AUDIO_SHIFT_SEED",
    "audio_shift_generator",
    "AUDIO_REPLACE_FRACTION",
    "ExtractionRecord", "RowScorer", "build_self_splice_corpus",
    "source_manifest_indices", "source_csv_bytes",
    "SOURCE_CSV_ROWS",
    "SOURCE_ROWS_PER_THIRD", "MIN_DONOR_FRAMES",
    "ARTIFACT_COLUMN_TOKENS",
    "SelfSpliceCorpus", "write_corpus_npz", "ROW_KINDS", "CORPUS_KEYS",
    "N_SCALAR_COLUMNS",
]
