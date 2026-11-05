"""Label self-splice proposals by assignment to synthetic segments."""

from __future__ import annotations

import numpy as np

from ..artifacts import FPS
from ..corpus.schema import SelfSpliceCorpus
from ..eval.matching import greedy_assign, iou_matrix
from .chain import PreparedProposals

ASSIGNMENT_IOU = 0.5


def synthetic_segment_labels(prepared: PreparedProposals,
                             corpus: SelfSpliceCorpus,
                             iou_threshold: float = ASSIGNMENT_IOU,
                             ) -> np.ndarray:
    """True for proposals claimed by a synthetic segment of their row.

    Rows are matched by position: ``prepared.clip_index`` indexes the corpus.
    Within a row, segments claim proposals in score order with the official
    evaluator's greedy rule at the float32 seconds-domain IoU threshold."""
    start_s = (prepared.start_frame / FPS).astype(np.float32)
    end_s = ((prepared.end_frame + 1) / FPS).astype(np.float32)
    is_tp = np.zeros(len(prepared), dtype=bool)
    blocks = np.flatnonzero(np.r_[True, prepared.clip_index[1:]
                                  != prepared.clip_index[:-1]])
    bounds = np.r_[blocks, len(prepared)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        segments = corpus.row_segments(int(prepared.clip_index[a]))
        if not len(segments):
            continue
        ious = iou_matrix(start_s[a:b], end_s[a:b],
                          segments[:, 0] / FPS, (segments[:, 1] + 1) / FPS)
        is_tp[a:b] = greedy_assign(ious, iou_threshold)
    return is_tp
