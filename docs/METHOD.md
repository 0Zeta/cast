# Method implementation

The paper specifies the method; this page connects its terms to the code.

| Paper component | Implementation |
|---|---|
| Cross-modal agreement; audio-side and video-side AVSR agreement | `cast.extract.scores` |
| Visual and audio artifact scores; vector synchrony residual | `cast.extract.heads` |
| Genuine-reference null models | `cast.scan.nulls` |
| Interval tests, calibration and fusion | `cast.scan.interval_scan`, `calibration`, `fusion` |
| Proposal selection and frame-evidence contrast | `cast.scan.emissions`, `rescoring` |
| Audio boundary cue | `cast.ranking.audio_boundary` |
| Cross-clip proposal ranker | `cast.ranking.ranker`, `chain` |
| Localization and detection metrics | `cast.eval` |

## Interval scan

Five scalar score groups each contribute four statistics; the vector
synchrony residual contributes two, giving 22 channels per interval.
Intervals span at most 127 frames and leave at least 25 safe-complement
frames. `min_interval_frames=25` names that complement constraint, not a
25-frame proposal minimum. The proposal ranking floor is three frames.

Genuine reference pools fit the null models and calibration tables.
Controlled corruptions fit the candidate, frame, and clip combinations.
Within a round, the fitting, calibration, and fusion subdivisions are
separate. The genuine fusion pool is subdivided 2:1:1 by the recorded clip
sampling identity. Speaker hashes and sampling seeds determine the
reference experiment's partitions and draws.

## Selection and ranking

The method applies temporal NMS at IoU 0.5 with a budget of 50 proposals.
It then adds the unscaled inside-minus-two-frame-flank contrast of the fused
frame track.

Ranking removes proposals below three frames and merges identical intervals.
It assigns zero-based within-clip ranks using stable ordering.
The ranker combines rank, adjusted interval score, and audio boundary cue.
It applies no clip-level proposal gate or boundary refinement.

The audio boundary cue comes from full-band positive log-power flux on mono
16-kHz audio:

1. A 512-sample Hann STFT with hop 160 gives a 100-Hz curve.
2. Positive differences of log power are summed over frequency, then converted
   to stable within-clip percentile ranks.
3. The cue averages the maximum percentile within 80 ms of each measurable boundary.

The curve plotted in figures is the **full-band flux percentile**.
Its boundary aggregate is the **audio boundary cue**.

The optional semi-Markov segmentation output is separate from the ranked
proposal list. Its decode and the diagnostic contrast outputs retain their
own roles; neither replaces the adopted ranking emission.
