// Shared palette, fonts, and illustration geometry.
#let tokens = toml("tokens.toml")
#let text-font = tokens.fonts.text
#let ink = rgb(tokens.colors.ink)
#let soft = rgb(tokens.colors.soft)
#let line-gray = rgb(tokens.colors.line)
#let track-bg = rgb(tokens.colors.track_background)
#let blue = rgb(tokens.colors.proposal)
#let sky = rgb("#56b4e9")
#let verm = rgb(tokens.colors.ground_truth)
#let orange = rgb("#d55e00")

// Candidate order: top to bottom in each column, then left to right.
#let comparison-option = "selected"
#let comparison-config = (counts: (10, 9, 7, 4), gap: 2mm,
                          ids: ((1, 1), (0, 3), (3, 2), (1, 7)))
#let candidate-divisions = comparison-config.counts
#let candidate-color(lane, index) = {
  let ordinal = candidate-divisions.slice(0, lane).fold(0, (sum, count) => sum + count) + index
  let position = ordinal / (candidate-divisions.sum() - 1)
  if position < .3 {
    blue.darken(35% * (1 - position / .3))
  } else {
    blue.lighten(78% * (position - .3) / .7)
  }
}

#let panel-height = 50mm
#let small-card-width = 11.5mm
#let integrated-label-width = 2.3mm
// One physical stroke width for every proposal interval, in every orientation.
#let interval-width = 1.75pt
#let clip-frame-count = 7
#let embedding-width = 4mm
#let encoder-funnel-width = 6mm
#let encoder-content-width = encoder-funnel-width + 1.6mm + embedding-width + .8mm + 2mm
#let title-height = 4.2mm
#let symbol-height = 3.2mm
#let stage-gap = 1.6mm
#let time-pad = 1.1mm
// Preserve square, full-height frames and reserve room for the waveform.
#let input-card-width = integrated-label-width + 2 * time-pad + 1.2mm + (panel-height - 2 * time-pad) / clip-frame-count + 4.8mm
#let candidate-gap = if comparison-config == none { 1.6mm } else { comparison-config.gap }
#let candidate-interval(lane, index) = {
  let count = candidate-divisions.at(lane)
  let gap = candidate-gap / (panel-height - 2 * time-pad)
  let duration = (1 - (count - 1) * gap) / count
  let start = index * (duration + gap)
  (start, start + duration)
}
#let box-radius = .75mm
#let method-arrow-height = 1.8mm
#let time-axis-width = 2.8mm
#let time-axis-gap = .6mm

// One three-step typography scale for the complete figure. Components must use
// these tokens instead of introducing near-duplicate point sizes locally.
#let type-title = 7pt
#let type-body = 6.5pt
#let type-compact = 5.5pt
#let math-font = tokens.fonts.math

// One figure-level source of truth for the two temporal forgery bands.
#let band-a = (1 / 12, 3 / 12)
#let band-b = (6 / 12, 10 / 12)
#let forgery-bands = (band-a, band-b)
#let band-fill = verm.transparentize(85%)
#let standard-panel-fill = white.transparentize(30%)

// One proposal dataset shared by Filter, Rescore, and Rank. Later stages may
// change score styling or horizontal rank lane, never temporal endpoints.
#let proposal-intervals = (
  (band-a.at(0), band-a.at(1), blue),
  (band-b.at(0), band-b.at(1), blue),
  (.34, .41, sky),
  (.88, .95, sky),
)

#let title-style(body) = text(
  font: text-font,
  size: type-title,
  weight: "bold",
  fill: ink,
  body,
)

#let symbol-style(body) = text(
  font: math-font,
  size: type-compact,
  fill: soft,
  body,
)

#let label(body, size: type-body, color: ink, weight: "regular") = text(
  font: text-font,
  size: size,
  fill: color,
  weight: weight,
  body,
)

// One stage grammar. The grid owns all placement; a component only receives the
// width that the grid assigned to its cell. Removing or reordering a stage therefore
// requires editing one item in the stage data, not any downstream coordinates.
#let stage(title, symbol, render) = grid(
  columns: (1fr,),
  rows: (title-height, symbol-height, panel-height),
  row-gutter: 0mm,
  align: center + horizon,
  if title == none { [] } else { title-style(title) },
  if symbol == none { [] } else { move(dy: -.25mm, symbol-style(symbol)) },
  layout(size => render(size.width, panel-height)),
)
