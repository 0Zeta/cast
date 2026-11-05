// The method illustration retains its selected candidate interval identities.
#import "../shared/styles/illustration.typ" as illustration

#let ink = illustration.ink
#let soft = illustration.soft
#let line-gray = illustration.line-gray
#let track-bg = illustration.track-bg
#let blue = illustration.blue
#let sky = illustration.sky
#let verm = illustration.verm
#let orange = illustration.orange
#let candidate-color = illustration.candidate-color

#let box-radius = illustration.box-radius
#let type-title = illustration.type-title
#let type-body = illustration.type-body
#let type-compact = illustration.type-compact
#let math-font = illustration.math-font
#let title-style = illustration.title-style
#let symbol-style = illustration.symbol-style
#let label = illustration.label

#let panel-height = illustration.panel-height
#let small-card-width = illustration.small-card-width
#let integrated-label-width = illustration.integrated-label-width
#let interval-width = illustration.interval-width
#let clip-frame-count = illustration.clip-frame-count
#let input-card-width = illustration.input-card-width
#let encoder-content-width = illustration.encoder-content-width
#let time-pad = illustration.time-pad
#let band-fill = illustration.band-fill
#let standard-panel-fill = illustration.standard-panel-fill

// Candidate lane 1, interval 3; candidate lane 3, interval 4.
// All temporal panels use time-pad for identical endpoint geometry.
#let comparison-option = illustration.comparison-option
#let upper-candidate-id = if illustration.comparison-config == none { (0, 2) } else {
  illustration.comparison-config.ids.at(if comparison-option == "selected" { 0 } else { 1 })
}
#let lower-candidate-id = if illustration.comparison-config == none { (2, 3) } else { illustration.comparison-config.ids.at(2) }
#let band-a = illustration.candidate-interval(..upper-candidate-id)
#let band-b = illustration.candidate-interval(..lower-candidate-id)
#let forgery-bands = (band-a, band-b)
#let retained-candidate-ids = if comparison-option == "current" {
  ((0, 2), (1, 3), (2, 3), (0, 7))
} else {
  illustration.comparison-config.ids
}
#let removal-iou-threshold = if comparison-option == "current" { .5 } else { .4 }
#let candidate-proposal(lane, index) = {
  let (a, b) = illustration.candidate-interval(lane, index)
  (a, b, candidate-color(lane, index))
}
#let proposal-intervals = retained-candidate-ids.map(((lane, index)) => candidate-proposal(lane, index))
#if comparison-option != "current" {
  for proposal in proposal-intervals {
    assert(proposal.at(0) >= .085 and proposal.at(1) <= .925)
  }
}
#for i in range(1, proposal-intervals.len()) {
  assert(proposal-intervals.at(i).at(0) > proposal-intervals.at(i - 1).at(1))
}
#let interval-iou(a, b) = {
  let intersection = calc.max(0, calc.min(a.at(1), b.at(1)) - calc.max(a.at(0), b.at(0)))
  intersection / (a.at(1) - a.at(0) + b.at(1) - b.at(0) - intersection)
}
#let removed-proposals = illustration.candidate-divisions.enumerate().map(((lane, count)) => {
  range(count).filter(index => not retained-candidate-ids.contains((lane, index))).map(index => {
    let proposal = candidate-proposal(lane, index)
    let owners = proposal-intervals.enumerate().filter(((i, retained)) => interval-iou(proposal, retained) >= removal-iou-threshold)
    if owners.len() > 0 { ((proposal: proposal, owner: owners.first().at(0), lane: lane),) } else { () }
  }).flatten()
}).flatten()

// Stage labels live inside their components; there is no external header row.
#let method-arrow-width = 2.4mm
#let stage-gap = method-arrow-width + .6mm
#let figure-height = panel-height
