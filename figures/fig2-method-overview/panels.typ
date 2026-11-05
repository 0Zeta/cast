#import "@preview/cetz:0.5.2": canvas, draw
#import "style.typ": *
#import "components.typ": panel-side-label

#let outline = (paint: line-gray, thickness: .50pt)
#let candidate-stroke = (paint: line-gray, thickness: interval-width, cap: "round")

#let _bounds(w, h) = draw.rect((0, 0), (w, h), stroke: none, fill: none)

// Every stage connector uses the same physical shape.
#let method-flow-arrow() = {
  let width = 2.4
  let height = 1.9
  let outline = (
    (0, .25), (.5, .25), (.5, 0), (1, .5),
    (.5, 1), (.5, .75), (0, .75),
  )
  canvas(length: 1mm, {
    _bounds(width, height)
    draw.line(..outline.map(((x, y)) => (x * width, y * height)),
      close: true, fill: soft, stroke: none)
  })
}

#let connector-inset = .9mm
#let thin-flow-stroke = (paint: soft, thickness: .55pt, cap: "round", join: "round")

// Arrowheads have fixed dimensions in page millimeters, independently of shaft length.
#let thin-flow-arrow(start, end) = {
  let dx = end.at(0) - start.at(0)
  let dy = end.at(1) - start.at(1)
  let distance = calc.sqrt(dx * dx + dy * dy)
  let ux = dx / distance
  let uy = dy / distance
  let bx = end.at(0) - .4 * ux
  let by = end.at(1) - .4 * uy
  draw.line(start, end, stroke: thin-flow-stroke)
  draw.line((bx - .3 * uy, by + .3 * ux), end,
    (bx + .3 * uy, by - .3 * ux), stroke: thin-flow-stroke)
}

#let _card(w, h) = draw.rect(
  (0, 0),
  (w, h),
  radius: box-radius / 1mm,
  stroke: outline,
  fill: standard-panel-fill,
)

// Preserve the candidate identities, endpoints, and colors without adjustment.
#let selection-proposals = proposal-intervals

#let redundancy-removal-panel(width, height) = {
  let w = (integrated-label-width + small-card-width) / 1mm
  let h = height / 1mm
  canvas(length: 1mm, {
    _bounds(w, h)
    _card(w, h)
    panel-side-label([Redundancy Removal], h,
      content-left: (integrated-label-width + time-pad + (small-card-width - 2 * time-pad) / 8 - interval-width / 2) / 1mm)
    let data-left = integrated-label-width / 1mm
    let data-w = small-card-width / 1mm
    let p = time-pad / 1mm
    let ih = h - 2 * p
    let lane-x(lane) = data-left + p + (lane + .5) * (data-w - 2 * p) / 4
    let y(f) = h - p - f * ih

    let retained-x = retained-candidate-ids.map(((lane, index)) => lane-x(lane))
    for (index, removed) in removed-proposals.enumerate() {
      let proposal = removed.proposal
      let owner = removed.owner
      let (a, b, color) = proposal
      let x = lane-x(removed.lane)
      assert(interval-iou(proposal, selection-proposals.at(owner)) >= removal-iou-threshold)
      draw.line((x, y(a)), (x, y(b)), stroke: candidate-stroke)
      draw.content((x, y(a) + 1.5), label([$times$], size: type-body, color: orange))
    }
    for (index, proposal) in selection-proposals.enumerate() {
      let (a, b, color) = proposal
      draw.line(
        (retained-x.at(index), y(a)),
        (retained-x.at(index), y(b)),
        stroke: (paint: color, thickness: interval-width, cap: "round"),
      )
    }
  })
}

#let within-clip-order-panel(width, height) = {
  let w = (integrated-label-width + small-card-width) / 1mm
  let h = height / 1mm
  canvas(length: 1mm, {
    _bounds(w, h)
    _card(w, h)
    panel-side-label([Within-Clip Ordering], h,
      content-left: (integrated-label-width + .75mm + (small-card-width - 1.5mm) / 8 - interval-width / 2) / 1mm)
    let data-left = integrated-label-width / 1mm
    let data-w = small-card-width / 1mm
    let p = time-pad / 1mm
    let ih = h - 2 * p
    let y(f) = h - p - f * ih
    let ordered-proposals = (0, 2, 1, 3).map(index => selection-proposals.at(index))
    for (index, proposal) in ordered-proposals.enumerate() {
      let (a, b, color) = proposal
      let rank = index + 1
      // CSS-like space-around inside an additionally inset content area.
      let side-padding = .75
      let content-width = data-w - 2 * side-padding
      let x = data-left + side-padding + (index + .5) * content-width / 4
      draw.content((x, h - 1.5), label([#rank.], size: type-body), anchor: "north")
      draw.line(
        (x, y(a)),
        (x, y(b)),
        stroke: (paint: color, thickness: interval-width, cap: "round"),
      )
    }
  })
}
