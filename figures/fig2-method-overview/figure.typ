#import "../shared/styles/illustration.typ": ink, type-body, math-font, text-font
#import "@preview/cetz:0.5.2": canvas, draw
#import "style.typ": *
#import "components.typ": (
  clip-panel, encoder-panel, score-panel, candidate-panel,
  scan-panel, calibration-panel, fusion-plus-glyph,
  scoring-row-y, interval-score-geometry, score-time-y, scan-left-inset,
  scan-row-geometry, calibration-row-geometry, fusion-radius,
)
#import "panels.typ": (
  method-flow-arrow, thin-flow-arrow, thin-flow-stroke, connector-inset,
  redundancy-removal-panel, within-clip-order-panel,
)

// Measured from the Slides PDF, normalized to a 166 mm page.
// Renderers keep native coordinates within the measured stage boxes.
#let sy = 52.063 / 50
#let sx = 1.0195
#let panel-top = .560mm
#let height = 52.063mm
#let stages = (
  (id: "input", left: .520mm, width: 17.662mm, native: input-card-width, render: (w, h) => clip-panel(w, h, bands: forgery-bands)),
  (id: "encoder", left: 21.241mm, width: 14.4mm * sx, native: 14.4mm, render: (w, h) => encoder-panel(w, h, show-family-formulas: false, show-frozen: false)),
  (id: "scores", left: 38.978mm, width: 14.6mm * sx, native: 14.6mm, render: (w, h) => score-panel(w, h, bands: forgery-bands)),
  (id: "candidates", left: 56.918mm, width: 14.069mm, native: 13.8mm, render: candidate-panel),
  (id: "scan", left: 74.043mm, width: 26.466mm, native: 25.9625mm, render: (w, h) => scan-panel(w, h, bands: forgery-bands)),
  (id: "calibration", left: 103.566mm, width: 17.643mm, native: 17.3094mm, render: calibration-panel),
  (id: "fusion", left: 124.260mm, width: 9mm * sx, native: 9mm, render: (w, h) => fusion-plus-glyph(w, h, show-label: true, center-y: 25)),
  (id: "removal", left: 134.129mm, width: 14.069mm, native: 13.8mm, render: redundancy-removal-panel),
  (id: "ordering", left: 151.254mm, width: 14.069mm, native: 13.8mm, render: within-clip-order-panel),
)

#let stage(id) = stages.find(s => s.id == id)
#let page-y(native-y) = .560 + (50 - native-y) * sy
#let page-x(id, native-x) = (stage(id).left + native-x * (stage(id).width / stage(id).native)) / 1mm
#let draw-band(band, x, width) = {
  let h = 50mm - 2 * time-pad
  place(top + left, dx: x,
    dy: panel-top + (time-pad + band.at(0) * h) * sy,
    rect(width: width, height: (band.at(1) - band.at(0)) * h * sy,
      fill: band-fill, stroke: none))
}

#let render-figure() = block(width: 166mm, height: 53.626mm)[
  // Background bands begin after the input card; its media get one foreground overlay.
  #for band in forgery-bands {
    let input-right = stage("input").left + stage("input").width
    draw-band(band, input-right, 165.323mm - input-right)
  }
  #for item in stages {
    place(top + left, dx: item.left, dy: panel-top,
      scale(x: item.width / item.native * 100%, y: sy * 100%, reflow: true,
        block(width: item.native, height: 50mm)[
          #place(center + horizon, (item.render)(item.native, 50mm))
        ]))
  }
  #for band in forgery-bands {
    draw-band(band, stage("input").left, stage("input").width)
  }
  #place(top + left, canvas(length: 1mm, {
    draw.rect((0, 0), (166, 53.626), fill: none, stroke: none)
    // Page coordinates below use y increasing upward, as CeTZ expects.
    let y(native-y) = 53.626 - page-y(native-y)
    let mid = 53.626 - (.560 + 52.063 / 2)
    for (a, b) in (("input", "encoder"), ("encoder", "scores"),
                   ("scores", "candidates"), ("candidates", "scan"),
                   ("fusion", "removal"), ("removal", "ordering")) {
      let before = stage(a)
      let after = stage(b)
      let right = if a == "fusion" { 131.100mm } else { before.left + before.width }
      let cx = (right + after.left) / 2 / 1mm
      draw.content((cx, mid), method-flow-arrow())
    }
    let scan = stage("scan")
    let cal = stage("calibration")
    let scan-row = scan-row-geometry(scan.native)
    let cal-row = calibration-row-geometry(cal.native)
    let inset = connector-inset / 1mm
    // The four calibrated p-values join a shared orthogonal bus in Slides.
    let junction = 123.800
    let rows = range(4).map(i => y(scoring-row-y(50mm, i)))
    for row in range(4) {
      let row-y = rows.at(row)
      thin-flow-arrow((page-x("scan", scan-row.arrow-start), row-y),
        (page-x("scan", scan-row.arrow-end), row-y))
      thin-flow-arrow(((scan.left + scan.width) / 1mm - inset, row-y),
        (cal.left / 1mm + inset, row-y))
      thin-flow-arrow((page-x("calibration", cal-row.arrow-start), row-y),
        (page-x("calibration", cal-row.arrow-end), row-y))
      draw.line(((cal.left + cal.width) / 1mm - inset, row-y), (junction, row-y),
        stroke: thin-flow-stroke)
    }
    draw.line((junction, rows.first()), (junction, rows.last()), stroke: thin-flow-stroke)
    let fusion-left = page-x("fusion", stage("fusion").native / 2 - fusion-radius)
    thin-flow-arrow((junction, mid), (fusion-left - .25, mid))
    // Keep the zoom attached to the same interval in the illustrative score.
    let excerpt = interval-score-geometry(50mm, forgery-bands)
    let origin-x = 38.978 + (integrated-label-width / 1mm + small-card-width / 1mm / 2) * sx
    let zoom-left = 74.043 + (scan-left-inset / 1mm - 1.48) * sx
    for (fraction, corner-y) in ((excerpt.lo, excerpt.top), (excerpt.hi, excerpt.bottom)) {
      draw.line((origin-x, y(score-time-y(50mm, fraction))),
        (zoom-left + .22, y(corner-y)), stroke: (paint: line-gray, thickness: .4pt))
    }
  }))
]

#set page(width: 166mm, height: 53.626mm, margin: 0mm, fill: white)
#set text(font: text-font, size: type-body, fill: ink, fallback: false)
#show math.equation: set text(font: math-font)
#set par(justify: false)

#context render-figure()
