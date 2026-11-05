#import "@preview/cetz:0.5.2": canvas, draw

#let spec = json(sys.inputs.at("spec"))
#let style = spec.style
#let width = style.width_mm * 1mm
#let gutter = style.label_width_mm * 1mm
#let gap = style.label_gap_mm * 1mm
#let column-labels = spec.layout == "column"
#let plot-width = if column-labels { width - gutter - gap } else { width }
#let text-size = style.text_pt * 1pt
#let track-height = style.track_height_mm * 1mm
#let ink = rgb(spec.colors.ink)
#let quiet = rgb(spec.colors.soft)
#let rule = rgb(spec.colors.line)
#let truth = rgb(spec.colors.ground_truth)
#let proposal = rgb(spec.colors.proposal)
#let wash = truth.transparentize(91%)

#set page(width: width, height: auto, margin: 0mm, fill: white)
#set text(font: style.font, size: text-size, fill: ink, fallback: false)
#set par(leading: .12em, spacing: 0pt)
#set block(spacing: 0pt)

#let names = (
  cos: [Cross-modal agreement],
  ja_B: [Audio-side AVSR agreement],
  jv_B: [Video-side AVSR agreement],
  sync: [Synchrony residual],
  vart_C: [Visual artifact score],
  aart_C: [Audio artifact score],
  flux: [Full-band flux percentile],
  fused: [Fused frame score],
  marginal: [Marginal inclusion],
)
#let column-names = (
  cos: [Cross-modal#linebreak()agreement],
  ja_B: [Audio-side AVSR#linebreak()agreement],
  jv_B: [Video-side AVSR#linebreak()agreement],
  sync: [Synchrony#linebreak()residual],
  vart_C: [Visual artifact#linebreak()score],
  aart_C: [Audio artifact#linebreak()score],
  flux: [Full-band flux#linebreak()percentile],
  fused: [Fused frame#linebreak()score],
  marginal: [Marginal#linebreak()inclusion],
)

#let row(label, body, label-fill: ink) = {
  if column-labels {
    grid(
      columns: (gutter, plot-width), column-gutter: gap,
      align: (left + horizon, left + horizon),
      text(fill: label-fill, label), body,
    )
  } else {
    if label != [] {
      block(below: .5mm, text(fill: label-fill, label))
    }
    body
  }
}

// Every drawable uses the same edge-to-edge linear time coordinates.
#let track(data, gt, peaks: ()) = {
  let w = plot-width / 1mm
  let h = track-height / 1mm
  canvas(length: 1mm, {
    draw.rect((0, 0), (w, h), fill: white, stroke: none)
    for interval in gt {
      draw.rect((interval.at(0) * w, 0), (interval.at(1) * w, h),
                fill: wash, stroke: none)
    }
    if data.baseline != none and data.baseline >= 0 and data.baseline <= 1 {
      let y = data.baseline * h
      draw.line((0, y), (w, y),
        stroke: (paint: rule, thickness: .45pt, dash: "dashed"))
    }
    draw.line(..data.points.map(p => (p.at(0) * w, p.at(1) * h)),
              stroke: (paint: ink, thickness: .8pt), fill: none)
    for p in peaks {
      draw.circle((p.at(0) * w, p.at(1) * h), radius: .65,
        fill: proposal, stroke: (paint: white, thickness: .5pt))
    }
    draw.line((0, 0), (w, 0), stroke: (paint: rule, thickness: .35pt))
  })
}

#let interval-lane(intervals, fill, height: 3.3mm) = {
  let w = plot-width / 1mm
  let h = height / 1mm
  canvas(length: 1mm, {
    draw.rect((0, 0), (w, h), fill: none, stroke: none)
    draw.line((0, h / 2), (w, h / 2),
              stroke: (paint: rule.lighten(42%), thickness: 2pt))
    for interval in intervals {
      draw.rect((interval.at(0) * w, h / 2 - .85),
                (interval.at(1) * w, h / 2 + .85),
                fill: fill, stroke: none)
    }
  })
}

// The media row reserves clearance so bracket caps sit outside its pixels.
#let modality-mark(gt, forged, height) = {
  let w = plot-width / 1mm
  let h = height / 1mm
  canvas(length: 1mm, {
    draw.rect((0, 0), (w, h), fill: none, stroke: none)
    if forged {
      for interval in gt {
        let a = interval.at(0) * w
        let b = interval.at(1) * w
        let cap = calc.min(1.5, (b - a) / 3)
        let inset = .2
        let stroke = (paint: truth, thickness: 1.05pt)
        draw.line((a + cap, inset), (a, inset), (a, h - inset),
                  (a + cap, h - inset), stroke: stroke, fill: none)
        draw.line((b - cap, inset), (b, inset), (b, h - inset),
                  (b - cap, h - inset), stroke: stroke, fill: none)
      }
    }
  })
}

#let timeline(ticks) = {
  let w = plot-width / 1mm
  canvas(length: 1mm, {
    draw.rect((0, -4), (w, .5), fill: none, stroke: none)
    draw.line((0, 0), (w, 0), stroke: (paint: quiet, thickness: .5pt))
    for tick in ticks {
      let x = tick.x * w
      draw.line((x, 0), (x, -.7), stroke: (paint: quiet, thickness: .5pt))
      let anchor = if tick.x < .04 { "north-west" } else if tick.x > .96 {
        "north-east"
      } else { "north" }
      draw.content((x, -1.15), anchor: anchor,
        text(size: text-size, fill: quiet, tick.label))
    }
  })
}

#let panel(entry) = {
  let data = entry.data
  let asset(path) = read(entry.asset_root + "/" + path, encoding: none)
  let title = entry.at("title", default: "")
  let letter = entry.at("letter", default: "")
  if title != "" {
    block(width: width, below: 1.2mm,
      text(size: 9.5pt, weight: "bold")[
        #if letter != "" { [(#letter) ] }#title
      ])
  }
  for (index, modality) in entry.media.enumerate() {
    let label = if title != "" { [] } else if modality == "video" {
      [Video]
    } else { [Audio] }
    if title == "" and index == 0 and letter != "" {
      label = [#text(weight: "bold")[(#letter)] #label]
    }
    let image-height = plot-width / 9
    let forged = data.modalities.at(modality)
    let clearance = if forged { .6mm } else { 0mm }
    let row-height = image-height + 2 * clearance
    row(label, block(width: plot-width, height: row-height)[
      #context metadata((kind: "media", case: data.case_name, modality: modality,
                        x: here().position().x / 1pt, y: here().position().y / 1pt,
                        width: plot-width / 1pt, height: row-height / 1pt))
      #place(top + left, dy: clearance,
        block(width: plot-width, height: image-height, clip: true)[
          #if modality == "video" {
            grid(columns: (1fr,) * data.frames.len(), column-gutter: 0pt,
              ..data.frames.map(path => image(asset(path), width: image-height,
                height: image-height, fit: "contain")))
          } else {
            image(asset(data.spectrogram), width: plot-width, height: image-height,
                  fit: "stretch")
          }
        ])
      #place(top + left,
        modality-mark(data.gt, forged, row-height))
    ])
    v(.7mm)
  }
  row([Ground truth], interval-lane(data.gt, truth), label-fill: truth)
  if entry.at("decoded", default: false) {
    row([Decoded intervals], interval-lane(data.decoded, rgb(spec.colors.vector)),
        label-fill: rgb(spec.colors.vector))
  }
  for prop in data.proposals.filter(prop => prop.rank <= entry.proposals) {
    let digits = str(calc.round(prop.iou, digits: 2)).split(".")
    let decimals = digits.at(1, default: "")
    let iou = digits.first() + "." + decimals + "0" * (2 - decimals.len())
    let label = box([Proposal#sym.space.nobreak#prop.rank#h(.8mm)#text(size: 7.7pt)[(IoU#sym.space.nobreak#iou)]])
    row(label, interval-lane(((prop.start, prop.end),), proposal),
        label-fill: proposal)
  }
  v(1.2mm)
  for (index, key) in entry.scores.enumerate() {
    let datum = data.tracks.at(key)
    let arrow = if datum.direction == "up" { [↑] } else { [↓] }
    let role = entry.at("score_roles", default: ())
    let name = if column-labels { column-names.at(key) } else { names.at(key) }
    let label = [#name#sym.space.nobreak#arrow]
    if role.len() > index {
      label = [#text(size: 7.3pt, fill: quiet, role.at(index)) #linebreak()#label]
    }
    let peaks = if key == "flux" and entry.proposals > 0 and entry.at("flux_markers", default: false) {
      data.flux_peaks
    } else { () }
    row(label, block[
      #context metadata((kind: "track", case: data.case_name, key: key,
                        x: here().position().x / 1pt, y: here().position().y / 1pt,
                        width: plot-width / 1pt, height: track-height / 1pt))
      #track(datum, data.gt, peaks: peaks)
    ])
    v(.65mm)
  }
  v(.5mm)
  row([], timeline(data.ticks))
}

#for (index, entry) in spec.panels.enumerate() {
  if index > 0 { v(style.at("panel_gap_mm", default: 4) * 1mm) }
  panel(entry)
}
