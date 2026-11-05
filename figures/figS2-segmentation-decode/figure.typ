#import "@preview/cetz:0.5.2": canvas, draw
#let spec = json(sys.inputs.at("spec"))
#let ink = rgb(spec.colors.ink)
#let rule = rgb(spec.colors.line)
#let red = rgb(spec.colors.ground_truth)
#let blue = rgb(spec.colors.proposal)
#set page(width: 174.625mm, height: auto, margin: 3mm)
#set text(font: spec.fonts.text, size: 8.3pt, fill: ink, fallback: false)
#set par(spacing: 0pt)
#canvas(length: 1mm, {
  let w = 144
  let h = 39
  let x(t) = (t - spec.window.at(0)) / (spec.window.at(1) - spec.window.at(0)) * w
  let y(score) = (score - spec.ymin) / (spec.ymax - spec.ymin) * h
  draw.rect((-21, -19), (w + 2, h + 7), stroke: none, fill: none)
  draw.content((0, h + 4), anchor: "west", [Retained candidates (at most #spec.top)])
  draw.line((0, h), (0, 0), (w, 0), stroke: (paint: rule, thickness: .5pt))
  for value in spec.yticks {
    draw.content((-2, y(value)), anchor: "east", str(calc.round(value, digits: 2)))
  }
  draw.content((-17, h / 2), angle: 90deg, [Candidate score])
  for item in spec.candidates {
    draw.line((x(item.start), y(item.score)), (x(item.end), y(item.score)),
      stroke: (paint: ink.lighten(45%), thickness: .7pt))
  }
  for (items, height, color, label) in (
    (spec.gt, -5, red, [Ground truth]),
    (spec.decoded.map(d => (d.start_s, d.end_s)), -9, blue, [Penalized decode]),
  ) {
    draw.content((-2, height), anchor: "east", text(fill: color, label))
    for interval in items {
      let start = calc.max(interval.at(0), spec.window.at(0))
      let end = calc.min(interval.at(1), spec.window.at(1))
      if start < end {
        draw.line((x(start), height), (x(end), height), stroke: (paint: color, thickness: 2pt))
      }
    }
  }
  for tick in spec.ticks {
    draw.line((tick.x * w, 0), (tick.x * w, -.7), stroke: (paint: rule, thickness: .5pt))
    draw.content((tick.x * w, -14), anchor: "north", tick.label)
  }
})
