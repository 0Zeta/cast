#import "@preview/cetz:0.5.2": canvas, draw
#let spec = json(sys.inputs.at("spec"))
#let colors = spec.colors
#let ink = rgb(colors.ink)
#let soft = rgb(colors.soft)
#let rule = rgb(colors.line)
#let palette = (colors.scalar, "#d55e00", colors.vector, colors.ink)
#let single = spec.panels.len() == 1
#set page(width: if single { 85.55mm } else { 174.625mm }, height: auto,
          margin: (x: 3mm, y: 2mm), fill: white)
#set text(font: spec.fonts.text, size: 8.3pt, fill: ink, fallback: false)
#set par(leading: .1em, spacing: 0pt)
#let percent(value) = str(calc.round(value * 100, digits: 1)) + "%"
#let chart(panel) = canvas(length: 1mm, {
  let w = 64
  let h = 40
  let x(v) = (v - panel.xmin) / (panel.xmax - panel.xmin) * w
  let y(v) = (v - panel.ymin) / (panel.ymax - panel.ymin) * h
  draw.rect((-11, -12), (w + 2, h + 14), stroke: none, fill: none)
  draw.line((0, h), (0, 0), (w, 0), stroke: (paint: rule, thickness: .65pt))
  for value in panel.yticks {
    draw.line((-.7, y(value)), (w, y(value)),
      stroke: (paint: rule.lighten(55%), thickness: .25pt))
    draw.content((-1.4, y(value)), anchor: "east", text(size: 7.4pt, fill: soft, percent(value)))
  }
  for value in panel.xticks {
    draw.line((x(value), 0), (x(value), -.7), stroke: (paint: rule, thickness: .5pt))
    draw.content((x(value), -1.5), anchor: "north",
      text(size: 7.4pt, fill: soft, if panel.xpercent { percent(value) } else { str(value) }))
  }
  if panel.diagonal {
    let end = calc.min(panel.xmax, panel.ymax)
    draw.line((x(0), y(0)), (x(end), y(end)),
      stroke: (paint: rule, dash: "dashed", thickness: .7pt))
  }
  for (index, series) in panel.series.enumerate() {
    let color = rgb(palette.at(calc.rem(index, palette.len())))
    let stroke = if series.at("dashed", default: false) {
      (paint: color, thickness: 1pt, dash: "dashed")
    } else { (paint: color, thickness: 1pt) }
    draw.line(..series.points.map(p => (x(p.x), y(p.y))), stroke: stroke)
    for p in series.points {
      if p.lo != none {
        draw.line((x(p.x), y(p.lo)), (x(p.x), y(p.hi)), stroke: (paint: color, thickness: .65pt))
        for bound in (p.lo, p.hi) {
          draw.line((x(p.x) - .65, y(bound)), (x(p.x) + .65, y(bound)), stroke: (paint: color, thickness: .65pt))
        }
      }
      if p.lo != none or series.at("markers", default: false) or panel.diagonal and (p.x == .01 or p.x == .05 or p.x == .1) {
        if series.at("dashed", default: false) {
          draw.rect((x(p.x) - .5, y(p.y) - .5), (x(p.x) + .5, y(p.y) + .5),
                    fill: color, stroke: (paint: white, thickness: .4pt))
        } else {
          draw.circle((x(p.x), y(p.y)), radius: .55, fill: color, stroke: (paint: white, thickness: .4pt))
        }
      }
    }
    let ly = h + 9 - index * 3.8
    draw.line((0, ly), (4, ly), stroke: stroke)
    draw.content((5.5, ly), anchor: "west", text(size: 7pt, series.name))
  }
  draw.content((w / 2, -8), panel.xlabel)
  draw.content((-9, h / 2), angle: 90deg, panel.title)
})
#grid(columns: (1fr,) * spec.panels.len(), column-gutter: 5mm,
      ..spec.panels.map(panel => chart(panel)))
#if spec.confidence != none { text(size: 7pt, fill: soft)[Saved #percent(spec.confidence) marginal confidence intervals. ] }
#if spec.note != "" { text(size: 7pt, fill: soft, spec.note) }
