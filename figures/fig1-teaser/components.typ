#import "@preview/cetz:0.5.2": canvas, draw
#import "style.typ": *

#let clip-frames = (
  "../shared/assets/frames/frame_030.png",
  "../shared/assets/frames/frame_038.png",
  "../shared/assets/frames/frame_044.png",
  "../shared/assets/frames/frame_048.png",
  "../shared/assets/frames/frame_052.png",
)
#assert.eq(clip-frames.len(), frame-count)

// Reuse the canonical waveform generated from the same source clip.
#let waveform-source = read("../shared/assets/audio_waveform.tex")
#let waveform-matches = waveform-source.matches(regex("[0-9]+/([0-9]+\\.[0-9]+)"))
#let waveform-data = waveform-matches.map(match => float(match.captures.at(0)))
#assert.eq(waveform-data.len(), 144)

#let _bounds(w, h) = draw.rect((0, 0), (w, h), stroke: none, fill: none)

#let _is-forged(f) = forgery-bands.any(band => (
  f >= band.at(0) and f <= band.at(1)
))

#let teaser-canvas(width, height) = {
  let w = width / 1mm
  let h = height / 1mm
  let track-x = timeline-left / 1mm
  let track-w = timeline-width / 1mm
  let video-h = video-height / 1mm
  let audio-h = audio-height / 1mm
  let frame-w = track-w / frame-count

  let video-top = h - 1.0
  let video-bottom = video-top - video-h
  let rail-h = 1.0
  let audio-center = video-bottom - 5.7
  let axis-y = 1.7
  let fake-center = axis-y + .35 * (audio-center - axis-y)
  let label-x = 3.5

  canvas(length: 1mm, {
    _bounds(w, h)

    // A single pair of temporal bands connects video, audio, labels, and axis.
    // This is the structural idea in the supervisor sketch.
    for band in forgery-bands {
      let x0 = track-x + band.at(0) * track-w
      let x1 = track-x + band.at(1) * track-w
      draw.rect(
        (x0, axis-y + .10),
        (x1, video-top),
        fill: band-fill,
        stroke: none,
      )
    }

    // Five large, evenly sampled frames form one continuous filmstrip.
    for (index, asset) in clip-frames.enumerate() {
      let x0 = track-x + index * frame-w
      let x1 = x0 + frame-w
      draw.content(
        (x0, video-bottom + rail-h),
        (x1, video-top - rail-h),
        image(asset, width: 100%, height: 100%, fit: "cover"),
        padding: 0,
      )
    }

    // White cuts keep the five sampled frames legible without breaking the
    // continuous sprocket rails.
    for boundary in range(1, frame-count) {
      let x = track-x + boundary * frame-w
      draw.rect(
        (x - .22, video-bottom + rail-h),
        (x + .22, video-top - rail-h),
        fill: white,
        stroke: none,
      )
    }

    // The stronger image tint sits above the raster, while the pale shared band
    // remains visible in every non-raster lane.
    for band in forgery-bands {
      let x0 = track-x + band.at(0) * track-w
      let x1 = track-x + band.at(1) * track-w
      draw.rect(
        (x0, video-bottom + rail-h),
        (x1, video-top - rail-h),
        fill: verm.transparentize(68%),
        stroke: none,
      )
    }

    // Film rails and holes share the illustration's dark-ink and white styling.
    draw.rect(
      (track-x, video-bottom),
      (track-x + track-w, video-bottom + rail-h),
      fill: ink,
      stroke: none,
    )
    draw.rect(
      (track-x, video-top - rail-h),
      (track-x + track-w, video-top),
      fill: ink,
      stroke: none,
    )
    for frame in range(0, frame-count) {
      for slot in (.125, .375, .625, .875) {
        let x = track-x + (frame + slot) * frame-w
        draw.rect(
          (x - .32, video-bottom + .24),
          (x + .32, video-bottom + .76),
          radius: .08,
          fill: white,
          stroke: none,
        )
        draw.rect(
          (x - .32, video-top - .76),
          (x + .32, video-top - .24),
          radius: .08,
          fill: white,
          stroke: none,
        )
      }
    }
    draw.rect(
      (track-x, video-bottom),
      (track-x + track-w, video-top),
      stroke: (paint: line-gray, thickness: .35pt),
      fill: none,
    )

    // Exact 144-sample waveform; only samples inside the two shared intervals
    // switch from the canonical blue to forgery red.
    for (index, amplitude) in waveform-data.enumerate() {
      let f = (index + .5) / waveform-data.len()
      let x = track-x + f * track-w
      let half = amplitude * audio-h / 2
      let color = if _is-forged(f) { verm } else { blue }
      draw.line(
        (x, audio-center - half),
        (x, audio-center + half),
        stroke: (paint: color, thickness: .48pt, cap: "round"),
      )
    }

    draw.content(
      (label-x, (video-bottom + video-top) / 2),
      label([video], size: type-title),
    )
    draw.content(
      (label-x, audio-center),
      label([audio], size: type-title),
    )

    for band in forgery-bands {
      let center = track-x + (band.at(0) + band.at(1)) * track-w / 2
      draw.content(
        (center, fake-center),
        label([fake], size: type-title, color: verm),
      )
    }

    // Minimal seconds axis: band boundaries are the only ticks because they are
    // the only temporal positions that carry semantic meaning in this teaser.
    let axis-end = track-x + track-w
    draw.line(
      (track-x, axis-y),
      (axis-end, axis-y),
      stroke: (paint: soft, thickness: .58pt, cap: "round"),
    )
    draw.line(
      (axis-end - .72, axis-y + .48),
      (axis-end, axis-y),
      (axis-end - .72, axis-y - .48),
      stroke: (paint: soft, thickness: .58pt, cap: "round", join: "round"),
    )
    for band in forgery-bands {
      for endpoint in band {
        let x = track-x + endpoint * track-w
        draw.line(
          (x, axis-y - .70),
          (x, axis-y + .70),
          stroke: (paint: soft, thickness: .58pt, cap: "round"),
        )
      }
    }
    draw.content(
      (axis-end - .18, axis-y + 1.15),
      symbol-style([$T$]),
      anchor: "south-east",
    )
  })
}
