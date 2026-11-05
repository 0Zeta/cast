// The teaser uses the shared illustration palette and fonts.
#import "../shared/styles/illustration.typ" as illustration

#let ink = illustration.ink
#let soft = illustration.soft
#let line-gray = illustration.line-gray
#let blue = illustration.blue
#let verm = illustration.verm
#let box-radius = illustration.box-radius
#let type-title = illustration.type-title
#let type-body = illustration.type-body
#let math-font = illustration.math-font
#let text-font = illustration.text-font
#let label = illustration.label
#let symbol-style = illustration.symbol-style

#let forgery-bands = ((2 / 12, 3 / 12), (7 / 12, 9 / 12))
#let band-fill = illustration.band-fill

#let teaser-width = 85.55mm
#let teaser-height = 32.7mm
#let teaser-margin-x = 1.5mm
#let teaser-margin-y = 1.5mm

// The left label lane and right timeline fill the one-column artwork width.
#let timeline-left = 10mm
#let timeline-width = 71.75mm
#let frame-count = 5
#let video-height = timeline-width / frame-count
#let audio-height = 3mm
