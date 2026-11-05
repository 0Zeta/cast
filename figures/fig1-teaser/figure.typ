#import "style.typ": *
#import "components.typ": teaser-canvas

#set page(
  width: teaser-width,
  height: teaser-height,
  margin: (x: teaser-margin-x, y: teaser-margin-y),
  fill: white,
)
#set text(font: text-font, size: type-body, fill: ink, fallback: false)
#show math.equation: set text(font: math-font)
#set par(justify: false)

#layout(size => teaser-canvas(size.width, size.height))
