#import "templates/thesis.typ": project
#import "metadata.typ": details
#show link: set text(fill: rgb(0, 0, 255))
#show: body => project(details, body)

= Example with Lorem Ipsum

#lorem(450)

= Introduction
#rect(
  width: 100%,
  radius: 10%,
  stroke: 0.5pt,
  fill: yellow,
)[
  Note: Introduce the topic of your thesis, e.g. with a little historical overview.
]
