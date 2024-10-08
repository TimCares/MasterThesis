#let cover(details) = {
  set page(numbering: none, margin: {(x: 25mm, y: 35mm)})

  image("../figures/wordmark_black.png", width: 20%)

  place(horizon, dy: -20mm,[
    #set text(size: details.fontSize * 2, weight: "bold")
    #details.title
    
    #set text(size: 1.25 * details.fontSize, weight: "regular")

    #details.author.name

    Master's thesis in Applied Computer Science at Faculty IV – Wirtschaft und Informatik Hochschule Hannover
    
    #linebreak()
    #details.date
  ])

  place(bottom + right, dy: 20mm, image("../figures/logo_colour.png", width: 45mm))
}