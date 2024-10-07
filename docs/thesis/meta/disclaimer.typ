#let disclaimer(details) = {

  set page(numbering: "I")
  counter(page).update(1) 

  let all = (
    (
      text(weight: "bold", details.author.role), 
      (details.author.name, ..details.author.details).join("\n")
    ),
    ..details.examiners.map(examiner => 
      (
        text(weight: "bold", examiner.role),
        examiner.details.join("\n")
      )
    )
  )

  align(top, grid(
    columns: (35mm, 1fr),
    gutter: 2 * details.fontSize,
    ..all.flatten()
  ))

  align(horizon,[
    #set par(justify: true)
      This content is subject to the terms of a Creative Commons Attribution 4.0 
      License Agreement, unless stated otherwise. Please note that this license 
      does not apply to quotations or works that are used based on another
      license. To view the terms of the license, please click on the hyperlink
      provided.

    _#link("https://creativecommons.org/licenses/by/4.0/deed")_
  ])

  align(bottom,[
    #set par(justify: true)  
      I hereby declare on oath that I have written the submitted Master's thesis
      independently and without outside help, that I have not used any sources or aids other
      than those I have specified and that I have marked the passages taken, either
      literally or in terms of content, from the sources as such.

    #v(15mm)
    #grid(
        columns: 2,
        gutter: 1fr,
        "Hannover, October 9, 2024" , details.author.name
    ) 
  ])
}