#let project(body) = {

  import "meta/cover.typ": cover
  import "meta/disclaimer.typ": disclaimer
  import "meta/acknowledgement.typ": acknowledgement
  import "meta/metadata.typ": details

  // ==========================================================================
  // SETTINGS

  set document(
    title: details.title,
    author: details.author.name,
  )

  set page(
    paper: "a4",
    margin: 
      if details.doubleSided 
        {(y: 35mm, inside: 35mm, outside: 18mm)}
      else 
        {(x: 25mm, y: 35mm)},
    header-ascent: 10mm,
    footer-descent: 10mm,
  ) 

  set text(
    // font: "Linux Libertine", 
    font: "CMU Serif", 
    size: details.fontSize, 
    lang: details.language
  )

  show link: set text(fill: rgb(0, 0, 255))
  show math.equation: set text(weight: 400)
  set math.equation(numbering: "(1)")

  show heading: set block(below: 1.2em, above: 1.75em)
  show heading.where(level:1): set text(size: 2 * details.fontSize)
  show heading.where(level:2): set text(size: 1.5 * details.fontSize)
  show heading.where(level:3): set text(size: 1.25 * details.fontSize)
  set heading(numbering: "1.1")

  set par(leading: 0.65em, justify: true)

  set cite(style: "alphanumeric")

  // ============================================================================

  cover(details)
  pagebreak()
  pagebreak()
  set page(numbering: "I", number-align: left)
  counter(page).update(1) 

  disclaimer(details)

  acknowledgement()

  include("meta/abstract.typ")

  // ============================================================================
  // TOC

  set par(leading: 1em)

  outline(
    depth: 3,
    indent: 2em
  )

  // ============================================================================
  // TEXT SETTINGS

  set par(leading: 0.65em, justify: true)

  show heading.where(level:1): chapter => [
    #pagebreak()
    #v(35mm)
    #set text(size: 2 * details.fontSize)
    #chapter
  ]

  set page(
    header: locate(loc => {
      let query_before = query(heading.where(level: 1).before(loc), loc)
      let query_after = query(heading.where(level: 1).after(loc), loc)

      if (query_after.len() != 0 and 
          query_after.first().location().page() == loc.page()) {
        return
      }

      if (query_before.last() != 0 and 
          query_before.last().outlined) {[
        #align(center, query_before.last().body)
        #line(length: 100%, stroke: .5pt + black)
      ]}
    }),
    footer: locate(loc => {
      let query = query(heading.where(level: 1).before(loc), loc)

      if query.len() == 0 {
        return
      }

      // last because it orders heading in occuring form begin to loc
      let res = query.last()

      if res.location().page() == loc.page() {
        align(center, counter(page).display("1"))
      } else {[
        #line(length: 100%, stroke: .5pt + black)
        #align(center, counter(page).display("1"))
      ]}
    })
  )

  set page(numbering: "1")
  counter(page).update(1)

  set figure(placement: top)

  body

  include("introduction/motivation.typ")
  include("introduction/goals_and_contributions.typ")
  include("introduction/required_background_knowledge.typ")
  include("introduction/overview.typ")

  include("background/basic_loss_functions.typ")
  include("background/knowledge_distillation.typ")
  include("background/transformer.typ")
  include("background/self_supervised_learning.typ")
  include("background/contrastive_learning.typ")

  include("background/related_work/deep_aligned_representations.typ")
  include("background/related_work/clip.typ")

  include("methodology/tools_and_approach.typ")  
  include("methodology/data_collection_and_preprocessing.typ")

  include("experiments/unimodal_knowledge_distillation.typ")

  include("experiments/transformer_shre.typ")
  include("experiments/differences_to_unimodal_kd.typ")
  include("experiments/tte.typ")

  //include("experiments/region_descriptions.typ")
  //nclude("experiments/itm.typ")
  include("experiments/memory_bank.typ")

  include("experiments/cmli.typ")

  include("experiments/modality_invariant_targets.typ")
  //include("experiments/quantizing_visual_features.typ")
  
  include("experiments/teacher_ablations.typ")
  include("experiments/fine_tuning.typ")

  //include("experiments/ablation_study_data.typ")
  include("experiments/limitations.typ")

  include("experiments/discussion_of_results.typ")

  include("conclusion/conclusion.typ")
  include("conclusion/outlook.typ")
  include("conclusion/impact.typ")

  // Appendix.
  set heading(numbering: "A.1")
  counter(heading).update(0)
  set figure(placement: none)
  include("appendix/hyperparameters.typ")
  include("appendix/pseudocode.typ")
  include("appendix/figures_and_visualizations.typ")
  include("appendix/technical_details.typ")
  bibliography("references.bib")
}
#show: body => project(body)