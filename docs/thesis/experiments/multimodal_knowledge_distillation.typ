== Multimodal Knowledge Distillation
== Seperate Self-Attention

=== Baseline

#figure(
  table(
  columns: (25%, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: none,
    table.hline(),
    table.header(
      table.cell(rowspan: 3, colspan: 1, align:horizon, [*Model*]),
      table.cell(colspan: 6, [*MSCOCO (5K test set)*]),
      table.cell(colspan: 6, [*Flickr30K (1K test set)*]),
      table.cell(colspan: 3, [Image $arrow.r$ Text]),
      table.cell(colspan: 3, [Text $arrow.r$ Image]),
      table.vline(stroke: .4pt),
      table.cell(colspan: 3, [Image $arrow.r$ Text]),
      table.cell(colspan: 3, [Text $arrow.r$ Image]),
      table.hline(start: 1, end: 4, stroke: .2pt),
      table.hline(start: 4, end: 7, stroke: .2pt),
      table.hline(start: 7, end: 10, stroke: .2pt),
      table.hline(start: 10, end: 13, stroke: .2pt),
      [R@1], [R@5], [R@10], [R@1], [R@5], [R@10], [R@1], [R@5], [R@10], [R@1], [R@5], [R@10]
    ),
    table.hline(stroke: .4pt),
    [FLAVA], [42.74], [76.76], [-], [38.38], [67.47], [-], [67.7], [94.0], [-], [65.22], [89.38], [-],
    [Data2Vec2], [0.02], [0.08], [0.22], [0.01], [0.10], [0.19], [0.02], [0.12], [0.26], [0.02], [0.06], [0.12],
    [*MM-D2V2 (Ours)*], [4.24], [12.12], [17.96], [1.77], [6.54], [10.91], [1.2], [4.88], [8.18], [0.54], [2.52], [4.58],
    [*MM-D2V2 (Ours)†*], [31.72], [56.78], [67.9], [12.42], [31.05], [42.5], [7.7], [26.18], [37.6], [4.08], [17.01], [24.26],
    table.hline(),
  ),
  caption: [Comparison of Zero-shot Image-Text and Text-Image Retrieval of first results with FLAVA and Data2Vec2 papers. Because Data2Vec2 is a unimodal model, we embed each image with the D2V2-Image model and each text with the D2V2-Text model. This yields unusable results, as there has been no incentive for the models to learn a shared representation, as both are unimodal. This is why we had to use both the image and the text model to embed the data. \ *†*: This version has been trained with BEiT-2 as the teacher model, not the D2V2 Image model.],
)<image_text_retrieval_1>


#figure(
  table(
  columns: (25%, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: none,
    table.hline(),
    table.header(
      table.cell(rowspan: 3, colspan: 1, align:horizon, [*Model*]),
      table.cell(colspan: 6, [*MSCOCO (5K test set)*]),
      table.cell(colspan: 6, [*Flickr30K (1K test set)*]),
      table.cell(colspan: 3, [Image $arrow.r$ Text]),
      table.cell(colspan: 3, [Text $arrow.r$ Image]),
      table.vline(stroke: .4pt),
      table.cell(colspan: 3, [Image $arrow.r$ Text]),
      table.cell(colspan: 3, [Text $arrow.r$ Image]),
      table.hline(start: 1, end: 4, stroke: .2pt),
      table.hline(start: 4, end: 7, stroke: .2pt),
      table.hline(start: 7, end: 10, stroke: .2pt),
      table.hline(start: 10, end: 13, stroke: .2pt),
      [R@1], [R@5], [R@10], [R@1], [R@5], [R@10], [R@1], [R@5], [R@10], [R@1], [R@5], [R@10]
    ),
    table.hline(stroke: .4pt),
    table.cell([_Zero-Shot_], align: left), table.cell(colspan: 12, []),
    [FLAVA], [42.74], [76.76], [-], [38.38], [67.47], [-], [67.7], [94.0], [-], [65.22], [89.38], [-],
    [CLIP], [58.4], [81.5], [88.1], [37.8], [62.4], [72.2], [88.0],[98.7], [99.4], [68.7], [90.6], [95.2],
    [*MM-D2V2 (Ours)*], [31.72], [56.78], [67.9], [12.42], [31.05], [42.5], [7.7], [26.18], [37.6], [4.08], [17.01], [24.26],
    table.hline(stroke: .4pt),
    table.cell([_Finetune_], align: left), table.cell(colspan: 12, []),
    [BEiT-3], [84.8], [96.5],[98.3], [67.2], [87.7], [92.8], [98], [100], [100], [90.3], [98.7], [99.5],
    [VLMo], [74.8], [93.1], [96.9], [57.2], [82.6], [89.8], [92.3], [99.4], [99.9], [79.3], [95.7], [97.8],
    table.hline(),
  ),
  caption: [],
)<image_text_retrieval_2>
