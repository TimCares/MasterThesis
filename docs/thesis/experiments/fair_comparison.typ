== Fair Comparison with Supervised Teacher <fair_comparison_supervised_teacher>

WEIRD THAT SELF-SUPERVISED IS BETTER, EVEN IF SUPERVISED TEACHER IS OF SAME size
-> maybe BECAUSE CLASSES DO NOT CAPTURE SOME CONTENT THAT WELL -> LESS CAPACITY THEN EMBEDDING -> EVEN though
EMBEDDING IS MORE IMAGE SPECIFIC, IT HAS MORE CAPACITY TO CAPTURE THE CONTENT OF THE CAPTION
-> TEST BEIT2 SELS-SUPERVISED + SUPERVISED WITH SHRE -> WE SEE THE DIRECT EFFECT OF FINETUNING
-> ALSO TEST BEIT2 SUPERVISED + SUPERVISED WITH SHRE BUT REGRESS CLS TOKEN, NOT PROB DIST

Throughout the experiments with the end-to-end self-supervised approach, we observed significant improvements in performance.
This increase is especially notable when comparing the results with the supervised teacher used at the beginning of the experiments
on multimodal knowledge distillation. However, the comparison with our Transformer-based SHRe model "SHRe#sub[T]" is not entirely fair,
as the self-supervised teacher BEiTv2 is a much larger model compared to the ResNet-50-A1 teacher of "SHRe#sub[T]": BEiTv2 has 86M parameters,
while ResNet-50-A1 has only 25M parameters. To have an actual comparison 
between a self-supervised (our approach) and a supervised (SHRe approach) teacher, we:

1. Add all improvements that are not specific to the self-supervised approach to "SHRe#sub[T]".
2. Train SHRe#sub[T] with a supervised teacher that is comparable in size to BEiTv2.

As for the first point, we add the improvements from the self-supervised approach to "SHRe#sub[T]". This includes the following:
1. ...

For the second point, we keep BEiTv2 as the teacher, but now use the variant that has not only been trained self-supervised on ImageNet-1K,
but also finetuned on ImageNet-1K with labels, i.e., the supervised variant of BEiTv2. That way, we can observe the direct impact of
switching to a supervised teacher.

#show table: set text(8pt)
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
    [FLAVA @flava], [42.74], [76.76], [-], [38.38], [67.47], [-], [67.7], [94.0], [-], [65.22], [89.38], [-],
    [CLIP @clip], [58.4], [81.5], [88.1], [37.8], [62.4], [72.2], [88.0],[98.7], [99.4], [68.7], [90.6], [95.2],
    [VLMo @vlmo], [74.8], [93.1], [96.9], [57.2], [82.6], [89.8], [92.3], [99.4], [99.9], [79.3], [95.7], [97.8],
    [BEiT-3 @beit3], [*84.8*], [*96.5*],[*98.3*], [*67.2*], [*87.7*], [*92.8*], [*98*], [*100*], [*100*], [*90.3*], [*98.7*], [*99.5*],
    [$"SHRe"_T$], [37.06], [67.74], [79.7], [25.3], [54.73], [68.19], [49.9], [78.5], [88.5], [37.04], [67.34], [77.38],
    table.hline(stroke: .3pt),
    [$"SHRe"_T$ BEiTv2], [37.06], [67.74], [79.7], [25.3], [54.73], [68.19], [49.9], [78.5], [88.5], [37.04], [67.34], [77.38],
    [S-SMKE], [37.06], [67.74], [79.7], [25.3], [54.73], [68.19], [49.9], [78.5], [88.5], [37.04], [67.34], [77.38],
    table.hline(),
  ),
  caption: [],
)<image_text_retrieval_shre_fair_comparison>