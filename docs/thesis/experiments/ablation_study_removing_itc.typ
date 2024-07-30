#set math.equation(numbering: "(1)")
===== Ablation Study: Removing ITC
In the previous chapters, we made efforts to improve and extend the architecture and training of the model to achieve
better alignment of the modalities, which should go hand in hand with better retrieval performance.
However, we observed that contrastive learning is highly dependent on the granularity of the data (TODO: cite),
e.g. the quality of captions, and the number of negative examples (TODO: cite). While there are viable options to
overcome the latter, they are not always feasible, nor efficient, and the right hardware is required.
Therefore, we identify Image-Text Contrastive Learning (ITC) as a weak point of our approach.
The best approach would be to not use contrastive learning at all, which is why we will investigate the effects of an absence of ITC in this ablation study.

As an intersting side note, at the point of writing, the state-of-the-art (SOTA) vision-language model, BEiT-3, gives us a good reason to discard ITC. BEiT-3 pre-training is performed without contrastive learning, and the authors report SOTA results after fine-tuning on retrieval tasks MSCOCO @coco and Flickr30K @flickr30k. Even without fine-tuning, BEiT-3 achives competitive results on Flickr30K, and even outperforms models that where trained using contrastive learning (see @beit3_zero_shot). 

#figure(
  image(
  width: 50%,
  "../figures/cmli.png"),
  caption: [],
) <beit3_zero_shot>

- retrieval application then truly becomes zero-shot, and authors report high performance

- @shre mentions that without ITC, model outputs are not aligned, and therefore not suited for retrieval
- ablation study in (TODO cite) shows: TODO
- for current approach representations will still be aligned, because we are regressing the $mono(["I_CLS"])$ token of BEiT-2,
  not a probability distribution over the classes
- so we are regressing actual features -> loss of 0 would mean representations are exactly the same and therefore aligned
- @shre does prediction of teacher outputs on the level of classes, which are not feature/representation-based
- so it is true that in the supervised case, when regressing a probability distribution the final representation, i.e. the $mono(["I_CLS"])$ token,
  does not need to be aligned, only the probability distribution
- but again, we are explicitly regressing $mono(["I_CLS"])$, so representations are aligned

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
    [*Sx3HRe#sub[ITC]*], [41.36], [71.16], [82.0], [30.2], [59.46], [72.54], [9.5], [35.68], [50.18], [8.38], [37.54], [49.88],
    table.cell([_Zero-Shot_], align: left), table.cell(colspan: 12, []),
    [*Sx3HRe#sub[\~ITC]*], [33.52], [59.34], [70.14], [11.26], [29.12], [40.40], [8.86], [29.88], [41.98], [4.45], [18.42], [26.82],
    [*Sx3HRe#sub[COS]*], [35.48], [60.74], [71.44], [12.46], [31.74], [43.71], [9.44], [31.18], [43.50], [4.8], [20.98], [30.08],
    table.hline(),
  ),
  caption: [],
)<image_text_retrieval_shre_first>

#bibliography("../references.bib")