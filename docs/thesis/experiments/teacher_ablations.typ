=== Teacher Ablation Studies <teacher_ablations>
In this section we will investigate the impact of using teachers different from BEiTv2 @beitv2, but still self-supervised. We will compare
the results with an approach that does not make use of knowledge distillation at all.

==== Unimodal
First, we will train the same model as in the previous experiments, but this time using two different self-supervised image teachers:
Data2Vec2 @data2vec2 and DINO @dino. The choice of Data2Vec2 @data2vec2 is motivated by the fact that we already used it in
the unimodal image distillation in @unimodal_kd_vision, and because we initialize the image encoder of the multimodal model with
the weights of the first 6 layers of Data2Vec2. DINO @dino has shown to be a self-supervised model with an excellent understanding of image
features, and its attention mechanism is able to detect high-level semantics in images,
examples of which are illustrated in @dino_cls_attn_examples. We therefore test if this deep image understanding can be
used to better align image and text, or if the features are too image-specific to be useful for the multimodal task.

In both cases, we will use the contrastive target loss with a memory bank for the knowledge distillation process. As the output
representation of the teacher we keep using the representation of the $mono(["I_CLS"])$ token, $bold(h)_(v, L_s, mono(["I_CLS"]))$.
Both models are based on the same ViT-B/16 @vit @data2vec2 @dino architecture, which is the same as used by BEiTv2 @beitv2, so all teachers
are comparable in terms of model size and complexity.

==== Multimodal
Although the goal of this work is to explicitly _not_ use a multimodal teacher, we will still examine the impact of using one, and we
expect our student to reach a better performance than when using a unimodal teacher.



==== Removing Distillation
Throughout the previous sections we repeatedly attempted to improve our approach by reducing the gap between the text-to-image and image-to-text
knowledge distillation losses. Unfortunately, only one of them, the contrastive target loss, was successful, but only increased the performance
marginally. That is why we will now investigate whether the knowledge distillation process is beneficial at all.

To test this, we will train a version without any knowledge distillation, and only use the contrastive loss:

$
cal(L)_"S_SMKE" = cal(L)_"CL"
$

Since we now only focus on the alignment of the visual and textual features, we remove the linear layer that produced
the representations $bold(h)'''_(v, K, mono(["I_CLS"]))$ and $bold(h)'''_(w, K, mono(["T_CLS"]))$. They were used to predict
the teacher representation $bold(h)_(v, L_s, mono(["I_CLS"]))$ using the mse loss in earlier experiments, and using
the contrastive target loss in the latest successful experiments. However, without knowledge distillation, they are no longer
needed.

Everything else about the model architecture and training process remains the same, both the image and text encoder
are still initialized with layers from Data2Vec2 @data2vec2 and BERT @bert, respectively.

==== Results

#show table: set text(8pt)
#figure(
  table(
  columns: (25%, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: none,
    table.hline(),
    table.header(
      table.cell(rowspan: 3, colspan: 1, align:horizon, [*Teacher*]),
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
    [BEiTv2 @flava], [53.54], [81.1], [*89.52*], [35.65], [66.0], [77.77], [70.9], [92.1], [96.0], [52.72], [80.2], [87.46],
    [Data2Vec2 @data2vec2], [53.54], [81.1], [*89.52*], [35.65], [66.0], [77.77], [70.9], [92.1], [96.0], [52.72], [80.2], [87.46],
    table.hline(stroke: .2pt),
    [BEiT-3 @beit3], [53.54], [81.1], [*89.52*], [35.65], [66.0], [77.77], [70.9], [92.1], [96.0], [52.72], [80.2], [87.46],
    table.hline(stroke: .2pt),
    [-], [53.54], [81.1], [*89.52*], [35.65], [66.0], [77.77], [70.9], [92.1], [96.0], [52.72], [80.2], [87.46],
    table.hline(),
  ),
  caption: [
    Comparison of the retrieval performance when using different teachers for knowledge distillation.
  ],
)<image_text_retrieval_teachers>
#show table: set text(12pt)