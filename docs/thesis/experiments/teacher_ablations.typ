== Teacher Ablation Studies <teacher_ablations>
In this section we will investigate the impact of using teachers different from BEiTv2 @beitv2, but still self-supervised. We will compare
the results with an approach that does not make use of knowledge distillation at all.

=== Different Teachers
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

=== Removing Distillation
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

Since we are now not using a teacher that has been trained on ImageNet-1K (we are not using a teacher at all), no information about
the ImageNet-1K dataset can leak to our student model (see @s_smke_results). The performance on ImageNet-1K using CLIP zero-shot
classification is therefore, for the first time in this work, an actual zero-shot application.

=== Results

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
    table.hline(stroke: .2pt),
    [BEiTv2 @beitv2 (Baseline)], [53.54], [81.1], [*89.52*], [35.65], [66.0], [77.77], [70.9], [92.1], [96.0], [52.72], [80.2], [87.46],
    [Data2Vec2 @data2vec2 (8.68 $arrow.b$)], [43.08], [73.24], [83.84], [29.89], [59.59], [72.41], [55.0], [81.8], [87.7], [41.84], [70.2], [80.24],
    [$-$ (4.17 $arrow.b$)], [47.48], [76.46], [85.98], [34.20], [64.02], [75.94], [61.0], [86.0], [92.1], [47.1], [74.82], [83.78],
    table.hline(),
  ),
  caption: [
    Comparison of the retrieval performance when using different teachers for knowledge distillation. The value in parentheses next to the teacher
    indicates the average drop or gain of percentage points with respect to our default teacher BEiTv2 @beitv2.
  ],
)<image_text_retrieval_teachers>
#show table: set text(12pt)

#show table: set text(8pt)
#figure(
  table(
  columns: 3,
  stroke: none,
  table.hline(),
  table.header(
    [Teacher],
    [ImageNet-1K],
    [Training time (h)],
  ),
  table.hline(stroke: .6pt),
  [BEiTv2 @beitv2], [37.0], [7.3],
  [Data2Vec2 @data2vec2], [24.5], [7.4],
  [$-$], [25.8*$dagger$*], [*4.5*],
  table.hline(),
),
    caption: [
      Comparison of ImageNet-1K performance, using CLIP-like zero-shot classification, and training time for various teachers.
      *$dagger$* indicates zero-shot, and $-$ indicates no teacher.
    ]
) <imagenet_of_teachers>

We show a detailed comparison of the retrieval performance between different teacher in @image_text_retrieval_teachers, and the performance on
ImageNet-1K, including the training time for the distillation, in @imagenet_of_teachers. We observe that the choice of the teacher is significant
for the performance of the student, especially in the unimodal case. The performance when using the Data2Vec2 @data2vec2 teacher is significantly
lower compared to our baseline, which is BEiTv2 @beitv2. On average, lose 8.68 percentage points over all retrieval metrics, and 12.5 percentage points
on ImageNet-1K CLIP-like classification. We assume that this can be attributed to the different strategies used to train BEiTv2 and Data2Vec2.
In contrast to Data2Vec2, the architecture and loss of BEiTv2 forces the model to aggregate as much (global) information
as possible in the $mono(["I_CLS"])$ token @beitv2, which is the token we predict in the student model. Aggregating global information in one token
leads to it being less sensitive to smaller regions of the image, which crucial when we predict the representation of the $mono(["I_CLS"])$ token
using the caption of the image. Data2Vec2, on the other hand, only includes a small term in the loss that forces the model to aggregate global
information in the $mono(["I_CLS"])$ token. However, this term has a weight of only 1% of the total loss @data2vec2.

This loss term of Data2Vec2, which they refer to as the CLS loss @data2vec2, has such a small weight that even using no teacher at all, and only
the contrastive loss, leads to a better performance than when using Data2Vec2. It is particularly striking that the performance on ImageNet-1K
using CLIP-like zero-shot classification is increasing by over 1 percentage points when using no teacher at all, compared to using Data2Vec2
(see @imagenet_of_teachers). We observe this increase even though using no teacher at all is an actual zero-shot application, while when using
Data2Vec2, information about ImageNet-1K can leak to the student model (Data2Vec2 has been trained on Imagenet-1K @data2vec2),
making it _not_ a zero-shot application.
