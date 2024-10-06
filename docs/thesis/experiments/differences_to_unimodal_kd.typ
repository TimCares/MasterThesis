=== Self-Supervised Teacher <multimodal_knowledge_distillation_self_supervised_teacher>
==== Challenges of Self-Supervision <multimodal_knowledge_distillation_challenges>
With the architecture and the first benchmarks in place, we can now focus on the main goal of this research: Applying the idea
of SHRe @shre using a self-supervised teacher model. A consequence of a self-supervised teacher is that the teacher's prediction
for a given image is not a probability distribution over a set of classes, but merely a representation of the input sample.
This raises the question
which training objective to use when using a self-supervised teacher, as we cannot use the KL-Divergence loss anymore. This only works
on probability distributions, and not on representations.

In our experiments on unimodal knowledge distillation (image and text) a self-supervised teacher did not pose a problem, as both the
teacher and student received the same input, and the latter was able to regress all time steps of the teacher model.
However, this was only possible because the teacher and student received exactly the same input: The patch/text token
the teacher and student received at a time step $i$ was the same for both models, allowing the student to learn the teacher's
representation for each patch or text token, respectively. Since we predict the output of an image teacher model, in which ever form it may be,
the aforementioned still holds true when our multimodal student receives an image as the input. The output will be a representation
for each image patch, which is also the prediction of the teacher model, allowing for the same approach used in unimodal knowledge distillation
(see @unimodal_kd_vision).

When the multimodal model receives a text (an image's caption) as the input however, the teacher's prediction is still a representation of the image. This poses the following problems:

1. The content of the time steps is aligned: The text token at a time step $i$ does not
   necessarily describe the content of the image patch at the same time step $i$. Moreover, text naturally contains fill words such as "the", "a", "is",
   which do not have any meaning with respect to the content of an image. Another example are padding tokens, which do not contain any information
   at all, and are merely used for batching a set of texts.
2. The number of time steps is not aligned: The number of text tokens in a caption does not match the number of image patches.
   Consequently, there is not a 1:1 mapping between the time steps of the image and the text.

#figure(
  image("../figures/mm_kd_cls_token.png"),
  caption: [The meaning/content of time steps across modalities is not aligned, and the number of time steps will differ between modalities. This makes alignment on the level of individual time steps impossible. The cls token aggregates global information independent of time steps, and captures the meaning and interpretation of the respective input, making alignment possible. However, this requires the teacher cls token to not contain any modality-specific (in this case image) information. Image-Text example is taken from the COCO train set @coco.],
) <mm_kd_cls_token>

Both problems are illustrated in @mm_kd_cls_token.
It is therefore not possible to regress the representation of individual patches when the multimodal student model receives a text as the input. That means we have to resort to regressing the global representation of the image, which is the $mono(["I_CLS"])$ token.
This choice solves both of the aforementioned problems, as we do not rely on the content and placement of individual time steps.
To clarify, the concept of the forward pass remains the same as in SHRe @shre and our Transformer variant of the previous chapter (@transformer_shre):
For a single image-text pair, the image can be passed to the teacher, and the image and its caption can be passed to the student seperately.
When we focus on the global representation (the cls token), the teacher will always return a representation of the $mono(["I_CLS"])$ token,
aggregating global information of the image. The same holds true for the student, producing a representation of the $mono(["I_CLS"])$ token.
As a training objective we can then require:

$
min_(bold(h)^s_(v, mono(["I_CLS"])))||bold(h)^s_(v, mono(["I_CLS"])) - bold(h)^t_(v, L_t, mono(["I_CLS"]))||
$

We denote a representation generated by the student with the superscript $s$, and the teacher with the superscript $t$.
The objective forces the student to push its representation of the $mono(["I_CLS"])$ token as close as possible to the teachers representation of 
the same token.

Most importantly, the student can also be trained to push the representation of the caption, given by the $mono(["T_CLS"])$
token, as close as possible
to the teacher's representation of the image:

$
min_(bold(h)^s_(w,  mono(["T_CLS"])))||bold(h)^s_(w, mono(["T_CLS"])) - bold(h)^t_(v, L_t, mono(["I_CLS"]))||
$

The combined training objective when regressing only global information is then:

$
min_(bold(h)^s_(v, mono(["I_CLS"])), bold(h)^s_(w, mono(["T_CLS"])))
||bold(h)^s_(v, mono(["I_CLS"])) - bold(h)^t_(v, L_t, mono(["I_CLS"]))|| +
||bold(h)^s_(w, mono(["T_CLS"])) - bold(h)^t_(v, L_t, mono(["I_CLS"]))||
$

While this objective in theory forces the student to output the same global representation for an image and its caption as the teacher,
it requires the teacher to produce a representation of the $mono(["I_CLS"])$ token that is abstract enough to also describe the content of the caption.
Conretely, the representation of the $mono(["I_CLS"])$ token should *not* contain any image-specific information, as this would make it impossible
for the student to align the representation of the caption with that of the image: It is not possible to extract any image-specific information,
like the exact position of an object in the image, from the caption. Consequently, whether the representation of the
$mono(["I_CLS"])$ token produced by the teacher is abstract enough to also describe the content of the caption remains to be seen in the
following experiments.

The challenges that come with the choice of an *unimodal* self-supervised teacher raises the question why we do not directly use a *multimodal* model as the teacher.
This reason behind this choice can be attributed to the fact that the goal of this research is to train a multimodal model without using any existing,
especially pretrained, _multimodal_ components.
Instead, we aim to extract knowledge from purely unimodal models and learn to generate modality-invariant features from it (1), and to not rely on
labeled data in the end-to-end training of the multimodal model (2). This includes the teacher model, which should not be trained
on labeled data, but only self-supervised.

==== Feature-based Knowledge Distillation
Based on the previous discussion, we propose an adjustment to the loss used to train the multimodal student.
We reformulate the knowledge distillation loss to a feature-based loss, which implements the training objective discussed
in the previous section:

$
cal(L)_("KD") &= \ 
1/2 * cal(L)_("KD")^v &+ 1/2 * cal(L)_("KD")^w = \
1/2 * ||bold(h)^s_(v, mono(["I_CLS"])) - bold(h)^t_(v, L_t, mono(["I_CLS"]))||^2_2 &+ 
1/2 * ||bold(h)^s_(w, mono(["T_CLS"])) - bold(h)^t_(v, L_t, mono(["I_CLS"]))||^2_2
$ <kd_loss_mse>

The loss is the average of the mean squared error between the student's and teacher's representation of the $mono(["I_CLS"])$ token,
and between
the student's representation of the $mono(["T_CLS"])$ token and the teacher's representation of the $mono(["I_CLS"])$ token.

The target representation $bold(h)^t_(v, L_t, mono(["I_CLS"])) in RR^768$ stems from a new, now self-supervised, teacher model,
which is BEiTv2 @beitv2.
Since BEiTv2
is a Transformer-based self-supervised image model, it also returns a sequence of patch representations, and we extract the representation
$bold(h)^t_(v, L_t, mono(["I_CLS"])) in RR^768$
of the $mono(["I_CLS"])$ token from the output sequence. The teacher has $L_t=12$ Transformer layers, and is based on the ViT-B/16 @vit architecture.
It therefore is with almost 86M parameters significantly larger than the previous ResNet-50-A1 @resnet_50_a1 teacher model, which had 25M parameters.

The architecture of the student remains largely the same as before, the prediction of the student for the $mono(["I_CLS"])$ token of
the teacher, which is the global image represenation, is generated by the student's classification head. A consequence of this is that
the classification head now also outputs 768-dimensional representations,
instead of the 1000-dimensional output used previously for the ImageNet-1K classes. Therefore:
$bold(h)^s_(v, mono(["I_CLS"])) in RR^768$ and $bold(h)^s_(w, mono(["T_CLS"])) in RR^768$.
As a side note, the student's classification head is technically not a classification head anymore, as it does not predict any classes
but rather a representation of the teacher. We will refer to it as the "regression head" from now on.

==== Results <s_smke_results>
Apart from the change in the teacher and the impact on the loss, the training setup remains the same, and no hyperparameters are changed.
As we now do not follow the approach of SHRe, that is, predicting a probability distribution over ImageNet-1K classes @shre,
we name the new approach "#text(weight: "bold")[S]elf#text(weight: "bold")[-S]upervised #text(weight: "bold")[M]ultimodal
#text(weight: "bold")[K]nowledge #text(weight: "bold")[E]xtraction" (S-SMKE). 

Surprisingly, the results show a significant improvement in retrieval performance compared to the previous benchmarks,
as shown in @image_text_retrieval_S_SMKE. We achieve competetive performance with FLAVA @flava and CLIP @clip on COCO, and
even outperform CLIP on COCO R@10 image retrieval by more than 6 percentage points.

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
    [FLAVA @flava], [42.74], [76.76], [-], [*38.38*], [*67.47*], [-], [67.7], [94.0], [-], [65.22], [89.38], [-],
    [CLIP @clip], [*58.4*], [*81.5*], [88.1], [37.8], [62.4], [72.2], [*88.0*],[*98.7*], [*99.4*], [*68.7*], [*90.6*], [*95.2*],
    table.hline(stroke: .3pt),
    [SHRe#sub[T]], [44.6], [75.3], [85.64], [31.69], [62.1], [74.48], [58.8], [85.6], [92.4], [43.92], [74.06], [82.8],
    [S-SMKE ($arrow.t 4.38$)], [51.66], [79.9], [*88.66*], [36.17], [66.55], [*78.28*], [64.5], [88.4], [93.0], [51.78], [78.54], [86.46],
    table.hline(),
  ),
  caption: [
   An updated contrastive loss with a self-supervised teacher, which we denote as "S-SMKE" leads to an average gain of
   4.38% compared to our previous approach SHRe#sub[T]. On Flickr30K, we reach substantial improvements while still being behind
   FLAVA @flava and CLIP @clip.
  ],
)<image_text_retrieval_S_SMKE>
#show table: set text(12pt)

Less surprising is that we observe a drop in ImageNet-1K classification performance of more than 14 percentage points compared to
the supervised teacher of SHRe @shre (see left in @imagenet_1k_classification_S_SMKE). We consider this as not surprising,
because we previously used representations $bold(h)'''^s_(v, K, mono(["I_CLS"]))$ and $bold(h)'''^s_(w, K, mono(["T_CLS"]))$ for
the CLIP-like Imagenet-1K classification, which were also the representations used for predicting the ImageNet-1K classes as given by a supervised teacher
trained on ImageNet-1K. Therefore, those representations were optimized for predicting ImageNet-1K classes. Now, we regress the teacher's
representation of the $mono(["I_CLS"])$ token, and since the teacher was not trained on the ImageNet-1K classes, the student's representations
do not contain any information about them. This leads to a drop in classification performance.

While we now train the student without the ImageNet-1K classes, the CLIP-like zero-shot classification still cannot be considered
as zero-shot. This can be attributed to the fact that, although trained without the classes/labels, our teacher model BEiTv2 @beitv2
has been trained on ImageNet-1K. The knowledge contained in the teacher's representations of the $mono(["I_CLS"])$ token could therefore
leak information about the images of ImageNet-1K to the student. In order for it to be a true zero-shot approach the teacher would have to be
trained on a dataset disjoint from ImageNet-1K.

Speaking of the teacher, its increased size leads to a longer training time, as shown in @imagenet_1k_classification_S_SMKE (right),
which would be even more pronounced if we wouldn't use DDP with 2 GPUs.


#show table: set text(8pt)
#figure(
    stack(
        dir: ltr,
        spacing: 2mm,
   table(
    columns: 2,
    stroke: none,
    table.hline(),
    table.header(
      [*Approach*],
      [*Accuracy*],
      table.hline(stroke: .6pt),
    ),
    [Visual N-Grams @visual_n_grams $dagger$], [11.5], 
    [CLIP @clip $dagger$], [*76.2*], 
    [SHRe#sub[T] (ours)], [44.57], 
    [S-SMKE (ours)], [30.4],
    table.hline(),
  ),
   table(
    columns: 2,
    stroke: none,
    table.hline(),
    table.header(
      [*Approach*],
      [*Training time (h)*],
      table.hline(stroke: .6pt),
    ),
    [$"SHRE"_("DDP")$], [*4.8*],
    [SHRe#sub[T]], [5.27],
    [S-SMKE], [6.9],
    table.hline(),
  ),
    ),
    caption: [
      (Left) Accuracy of S-SMKE on the validation set of ImageNet-1k using CLIP-like zero-shot classification. We compare to
      our supervised (SHRe @shre) approach, CLIP @clip,
      and Visual N-Grams @visual_n_grams. The latter was developed as a first proof-of-concept of zero-shot image classification
      in 2017 @visual_n_grams @clip. We denote a full zero-shot approach with $dagger$.
      (Right) Development of the training time for the different approaches. Adding additional parameters through self-attention (SHRe#sub[T])
      and a large self-supervised teacher (BEITv2 @beitv2 in S-SMKE) increases the training time by more than two hours.
    ]
) <imagenet_1k_classification_S_SMKE>
#show table: set text(12pt)