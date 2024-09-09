=== Self-Supervised Teacher <multimodal_knowledge_distillation_self_supervised_teacher>
==== Challenges of Self-Supervision <multimodal_knowledge_distillation_challenges>
With the architecture and the first benchmarks in place, we can now focus on the main goal of this research: Applying the idea
of SHRe @shre using a self-supervised teacher model. A consequence of a self-supervised teacher is that the teacher's prediction
for a given image is not a probability distribution over a set of classes, but merely a representation of the input sample.
This raises the question
which training objective to use when using a self-supervised teacher, as we cannot use the KL-Divergence loss anymore. This only works
on probability distributions, and not on representations.

In preliminary experiments on unimodal knowledge distillation, a self-supervised teacher did not pose a problem, as both the
teacher and student received the same input, and the latter was able to regress all time steps of the teacher model.
However, this was only possible because the teacher and student received exactly the same input: For a given time step, 
the patch or text token at that time step was the same for both models, allowing the student to learn the teacher's
representation for each patch or text token, respectively. Since we predict the output of an image teacher model, in which ever form it may be,
the aforementioned still holds true when the multimodal model receives an image as the input. The output will be a representation
for each image patch, which is also the prediction of the teacher model, allowing for the same approach used in unimodal knowledge distillation
(see @unimodal_kd_vision).

When the multimodal model receives a text (an image's caption) as the input however, the teacher's prediction is still a representation of the image. This poses the following problems:

1. The number of patches (and therefore time steps) in an image is usually not the same as the number of text tokens of its corresponding caption.
   Consequently, we do not have a one-to-one correspondence between the time steps of the image and text models, and the student cannot regress
   every time step of the teacher model.

2. Even if the number of time steps were the same, the content of the time steps would not be aligned: The text token at a time step does not
   necessarily correspond to the content of the image patch at the same time step. Moreover, text naturally contains fill words such as "the", "a", "is",
   which do not have any meaning with respect to the content of an image. Another example are padding tokens, which do not contain any information
   at all, and are merely used for batching a set of texts.

Consequently, regressing the representation of individual patches when the multimodal student model receives a text as the input is not possible,
and we have to resort to regressing the global representation of the image, which is the $mono(["I_CLS"])$ token.
This choice solves both of the aforementioned problems, as we do not rely on the representations of individual time steps.
To clarify, the concept of the forward pass remains the same as in SHRe @shre and our Transformer variant of the previous chapter (@transformer_shre):
For a single image-text pair, the image can be passed to the teacher, and the image and its caption can be passed to the student seperately.
When we focus on the global representation (the cls token) returned, the teacher will always return a representation of the $mono(["I_CLS"])$ token,
aggregating global information of the image. The same holds true for the student, producing a representation of the $mono(["I_CLS"])$ token.
As a training objective we can then require:

$
min_(bold(h)^s_(v, K, mono(["I_CLS"])))||bold(h)^s_(v, K, mono(["I_CLS"])) - bold(h)^t_(v, L_t, mono(["I_CLS"]))||
$

We denote a representation generate by the student with the superscript $s$, and the teacher with the superscript $t$.
The objective forces the student to push its representation of the $mono(["I_CLS"])$ token as close as possible to the teachers representation of 
the same token. Most importantly, the student can also be trained to push the representation of the caption, given by the $mono(["T_CLS"])$
token, as close as possible
to the teacher's representation of the image:

$
min_(bold(h)^s_(w, K, mono(["T_CLS"])))||bold(h)^s_(w, K, mono(["T_CLS"])) - bold(h)^t_(v, L_t, mono(["I_CLS"]))||
$

An illustration of the problem posed by the misalignment of time steps and the solution of regressing the global representation of the image
is shown in @mm_kd_cls_token.

The combined training objective when regressing only global information is then:

$
min_(bold(h)^s_(v, K, mono(["I_CLS"])), bold(h)^s_(w, K, mono(["T_CLS"])))
||bold(h)^s_(v, K, mono(["I_CLS"])) - bold(h)^t_(v, L_t, mono(["I_CLS"]))|| +
||bold(h)^s_(w, K, mono(["T_CLS"])) - bold(h)^t_(v, L_t, mono(["I_CLS"]))||
$

While this objective in theory forces the student to output the same global representation for an image and its caption as the teacher,
it requires the teacher to produce a representation of the $mono(["I_CLS"])$ token that is abstract enough to also describe the content of the caption.
Conretely, the representation of the $mono(["I_CLS"])$ token should *not* contain any image-specific information, as this would make it impossible
for the student to align the representation of the caption with that of the image: It is not possible to extract any image-specific information,
like the exact position of an object in the image, from the caption. Consequently, whether the representation of the
$mono(["I_CLS"])$ token produced by the teacher is abstract enough to also describe the content of the caption remains to be seen in the
following experiments.

#figure(
  image("../figures/mm_kd_cls_token.png"),
  caption: [The meaning/content of time steps across modalities is not aligned, and the number of time steps will differ between modalities. This makes alignment on the level of individual time steps impossible. The cls token aggregates global information independent of time steps, and captures the meaning and interpretation of the respective input, making alignment possible. However, this requires the teacher cls token to not contain any modality-specific (in this case image) information. Image-Text example is taken from the COCO train set @coco.],
) <mm_kd_cls_token>

The challenges that come with the choice of a self-supervised teacher raises the question why we do not directly use a multimodal model as the teacher.
This reason behind this choice can be attributed to the fact that the goal of this research is to train a multimodal model without using any existing,
especially pretrained, _multimodal_ components.
Instead, we aim to extract knowledge from purely unimodal models and learn to generate modality-invariant features from it (1), and to not rely on
labeled data in the end-to-end training of the multimodal model (2). This includes the teacher model, which should not be trained
on labeled data, but only self-supervised.

==== Feature-based Knowledge Distillation
Based on the previous discussion, we propose an adjustment to the loss used to train the multimodal student.
We formulate the knowledge distillation loss to a feature-based loss, which implements the training objective discussed
in the previous section:

$
cal(L)_("KD") &= \ 
1/2 * cal(L)_("KD")^v &+ 1/2 * cal(L)_("KD")^w = \
1/2 * ||bold(h)'''^s_(v, K, mono(["I_CLS"])) - bold(h)^t_(v, L_t, mono(["I_CLS"]))||^2_2 &+ 
1/2 * ||bold(h)'''^s_(w, K, mono(["T_CLS"])) - bold(h)^t_(v, L_t, mono(["I_CLS"]))||^2_2
$ <kd_loss_mse>

The loss is the mean of the mean squared error between the student's and teacher's representation of the $mono(["I_CLS"])$ token,
and between
the student's representation of the $mono(["T_CLS"])$ token and the teacher's representation of the $mono(["I_CLS"])$ token.

The target representation $bold(h)^t_(v, L_t, mono(["I_CLS"])) in RR^768$ stems from a new, now self-supervised, teacher model,
which is BEiTv2 @beitv2.
Since BEiTv2
is a Transformer-based self-supervised image model it also returns a sequence of patch representations, and we extract the representation
$bold(h)^t_(v, L_t, mono(["I_CLS"]))$
of the $mono(["I_CLS"])$ token from the output sequence. The teacher has $L_t=12$ Transformer layers, and is based on the ViT-B/16 architecture.
It therefore is with almost 86M parameters significantly larger than the previous ResNet-50-A1 @resnet_50_a1 teacher model, which had 25M parameters.
The teacher being based on the ViT-B/16 architecture also means that the dimensionality of the output representations is 768.

The architecture of the student remains largely same as before, the prediction of the student for the $mono(["I_CLS"])$ token of
the teacher, which is the global image represenation, is generated by the student's classification head. A consequence of this is that
the classification head now also outputs 768-dimensional representations,
instead of the 1000-dimensional output used previously for the ImageNet-1K classes. Therefore:
$bold(h)'''^s_(v, K, mono(["I_CLS"])) in RR^768$ and $bold(h)'''^s_(w, K, mono(["T_CLS"])) in RR^768$.
As a side note, the student's classification head is technically not a classification head anymore, as it does not predict any classes
but rather a representation of the teacher.

==== Loss-Equilibrium Harms Alignment

An important change we make is removing the application of the contrastive loss on the outputs of the student's (classification) head
$bold(h)'''^s_(v, K, mono(["I_CLS"]))$ and $bold(h)'''^s_(w, K, mono(["T_CLS"]))$, which were also the representations used for image-text
retrieval and CLIP-like ImageNet-1K classification in the previous benchmarks.
We consider it as unwise to apply a contrastive loss on the output of the classification head, as this output is also used to
regress the teacher's representation $bold(h)^t_(v, L_t, mono(["I_CLS"]))$ of the $mono(["I_CLS"])$ token (see @kd_loss_mse).
In preliminary experiments, we found that requiring
the student's outputs $bold(h)'''^s_(v, K, mono(["I_CLS"]))$ and $bold(h)'''^s_(w, K, mono(["T_CLS"]))$ to be close to each other
under the contrastive loss, while also requiring them to be close to the teacher's representation $bold(h)^t_(v, L_t, mono(["I_CLS"]))$,
leads diminishing results on retrieval.

Usually, the combination of both losses would actually be a good idea, as it would push both representations closer together
(contrastive loss), while also pushing each of them closer to the teacher's
representation (knowledge distillation loss, see @kd_loss_mse).
However, since the teacher representation $bold(h)^t_(v, L_t, mono(["I_CLS"]))$ actually still contains image-specific information,
which we will illustrate later,
both losses work against each other:

The contrastive loss pushes the student's representations of the image $bold(h)'''^s_(v, K, mono(["I_CLS"]))$
and caption $bold(h)'''^s_(w, K, mono(["T_CLS"]))$ closer together, while
the knowledge distillation loss pushes both $bold(h)'''^s_(v, K, mono(["I_CLS"]))$ and $bold(h)'''^s_(w, K, mono(["T_CLS"]))$
towards the teacher representation $bold(h)^t_(v, L_t, mono(["I_CLS"]))$.

The issue arises because the teacher's representation $bold(h)^t_(v, L_t, mono(["I_CLS"]))$ still contains image-specific information.
This means the student's image representation $bold(h)'''^s_(v, K, mono(["I_CLS"]))$ will always be closer to
$bold(h)^t_(v, L_t, mono(["I_CLS"]))$ than the caption representation $bold(h)'''^s_(w, K, mono(["T_CLS"]))$ (see @mse_kd_loss_comparison),
because only the image input allows the student to extract this image-specific information, not the caption.
As a result, it is easier for the student to regress the teacher's image representation with its own image representation
than with its caption representation.

At some point, there will be an equilibrium between the two losses, where the student's image representation
is being pulled equally toward the teacher's representation (kd loss) and toward the caption representation (contrastive loss). The consequence of
this equilibrium is that the student's representations $bold(h)'''^s_(v, K, mono(["I_CLS"]))$ and $bold(h)'''^s_(w, K, mono(["T_CLS"]))$
will not be as close to each other as they could be without the knowledge distillation loss,
leading to suboptimal alignment between image and caption. This ultimately harms retrieval performance,
as the alignment between image and caption representations is weakened. A visual representation of the problem is shown in @cl_kd_vs_cl.

#figure(
  image(
  width: 75%,
  "../figures/cl+kd_vs_cl.png"),
  caption: [
      Using both the contrastive loss and the knowledge distillation loss on the final output of the student leads
      to suboptimal alignment between image and caption representations. (a) The greyed out arrow indicates that the minimum distance between the student's caption representation
      and the teacher's image representation is reached. Only the student's image representation can
      be transported closer to the teacher's image representation, but this force stands in an
      equilibrium with the contrastive loss between both student representations. The
      knowledge distillation loss leads to a suboptimal alignment between image and caption representations.
      (b) Using only the contrastive loss on the final output of the student allows the student's image and
      caption representations to be as close as possible to each other, leading to a better alignment between image and caption representations.
],
) <cl_kd_vs_cl>

The reason why we know that the teacher's representation $bold(h)^t_(v, L_t, mono(["I_CLS"]))$ still contains image-specific information
is the comparison between the training loss of $cal(L)_("KD")^v$ and $cal(L)_("KD")^w$ in @mse_kd_loss_comparison.
It is observable that the loss $cal(L)_("KD")^v$ is significantly lower than $cal(L)_("KD")^w$, which indicates that the student
is able to regress the teacher's image representation $bold(h)^t_(v, L_t, mono(["I_CLS"]))$ better
with its own image representation $bold(h)'''^s_(v, K, mono(["I_CLS"]))$ than with its caption representation $bold(h)'''^s_(w, K, mono(["T_CLS"]))$.
If the teacher's representation $bold(h)^t_(v, L_t, mono(["I_CLS"]))$ would not contain any image-specific information, say,
it encodes the information "dog" for an image of a dog, the student should be able to regress this representation with both
of its representations equally well. However,
this is not the case.

#figure(
  image(
  width: 50%,
  "../figures/mse_kd_loss_comparison.png"),
  caption: [
   Comparison between the Image-2-Image KD loss $cal(L)_("KD")^v$ and the Text-2-Image KD loss $cal(L)_("KD")^w$.
   The latter is significantly higher, indicating that the student is able to regress the teacher's
   image representation better with its own image representation than with its caption representation.
],
) <mse_kd_loss_comparison>

As mentioned before, our solution is to remove the contrastive loss $cal(L)_("CL"''')$ working on
$bold(h)'''^s_(v, K, mono(["I_CLS"]))$ and $bold(h)'''^s_(w, K, mono(["T_CLS"]))$.
This removes the alignment constraint between the student's image and caption representations of the "classification" head.
Correspondingly, we now use the representations $bold(h)''^s_(v, K, mono(["I_CLS"]))$ and $bold(h)''^s_(w, K, mono(["T_CLS"]))$
for image-text retrieval and CLIP-like ImageNet-1K classification, which are still aligned under the contrastive loss $cal(L)_("CL"'')$.
The full contrastive loss is then:

$
cal(L)_("CL") &= \
1/2 * (cal(L)_("CL"') &+ cal(L)_("CL"'')) = \
1/4cal(L)_("CL"')^("i2t") &+ 1/4cal(L)_("CL"')^("t2i") + \
1/4cal(L)_("CL"'')^("i2t") &+ 1/4cal(L)_("CL"'')^("t2i")
$ <contrastive_loss_sx3hre>

The "classification" head, which should rather be called the "regression" head, considering we regress the teacher's representation
under the mean squared error loss, is now only used for the knowledge distillation loss. Since its output is now not used for any
vision-language tasks like retrieval, we can discard it after training. What remains is simply the output of the shared Transformer
layer. With this approach, we focus on aligning the modalities through the contrastive loss, while still utilizing the knowledge
learned by the teacher model. 

==== Results
Apart from the change in the teacher and the loss functions, the training setup remains the same, and no hyperparameters are changed.
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

Less surprisingly, we observe a drop in ImageNet-1K classification performance of more than 14 percentage points compared to
the supervised teacher approach SHRe @shre, as shown in @imagenet_1k_classification_S_SMKE (left). We consider this as not surprising,
because we previously used representations $bold(h)'''^s_(v, K, mono(["I_CLS"]))$ and $bold(h)'''^s_(w, K, mono(["T_CLS"]))$ for
the classification, which were also the representations used for predicting the ImageNet-1K classes as given by a supervised teacher
trained on ImageNet-1K. Therefore, those representations were optimized for predicting ImageNet-1K classes. Now, we regress the teacher's
representation of the $mono(["I_CLS"])$ token, and since the teacher was not trained on the ImageNet-1K classes, the student's representations
do not contain any information about them. This leads to a drop in classification performance.

While we now train the student without the ImageNet-1K classes, the CLIP-like zero-shot classification still cannot be considered
as zero-shot. This can be attributed to the fact that, although trained without the classes/labels, our teacher model BEiTv2 @beitv2
has been trained on ImageNet-1K. The knowledge contained in the teacher's representations of the $mono(["I_CLS"])$ token could therefore
leak information about the images of ImageNet-1K to the student. In order for it to be a true zero-shot approach, the teacher would have to be
trained on a dataset that is disjoint from ImageNet-1K.

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
      (Left) Accuracy of S-SMKE on the validation set of ImageNet-1k, using CLIP zero-shot classification, compared to CLIP @clip
      and Visual N-Grams @visual_n_grams. The latter was developed as a first proof-of-concept of zero-shot image classification
      in 2017 @visual_n_grams @clip. We denote a full zero-shot approach with $dagger$.
      (Right) Development of the training time for the different approaches. Adding additional parameters through Self-Attention (SHRe#sub[T])
      and a large self-supervised teacher (S-SMKE) increases the training time by more than two hours.
    ]
) <imagenet_1k_classification_S_SMKE>
#show table: set text(12pt)