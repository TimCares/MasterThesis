=== Contrastive Distillation <modality_invariant_targets>
Throughout the previous experiments we have seen that the misalignment between image patches and text tokens leads to problems when regressing the image features
of the teacher. While we are already exceeding the performance of the supervised teacher by predicting the $mono(["I_CLS"])$
token of the teacher, there is still room for improvement.
A glance at the loss $cal(L)_"KD"$, which is still the one without Target-CMLI (defined again in @kd_loss_mse_2 for ease of access),
shows that the loss for the image features is clearly lower than that for the text features. This indicates that the $mono(["I_CLS"])$
token of the teacher still contains
image-specific information. If that were not the case, the loss for both components, i.e. $cal(L)^v_"KD"$ and $cal(L)^w_"KD"$, would be similar.
Consequently, we aim to introduce a modality-invariant target loss that is less affected by the image-specific information in the $mono(["I_CLS"])$ token of the teacher.

$
cal(L)_("KD") &= \ 
1/2 * cal(L)_("KD")^v &+ 1/2 * cal(L)_("KD")^w = \
1/2 * ||bold(h)^s_(v, mono(["I_CLS"])) - bold(h)^t_(v, L_t, mono(["I_CLS"]))||^2_2 &+ 
1/2 * ||bold(h)^s_(w, mono(["T_CLS"])) - bold(h)^t_(v, L_t, mono(["I_CLS"]))||^2_2
$ <kd_loss_mse_2>

#figure(
  image(
  width: 75%,
  "../figures/kd_loss_comparison.png"),
  caption: [
    Training loss for the image and text component of the $cal(L)_"KD"$ loss. The image component $cal(L)^v_"KD"$ shows a significantly lower loss
    compared to the text component $cal(L)^t_"KD"$, indicating that the $mono(["I_CLS"])$ of the teacher still contains image-specific information.
  ],
) <kd_loss_comparison>

==== Contrastive Loss <contrastive_target_loss>
We identify the MSE loss, used as a criterion for knowledge distillation, as unfavorable, as it enforces the student representation of the caption
$bold(h)^s_(w, mono(["T_CLS"]))$ to be the same as the teacher representation of the image $bold(h)^t_(v, L_t, mono(["I_CLS"]))$. 

Recall that the student representation shown above is used to predict the teacher representation of the image, and it _not_ used
for image-text retrieval/alignment. However, we still call it "student representation of the caption" for simplicity.

The loss
only becomes zero if both are identical. This is less of a problem for $bold(h)^s_(v, mono(["I_CLS"]))$, as this is the image representation
of the student, which can contain image-specific information. However, the student is not able to extract the image-specific information from the caption, so they can't be encoded in $bold(h)^s_(w, mono(["T_CLS"]))$.

To address this issue, we propose a contrastive distillation loss, which is inspired by contrastive learning. This loss collects all teacher representations
from the current batch ${bold(h)^t_((v, L_t, mono(["I_CLS"])), b)}^B_(b=1)$, and computes the cosine similarity between those representations 
and the student representation of a caption $bold(h)^s_(w, mono(["T_CLS"]))$. Similar to the contrastive loss, we aim to maximize the similarity
between the student representation of caption $bold(h)^s_((v, mono(["I_CLS"])), i)$
and the teacher representations of image $bold(h)^t_((v, L_t, mono(["I_CLS"])), i)$ in the batch, while minimizing the similarity between the student
representation of caption $bold(h)^s_((v, K, mono(["I_CLS"])), i)$
and the teacher representations of all other images $bold(h)^t_((v, L_t, mono(["I_CLS"])), j), j eq.not i$
in the batch.

This has the advantage that we are now focusing less on
making $bold(h)^s_((v, K, mono(["I_CLS"])), i)$ the "same" as $bold(h)^t_((v, L_t, mono(["I_CLS"])), i)$,
but rather on which representations match, and which do not.
This puts more emphasis on
the relative similarity between both representations with respect to the similarity to other images in the batch
($bold(h)^t_((v, L_t, mono(["I_CLS"])), j), j eq.not i$). Maximizing the cosine similarity is further a less strict criterion than
minimizing the MSE, as the cosine similarity only requires both vectors/representations to point in the same direction, but not necessarily to be identical.

Like in the contrastive loss, if the representations are good enough, then
$bold(h)^s_((v, mono(["I_CLS"])), i)$ and $bold(h)^t_((v, L_t, mono(["I_CLS"])), i)$ will have more in common than
$bold(h)^s_((v, mono(["I_CLS"])), i)$ and $bold(h)^t_((v, L_t, mono(["I_CLS"])), j)$, with $j eq.not i$. Consequently,
the student representation of the caption $i$ is more likely match the teacher representations of the image $i$. 

==== Implementation <contrastive_target_loss_implementation>
The implementation closely follows that of the image-text contrast of @vision_language_contrast.
We concatenate all teacher representations of the image ($mono(["I_CLS"])$ token) in the batch to a tensor,
and all student representations
of the caption ($mono(["T_CLS"])$ token) to another tensor:

$
bold(I)_t = [bold(h)^t_((v, L_t, mono(["I_CLS"])), 1), bold(h)^t_((v, L_t, mono(["I_CLS"])), 2), ..., bold(h)^t_((v, L_t, mono(["I_CLS"])), B')]
in RR^(B' times D) \
bold(T)_s = [bold(h)^s_((w, mono(["T_CLS"])), 1), bold(h)^s_((w, mono(["T_CLS"])), 2), ..., bold(h)^s_((w, mono(["T_CLS"])), B')]
in RR^(B' times D)
$

Additionally, we also collect all image representations of the student ($mono(["I_CLS"])$ token) in the batch to a tensor:

$
bold(I)_s = [bold(h)^s_((v, mono(["I_CLS"])), 1), bold(h)^s_((v, mono(["I_CLS"])), 2), ..., bold(h)^s_((v, mono(["I_CLS"])), B')]
in RR^(B' times D)
$

We define $B'$ as the combined batch size over all devices, as we can gather all representations from all devices (see @larger_batch_sizes_ddp).
It therefore holds that $B' = B * P$, where $P$ is the number of devices. In our case we use $P=2$ devices and $B=256$
samples per device, resulting in $B'=2*256=512$.

The cosine similarity can again be computed efficiently using matrix multiplication of the normalized representations:

$
bold(L)^w = delta(bold(T)_s) delta(bold(I)_t)^T in RR^(B' times B') \
bold(L)^v = delta(bold(I)_s) delta(bold(I)_t)^T in RR^(B' times B') \
$

Here, $delta$ denotes the normalization:

$
delta(bold(X)) &= [delta(bold(x)_1), delta(bold(x)_2), ..., delta(bold(x)_B')] in RR^(B' times D) \
delta(bold(x)) &= bold(x) / (||bold(x)||_2)
$

Notice how the similarity is both computed between the student representation of the caption and the teacher representations of the image,
which is the important part, and between the student representations of the image and the teacher
representations of the image. The latter is the image-to-image ($"i2i"$) part of the knowledge distillation loss, which does
not suffer from the presence of image-specific information in the teacher representation of the image. Still, we also adapt
it to the contrastive loss for consistency with the text-to-image ($"t2i"$) loss.

Analogous to image-text contrast, we define the loss, which is just cross-entropy, as follows:

$
cal(L)^"t2i"_"KD" &= 1/B' sum_(i=1)^(B') -log exp(L^w_(i, i)) / (sum_(k=1)^(B') exp(L^w_(i, k))) \
cal(L)^"i2i"_"KD" &= 1/B' sum_(i=1)^(B') -log exp(L^v_(i, i)) / (sum_(k=1)^(B') exp(L^v_(i, k))) \
cal(L)_"KD" &= 1/2 * cal(L)^"t2i"_"KD" + 1/2 * cal(L)^"i2i"_"KD"
$ <contrastive_target_loss_eq>

We rename the components of the loss from $cal(L)^"w"_"KD"$ to $cal(L)^"t2i"_"KD"$, and
$cal(L)^"v"_"KD"$ to $cal(L)^"i2i"_"KD"$, in order to reflect the text-to-image and image-to-image parts
of contrastive learning, respectively.

When comparing our current configuration to our previous best, which was reached when introducing a token-type embedding
after the modality-specific encoders (@token_type_embeddings), we find that while the performance
on MSCOCO and Flickr30K retrieval remains unchanged, the performance on ImageNet-1K improves by nearly 3 percentage points.
This improvement on ImageNet-1K is likely due to our use of CLIP zero-shot classification, which involves retrieving the
most similar class prototype using a contrastive loss (see @clip_zero_shot_section). Since we now employ a contrastive loss
for knowledge distillation, and the weights of the teacher model — responsible for generating one component of the contrastive distillation loss
(the teacher representations $bold(I)_t$ of the images) — were originally trained on ImageNet-1K, the student model's
learning method aligns more closely with CLIP-like zero-shot classification. This alignment is likely to enhance performance on ImageNet-1K.

#show table: set text(8pt)
#figure(
  table(
  columns: 4,
  stroke: none,
  table.hline(),
  table.header(
    table.cell(rowspan: 2, colspan: 1, align:horizon, [KD Loss]),
    table.cell(rowspan: 2, colspan: 1, align:horizon, [ImageNet-1K]),
    table.cell(colspan: 2, align:horizon, [Retrieval]),
    [MSCOCO],
    [Flickr30K],
  ),
  table.hline(stroke: .6pt),
  [MSE], [30.3], [*67.2*], [78.29],
  [Contrastive], [*33.0*], [67.15], [*78.3*],
  table.hline(),
),
    caption: [Replacing the MSE loss with the contrastive distillation loss improves the performance on ImageNet-1K by almost 3 percentage points,
    while matching the performance on MSCOCO and Flickr30K retrieval.]
) <mse_loss_vs_contrastive>

==== Memory Bank <contrastive_target_loss_mb>
In @memory_bank_section, we evaluated the possibility of storing representations from previous batches in a memory bank.
The idea was that since the contrastive loss requires a large number of negative samples to be effective, we could use
representations from previous batches as additional negative samples. However, we found that using representations from previous batches
leads to a significant performance drop, as the representations were inconsistent.

Fortunately, the contrastive distillation loss does not suffer from outdated representations. This is because the representations we compare the
student representation to are from the teacher. The teacher's weights are frozen during training, meaning that the teacher's representations
are consistent over the entire training process. Therefore, all negative examples that are used in the contrastive distillation loss are consistent
with each other, meaning we can safely use a simple memory bank to store the teacher representations from previous batches.
Workarounds like a momentum encoder (@momentum_encoder) or proximal regularization of the features (@original_memory_bank) are not necessary.

The formulation of the loss does not change, but only the concatenated teacher representations.

$
bold(I)'_t &= bold(I)_t||bold(V) = \
[bold(h)^t_((v, L_t, mono(["I_CLS"])), 1), ..., bold(h)^t_((v, L_t, mono(["I_CLS"])), B'),&
bold(v)^t_((v, L_t, mono(["I_CLS"])), 1), ..., bold(v)^t_((v, L_t, mono(["I_CLS"])), G)]
in RR^((B'+G) times D)
$

$
bold(L)^w &= delta(bold(T)_s) delta(bold(I)'_t)^T in RR^(B' times G) \
bold(L)^v &= delta(bold(I)_s) delta(bold(I)'_t)^T in RR^(B' times G)
$

We denote $bold(v)^t_((v, L_t, mono(["I_CLS"])), i)$ as the teacher representation of the image $i$ from the memory bank, so from a previous batch,
and $G$ as the number of representations stored in the memory bank, i.e. the size ($1 lt.eq i lt.eq G$). We set $G=65,536$ in our experiments, which we orientate on the
ideal size found by MoCo @moco for contrastive learning (see @moco_vs_mb).

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
    [S-SMKE], [51.66], [79.9], [88.66], [36.17], [66.55], [*78.28*], [64.5], [88.4], [93.0], [51.78], [78.54], [86.46],
    [S-SMKE#sub[CDL]], [52.68], [80.56], [88.3], [36.5], [66.58], [*78.28*], [69.0], [89.2], [94.3], [51.48], [79.18], [86.66],
    [S-SMKE#sub[CDL_MB]], [53.54], [81.1], [*89.52*], [35.65], [66.0], [77.77], [70.9], [92.1], [96.0], [52.72], [80.2], [87.46],
    table.hline(),
  ),
  caption: [
    A contrastive distillation loss with memory bank especially improves text retrieval, while the performance on COCO image retrieval
    degrades slightly. S-SMKE#sub[CDL] denotes the contrastive distillation loss without memory bank, and S-SMKE#sub[CDL_MB]
    denotes the contrastive distillation loss with memory bank.
  ],
)<image_text_retrieval_ctl_mb>
#show table: set text(12pt)

A detailed look on the retrieval performance in @image_text_retrieval_ctl_mb shows a gain especially on text retrieval tasks,
and we are able to increase text retrieval on Flickr30K by approx. 2-3 percentage points in each metric.
// This is surprising, as we would expect
// an increase in image retrieval performance instead. At its core, the contrastive distillation loss uses text-image retrieval,
// expressed through
// the loss $cal(L)^"t2i"_"KD"$ in @contrastive_target_loss_eq. This should have a positive effect on text-image retrieval
// in COCO and Flickr30K, but we instead observe a slight decrease in performance. Simultaneously, the performance
// on image-text retrieval increases consistently, even though
// the contrastive distillation loss does not use image-text retrieval. We would expect the opposite.
The performance on ImageNet-1K is increased by an impressive 4 percentage points to 37% (33% before).
Considering the increase in image-text retrieval on Flickr30K and COCO this is actually to be expected:
In CLIP zero-shot classification, a candidate image is compared to all class prototypes,
which have been created from text descriptions of the classes. The class prototype,
which is a text representation, with the highest similarity to the image is then chosen as the predicted class, which is essentially
image-text retrieval. Therefore, if the performance on image-text retrieval increases, we would expect an increase in performance on CLIP zero-shot ImageNet-1K classification.

Since this is our _final_ framework for S-SMKE, we show an illustration our our method for efficient vision-language
learning in @ev_lp.
