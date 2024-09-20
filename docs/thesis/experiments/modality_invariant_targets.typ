=== Modality-Invariant Targets <modality_invariant_targets>
Throughout the previous experiments, we have seen that the misalignment between image patches and text tokens leads to problems when regressing the image features
of the teacher. While we are already exceeding the performance of the supervised teacher by predicting the $mono(["I_CLS"])$ of the teacher,
there is still room for improvement.
A glance at the loss $cal(L)_"KD"$, which is now again the previous one (defined again in @kd_loss_mse_2 for ease of access),
since Target-CMLI (@target_cmli_method) proved to be ineffective,
shows that the loss for the image features is clearly lower than that for the text features. This indicates that the $mono(["I_CLS"])$ of the teacher still contains
image-specific information. If that were not the case, the loss for both components, i.e. $cal(L)^v_"KD"$ and $cal(L)^w_"KD"$, would be similar.
Consequently, we aim to introduce a modality-invariant target loss that is less affected by the image-specific information in the $mono(["I_CLS"])$ of the teacher.

$
cal(L)_("KD") &= \ 
1/2 * cal(L)_("KD")^v &+ 1/2 * cal(L)_("KD")^w = \
1/2 * ||bold(h)'''^s_(v, K, mono(["I_CLS"])) - bold(h)^t_(v, L_t, mono(["I_CLS"]))||^2_2 &+ 
1/2 * ||bold(h)'''^s_(w, K, mono(["T_CLS"])) - bold(h)^t_(v, L_t, mono(["I_CLS"]))||^2_2
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

==== Contrastive Target Loss <contrastive_target_loss>
We identify the MSE loss, used as a criterion for knowledge distillation, as unfavourable, as it enforces the student representation of the caption
$bold(h)'''^s_(w, K, mono(["T_CLS"]))$ to be the same as the teacher representation of the image $bold(h)^t_(v, L_t, mono(["I_CLS"]))$. The loss
only becomes zero if both are identical. This is less of a problem for $bold(h)'''^s_(v, K, mono(["I_CLS"]))$, as this is the image representation
of the student, which can contain the image-specific information. However, the student is not able to extract the image-specific information from the caption.

To address this issue, we propose a contrastive target loss, which is inspired by contrastive learning. This loss collects all teacher representations
from the current batch ${bold(h)^t_((v, L_t, mono(["I_CLS"])), b)}^B_(b=1)$, and computes the cosine similarity between those representations 
and the student representation of a candidate caption $bold(h)'''^s_(w, K, mono(["T_CLS"]))$. Similar to the contrastive loss, we aim to maximize the similarity
between the student representation of caption $bold(h)'''^s_((v, K, mono(["I_CLS"])), i)$
and the teacher representations of image $bold(h)^t_((v, L_t, mono(["I_CLS"])), i)$ in the batch, while minimizing the similarity between the student
representation of caption $bold(h)'''^s_((v, K, mono(["I_CLS"])), i)$
and the teacher representations of all other images $bold(h)^t_((v, L_t, mono(["I_CLS"])), j), j eq.not i$
in the batch.

This has the advantage that we are now focusing less on
making $bold(h)'''^s_((v, K, mono(["I_CLS"])), i)$ and $bold(h)^t_((v, L_t, mono(["I_CLS"])), i)$ identical,
but rather on what representations match, and which do not
This puts more emphasis on
the relative similarity with respect to the similarity to other images in the batch. Maximizing the cosine similarity is further a less strict criterion than
minimizing the MSE, as the cosine similarity only requires both representations to be in the same direction, but not necessarily to be identical.

Like in the contrastive loss, if the representations are good enough
$bold(h)'''^s_((v, K, mono(["I_CLS"])), i)$ and $bold(h)^t_((v, L_t, mono(["I_CLS"])), i)$ will have more in common than
$bold(h)'''^s_((v, K, mono(["I_CLS"])), i)$ and $bold(h)^t_((v, L_t, mono(["I_CLS"])), j)$ for $i \neq j$, so
the student representation of the caption $i$ is more likely match the teacher representations of the image $i$. 

==== Implementation <contrastive_target_loss_implementation>
The implementation closely follows that of the image-text contrast of @vision_language_contrast.
We concatenate all teacher representations of the image $mono(["I_CLS"])$ in the batch to a tensor, and all student representations
of the caption $mono(["T_CLS"])$ to another tensor:

$
bold(I)_t = [bold(h)^t_((v, L_t, mono(["I_CLS"])), 1), bold(h)^t_((v, L_t, mono(["I_CLS"])), 2), ..., bold(h)^t_((v, L_t, mono(["I_CLS"])), B')]
in RR^(B' times D) \
bold(T)_s = [bold(h)'''^s_((w, K, mono(["T_CLS"])), 1), bold(h)'''^s_((w, K, mono(["T_CLS"])), 2), ..., bold(h)'''^s_((w, K, mono(["T_CLS"])), B')]
in RR^(B' times D)
$

Additionally, we also collect all image representations of the student $mono(["I_CLS"])$ in the batch to a tensor:

$
bold(I)_s = [bold(h)^s_((v, K, mono(["I_CLS"])), 1), bold(h)^s_((v, K, mono(["I_CLS"])), 2), ..., bold(h)^s_((v, K, mono(["I_CLS"])), B')]
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
delta(bold(x)) &= bold(x) / (||bold(x)||_2) in RR^D
$

Notice how the contrastive target loss is computed both between the student representation of the caption and the teacher representations of the image,
which is the important part we covered in the previous section, and between the student representations of the image and the teacher
representations of the image. The latter is the image-to-image part of the knowledge distillation loss, which does not suffer from
image-specific information in the teacher representation of the image, but we also adapt it to the contrastive loss for consistency.

Analogous to image-text contrast, we define the loss, which is just cross-entropy, as follows:

$
cal(L)^"t2i"_"KD" &= 1/B' sum_(i=1)^(B') -log exp(L^w_(i, i)) / (sum_(k=1)^(B') exp(L^w_(i, k))) \
cal(L)^"i2i"_"KD" &= 1/B' sum_(i=1)^(B') -log exp(L^v_(i, i)) / (sum_(k=1)^(B') exp(L^v_(i, k))) \
cal(L)_"KD" &= 1/2 * cal(L)^"t2i"_"KD" + 1/2 * cal(L)^"i2i"_"KD"
$



==== Memory Bank <contrastive_target_loss_mb>
In @memory_bank_section, we evaluated the possibility of storing representations, produced by the student, from previous batches in a memory bank.
The idea was that since the contrastive loss requires a large number of negative samples to be effective, we could use
representations from previous batches as additional negative samples. However, we found that using representations from previous batches
leads to a significant performance drop, as the representations were inconsistent.

Fortunately, the contrastive target loss does not suffer from outdated representations. This is because the representations we compare the
student representation to are from the teacher. The teacher's weights are frozen during training, meaning that the teacher's representations
are consistent over the entire training process. Therefore, all negative examples that are used in the contrastive target loss are consistent
with each other, meaning we can savely use a simple memory bank to store the teacher representations from previous batches.
Workarounds like a momentum encoder (@momentum_encoder) or proximal regularization of the features (@original_memory_bank) are not necessary.

The formulation of the loss does not change, but only the concatenated teacher representations.

$
bold(I)'_t &= bold(I)_t||bold(V) = \
[bold(h)^t_((v, L_t, mono(["I_CLS"])), 1), ..., bold(h)^t_((v, L_t, mono(["I_CLS"])), B'),&
bold(v)^t_((v, L_t, mono(["I_CLS"])), 1), ..., bold(v)^t_((v, L_t, mono(["I_CLS"])), G)]
in RR^((B'+G) times D)
$

$
bold(L)^w = delta(bold(T)_s) delta(bold(I)'_t)^T in RR^(B' times G) \
$

We denote $bold(v)^t_((v, L_t, mono(["I_CLS"])), i)$ as the teacher representation of the image $i$ from the memory bank, so from a previous batch,
and $G$ as the number of representations stored in the memory bank, i.e. the size. We set $G=65536$ in our experiments, which we orientate on the
ideal size found by MoCo @moco for contrastive learning (see @moco_vs_mb).