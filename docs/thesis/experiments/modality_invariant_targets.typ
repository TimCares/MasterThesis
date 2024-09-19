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


==== Memory Bank <contrastive_target_loss_mb>