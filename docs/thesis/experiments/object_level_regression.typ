The issue arises because the teacher's representation $bold(h)^t_(v, L_t, mono(["I_CLS"]))$ still contains image-specific information.
This means the student's image representation $bold(h)'''^s_(v, K, mono(["I_CLS"]))$ will always be closer to
$bold(h)^t_(v, L_t, mono(["I_CLS"]))$ than the caption representation $bold(h)'''^s_(w, K, mono(["T_CLS"]))$ (see @mse_kd_loss_comparison),
because only the image input allows the student to extract this image-specific information, not the caption.
As a result, it is easier for the student to regress the teacher's image representation with its own image representation
than with its caption representation.


The reason why we know that the teacher's representation $bold(h)^t_(v, L_t, mono(["I_CLS"]))$ still contains image-specific information
is the comparison between the training loss of $cal(L)_("KD")^v$ and $cal(L)_("KD")^w$ in @mse_kd_loss_comparison.
It is observable that the loss $cal(L)_("KD")^v$ is significantly lower than $cal(L)_("KD")^w$, which indicates that the student
is able to regress the teacher's image representation $bold(h)^t_(v, L_t, mono(["I_CLS"]))$ better
with its own image representation $bold(h)'''^s_(v, K, mono(["I_CLS"]))$ than with its caption representation $bold(h)'''^s_(w, K, mono(["T_CLS"]))$.
If the teacher's representation $bold(h)^t_(v, L_t, mono(["I_CLS"]))$ would not contain any image-specific information, say,
it encodes the information "dog" for an image of a dog, the student should be able to regress this representation with both
of its representations equally well. However,
this is not the case.