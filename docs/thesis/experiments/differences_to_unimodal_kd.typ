=== Challenges of Self-Supervision <multimodal_knowledge_distillation_challenges>

Our approach differs to that of SHRe @shre in that we will make use of a self-supervised teacher model.
The consequence is that the teacher's prediction for a given sample is not a probability distribution over a set of classes, but merely
a representation of the input sample. As mentioned in @shre_section, a probabilty distribution over a set of classes is to some extent
independent of the input modality, as each class can be seen as a semantic concept that can be present in both images and text.
SHRe works well with a supervised image teacher model, as the probability distribution over the classes of an image can somewhat describe
the content of the image's caption. Examples are shown in TODO.

For a unimodal self-supervised model however, this probability distribution does not exists, raising the question
which training objective to use when using a self-supervised teacher.
In preliminary experiments on unimodal knowledge distillation, a self-supervised teacher did not pose a problem, as both the
teacher and student received the same input, and the latter was able to regress all time steps of the teacher model.
However, this was only possible because the teacher and student received exactly the same input: For a given time step, 
the patch or text token at that time step was the same for both models, allowing the student to learn the teacher's
representation for each patch or text token, respectively. Since we predict the output of an image teacher model, in which ever form it may be,
the aforementioned still holds true when the multimodal model receives an image as the input. The output will be a representation
for each image patch, which is also the prediction of the teacher model, allowing for the same approach used in unimodal knowledge distillation,
see @unimodal_kd_vision.

When the multimodal model receives a text as the input, the teacher's prediction is still a representation of the image, and not the text
(the image's caption). This poses the following problems:

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
min_(bold(h)^s_(v, L_s, mono(["I_CLS"])))||bold(h)^s_(v, L_s, mono(["I_CLS"])) - bold(h)^t_(v, L_t, mono(["I_CLS"]))||
$

This forces the student to push its representation of the $mono(["I_CLS"])$ token as close as possible to the teachers representation of 
the same token. Most importantly, the student can also be trained to push the representation of the caption $mono(["T_CLS"])$ as close as possible
to the representation of the image $mono(["I_CLS"])$ produced by the teacher:

$
min_(bold(h)^s_(w, L_s, mono(["T_CLS"])))||bold(h)^s_(w, L_s, mono(["T_CLS"])) - bold(h)^t_(v, L_t, mono(["I_CLS"]))||
$

An illustration of the problem posed by the misalignment of time steps and the solution of regressing the global representation of the image
is shown in @mm_kd_cls_token.

The combined training objective when regressing only global information is then:

$
min_(bold(h)^s_(v, L_s, mono(["I_CLS"])), bold(h)^s_(w, L_s, mono(["T_CLS"])))
||bold(h)^s_(v, L_s, mono(["I_CLS"])) - bold(h)^t_(v, L_t, mono(["I_CLS"]))|| +
||bold(h)^s_(w, L_s, mono(["T_CLS"])) - bold(h)^t_(v, L_t, mono(["I_CLS"]))||
$

While this objective in theory forces the student to output the same global representation for an image and its caption as the teacher,
it requires the teacher to produce a representation of the $mono(["I_CLS"])$ token that is abstract enough to also describe the content of the caption.
Conretely, the representation of the $mono(["I_CLS"])$ token should *not* contain any image-specific information, as this would make it impossible
for the student to align the representation of the caption with that of the image. It is not possible the extract any image-specific information
like the pixel values of the exact position of an object in the image from the caption of the image. Consequently, whether the representation of the
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
