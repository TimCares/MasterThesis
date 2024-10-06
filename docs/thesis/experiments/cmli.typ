=== Target Cross-Modal Late Interaction <target_cross_modal_late_interaction>
==== Cross-Modal Late Interaction <cross_modal_late_interaction>
Until now, we used the global text and image representations, given by $mono(["T_CLS"])$ and $mono(["I_CLS"])$, for
contrastive learning.
This has the disadvantage that only global information is utilized, and fine-grained, token/patch-specific, information is not considered.
This can make retrieval difficult, especially if real-world concepts described by an image and a text differ
in small, yet important details. An example of this can be seen in @coco25k_retrieval_examples, where multiple retrievals
are incorrect, even though they are semantically very similar to the query. The differences between the query and the retrieved samples
are often so small that they are not captured by the global representations.
To address the issue of fine-grained alignment, the authors of FILIP @filip
introduced Cross-Modal Late Interaction (CMLI) for contrastive learning, which led to improvements in retrieval performance.

As shown in @cmli, no cosine similarity between $mono(["T_CLS"])$ and $mono(["I_CLS"])$ is computed, but instead the cosine 
similarity between all image patches $[bold(h)_(v, l, k)]_(1 lt.eq k lt.eq N)$ and text tokens $[bold(h)_(w, l, j)]_(1 lt.eq j lt.eq M)$, 
with $N$ being the number of image patches, and $M$ being the number of text tokens. 
Specifically, $N$ and $M$ denote the number of patches/tokens in a sequence that are not the cls token
($mono(["I_CLS"])$/$mono(["T_CLS"])$) or padding token ($mono(["PAD"])$) @filip. The choice to exclude
padding tokens is obvious, as they do not carry any semantic information. The cls token is excluded, as it is not specific
to any token/patch, and is used to represent
global information. We additionally exclude the end-of-sequence token ($mono(["EOS"])$), as it is not specific to any token/patch either.
The result is the cosine similarity between all image patches and text tokens of an image-text pair.

#figure(
  image(
  width: 75%,
  "../figures/cmli.png"),
  caption: [For a token/patch, CMLI finds the timestep with the highest semantic match from the other modality. This enables the model to associate small details of image and text with each other. Notice how through the $max$-operation patches containing grass are always associated with the word "grass", and the words "sheep" and "head" are matched with the head of the sheep (associations created through $max$ are shown in (2)). The cosine similarity is then the average of all associations between an image-text pair. Figure inspired and adapted from @filip.
  ],
) <cmli>

The next step is to find for each image patch $k$ the text token with the maximum cosine similarity to this image patch.

$
m_k^("i2t") = op("argmax", limits: #true)_(1 lt.eq j lt.eq M) [bold(h)_(v, l, k)] [bold(h)_(w, l, j)]^T
$

Likewise, for each text token $j$, we get the image patch with the maximum cosine similarity to this text token
$
m_j^("t2i") = op("argmax", limits: #true)_(1 lt.eq k lt.eq N) [bold(h)_(v, l, k)] [bold(h)_(w, l, j)]^T
$

This has an interesting effect: For each image patch the semantically most similar text token is found, and vice versa 
for each text token - the result of this operation can be seen in (2) of @cmli.
Consequently, the model will be able to associate small details of an image with individual text tokens, and vise versa. 
The actual cosine similarity between an image-text pair is then the average of all associations between an image patch and a text token.

$
s^("i2t")_(bold(H)_(v, l),bold(H)_(w, l)) = 1/N sum_(k=1)^N [bold(h)_(v, l, k)] [bold(h)_(w, l, m_k^("i2t"))]^T
$

$
s^("t2i")_(bold(H)_(v, l),bold(H)_(w, l)) = 1/M sum_(j=1)^M [bold(h)_(v, l, m_j^("t2i"))] [bold(h)_(w, l, j)]^T
$

Here, for one image-text pair, $m_k^("i2t")$ denotes the index of the text token with the highest cosine similarity to image patch $k$,
and $m_j^("t2i")$ the index of the image patch with the highest cosine similarity to text token $j$. $s^("i2t")_(bold(H)_(v, l),bold(H)_(w, l))$
denotes the the similarity score between an image representation $bold(H)_(v, l)$ and text representation $bold(H)_(w, l)$.
Vice versa, $s^("t2i")_(bold(H)_(v, l),bold(H)_(w, l)) $ denotes the similarity score between a text representation $bold(H)_(w, l)$ and an
image representation $bold(H)_(v, l)$. $l$ can denote any layer of the model, but is usually the
last layer, as the representations are most meaningful there.

In contrast to the standard contrastive learning, this similarity measure is not necessarily symmetric,
as e.g. a text token might have a maximum cosine similarity to another image patch than the image patch that has its maximum
similarity to the text token @filip.
The process in illustrated in @cmli.

While this approach allows for a fine-grained alignment of image and text, its practical implementation is very computationally
and memory intensive. For standard constrastive learning is is sufficient to compute the cosine similarity of the global representation
(cls token) between every possible image-text pair in a batch. If negative examples are gathered from all devices, then
the number of dot products to compute is defined as $(B*P)^2$, with $B$ being the batch size per device, 
and $P$ being the number of devices (in our case GPUs). As we use a batch size of $B=256$ per device, and use $P=2$ GPUs,
the number of dot products to compute is $(256*2)^2=262,144$. Considering that we perform this efficiently using matrix multiplication,
and the embedding size is 768 with float32 precision, we already need $262,144 * 768 * 4 "bytes" = 805.31 "MB" $ of GPU memory, which
is still manageable, since we have around 2 GB of GPU memory remaining for a step.

However, with CMLI, we need to compute the similarity between all possible image-text pairs, where the similarity
for one pair requires the computation of the cosine similarity between all image patches and text tokens of the image-text pair.
With a maximum text sequence length of 64 tokens @beit3
and an image sequence length of 197, the number of dot products to compute for just
one image-text pair is $197*64=12,608$. With a batch size of 256 per device, and 2 GPUs, the number of dot products
increases from $262,144$ to $256*2*12,608=6,455,296$. Even if the embedding dimension is reduced to 256, which is a simplification
done in FILIP @filip, we need $6,455,296 * 256 * 4 "bytes" = 6.61 "GB"$ of additional GPU memory just to store the result.
This is not feasible in our setup.

==== Method <target_cmli_method>

What is feasible though, is to apply CMLI to a setting where the computation is more lightweight.
As the driving factor behind the memory requirements in contrastive learning is that the similarity between all
possible image-text pairs in a batch is computed, this is removed when just the similarity between
positive pairs is computed, which is what we call Target-CMLI.

Target-CMLI is not used for contrastive learning, but rather to alleviate the problem of regressing
patch-level information of the teacher model.
Recall that in the current setting it is merely possible
to regress the global representations of the teacher model, and not the patch-level information.
This is because the teacher model outputs patch-level predictions for the image modality, but not for the text modality.
Consequently, the student model can replicate the output of the teacher model for the image modality, but not for the text modality,
as it is not possible to assign a text token to a specific image patch.
This is illustrated in @mm_kd_cls_token, and discussed in @multimodal_knowledge_distillation_challenges.

However, as seen in @cmli, when computing the cosine similarity between all image patches and text tokens of an image-text pair,
and selecting the $op("argmax")$ over all image patches with respect to a text token, then a corresponding, or at least similar,
image patch can be found for each text token ((2) of @cmli).
This means that the student model can replicate the output of the teacher model for the text modality by first selecting the most
similar image patch for each text token, and then minimizing the Mean Squared Error (MSE) between the teacher's representation of 
the selected image patch and the representation of the text token.

For the patch-level image representation of the student model that means that we can now also regress the patch-level information
of the teacher, and not only the global information. This was also possible in all previous experiments, as the order of the
image patches does not change between student and teacher image representations. However,
this would heavily bias the parameters of the shared Transformer block towards the image modality.

The definition of the loss changes as follows:

$
cal(L)_("KD")^("i2i") = 
cal(L)_"MSE" (bold(H)^s_(v, K), bold(H)^t_(v, L_t)) =
sum_(n=1)^N ||bold(h)^s_(v, K, n) - bold(h)^t_(v, L_t, n)||^2_2
$

For a given text representation we first need to find $m_j^("t2i")$ for each text token $j$, and then define the loss as:

$
cal(L)_("KD")^("t2i") =
cal(L)_"MSE" (bold(H)^s_(w, K), bold(H)^t_(v, L_t)) =
sum_(z=1)^M ||g(bold(h)^s_(w, K, z)) - g(bold(h)^t_(v, L_t, m_z^("t2i")))||^2_2
$

We denote $g(dot)$ as a linear projection to reduce the dimensionality of the image representation from the teacher, and the text representation
from the student, to 32. This is done to (1) reduce memory consumption when computing the similarity between all image patches and text tokens
for each image-text pair, and (2) to reduce the information that can be expressed by $g(bold(x))$. We motivate (2) by the fact
that matching individual image patches to text tokens is a very fine-grained task, and we want to avoid having pixel-level information
in the patch representations of the teacher model, which would, despite the matching via $op("argmax")$, cause problems with predicting those
representations, as we can't extract those pixel-level information from the text tokens.
The total knowledge distillation loss remains the same, and is the mean of the two losses:

$
cal(L)_("KD") = 1/2 * (cal(L)_("KD")^("i2t") + cal(L)_("KD")^("t2i"))
$

We use the mean instead of the sum, as we also use the contrastive loss, and we want both losses to have the same weight.
We find $m_k^("t2i")$ using an embedding dimension of 32, which is achieved through the projection $g(dot)$.
The hidden size of the model remains at 768.

What makes the implementation of Target-CMLI feasible is that we only need to find the most similar image patch for each text token
in a positive pair,
and not all possible image-text pairs in a batch. For a per-device batch size of 256, Target-CMLI requires $12,608*256=3,227,648$ dot products to compute.
With an embedding dimension of 32 just $3,211,264*32*4 "bytes" = 411 "MB"$ of additional GPU memory
is required. This is feasible in our setup.

==== Results <target_cmli_results>

The results, as seen in @t_cmli_results, can be considered as disappointing. We lose more than 2 percentage points in average retrieval
on both MSCOCO and Flickr30K, while the performance on ImageNet-1K decreases by nearly 4 percentage points.

#show table: set text(8pt)
#figure(
  table(
  columns: 4,
  stroke: none,
  table.hline(),
  table.header(
    table.cell(rowspan: 2, colspan: 1, align:horizon, [T-CMLI]),
    table.cell(rowspan: 2, colspan: 1, align:horizon, [ImageNet-1K]),
    table.cell(colspan: 2, align:horizon, [Retrieval]),
    [MSCOCO],
    [Flickr30K],
  ),
  table.hline(stroke: .6pt),
  [$times$], [*30.3*], [*67.2*], [*78.29*],
  [$checkmark$], [26.7], [65.68], [76.2],
  table.hline(),
),
    caption: [Comparison of adjusting the loss function $cal(L)_("KD")$ from regressing the $mono(["I_CLS"])$ of the teacher
    model to regressing the patch-level information of the teacher model using Target-CMLI. We observe a decrease in all metrics.]
) <t_cmli_results>
#show table: set text(12pt)

A look at a visualization (@t_cmli_examples) of matches between text tokens and their top-5 most similar image patches under cosine similarity
reveals that while there are some cases where the token has the highest similarity to an actual image patch belonging the the correct
object, the matching is far from perfect, and matches are often either completely unrelated, or scattered across almost random regions.
A matching that would be expected from a successful approach is illustrated by the authors of FILIP @filip, and can be seen in
@filip_cmli_examples. While the exaples of FILIP are based on CMLI for contrastive learning, and not our Target-CMLI for knowledge distillation,
the principle of matching text tokens to image patches remains the same, and the quality of the matches is expected to be similar.

In general, we observe that the patch with the highest similarity for a text token is often a patch completely unrelated to the token.
While it is true that, thanks to self-attention, the representation of an image patch not only contains information about the patch itself,
but also about patches it considers important, the self-attention map of the image patch with the highest similarity to a text token, which
is the matched image patch, does not show any clear signs that it contains aggregated information from patches that are part of the object
the text token describes. If that were the case, then the self-attention map of the matched image patch would show a clear focus on the object
the text token describes/represents. This is illustrated in @t_cmli_examples, where for each example, the left image shows the top 5 most similar
image patches for a text token under cosine similarity, and the right image shows the self-attention map of the image patch with the highest
similarity to the text token, i.e. the image patch $m_j^("t2i")$ for a text token $j$.
For the aforementioned, we observe a very inconsistent matching between text tokens and image patches. Especially examples "planes" and
"birds" show that a mismatch is not rare.

As a reference, we provide the self-attention map, w.r.t. the $mono(["I_CLS"])$ token, of the final Transformer layer
from the self-supervised image model DINO @dino in @dino_cls_attn_examples (Appendix). We would expect the self-attention map of
the matched image patch $m_j^("t2i")$ to the example text token $j$ (@t_cmli_examples) to be similar to that of DINO. However, this
is not always the case.

Moreover, the top 5 matched image patches are often scattered across the image, and not focused on a specific region, underlining that
the approach does not work as intended. Instead, a result similar to that of FILIP in @filip_cmli_examples is expected, where the matched
image patches are focused on the object the text token describes, and lie next to each other. Our results are much more similar to
that of CLIP, also illustrated by the authors of FILIP (@filip_cmli_examples), leading us to believe that a missing cross-modal
attention might be the reason.

We suspect this, because FILIP @filip, like VLMo @vlmo and BEiT-3 @beit3, uses cross-attention to align text and image. This allows
individual text tokens to attend to specific image patches, and vice versa, leading to e.g. text tokens to be able to "locate"
the objects they describe in an image. Cross-Modal Late Interaction then allows to apply this "locating of the matched image patch"
to contrastive learning and retrieval.

CLIP however, like our model, does not use cross-attention, and only uses global representations for contrastive learning. Therefore,
CLIP is not able to find relationships between individual text tokens and image patches, leading to a scattered matching, as seen in
@filip_cmli_examples.

While we do not use CMLI for contrastive learning, but for knowledge distillation, the same principles apply. Even worse, since the image patches
that we try to match to text tokens are not even the result of the student model, but of the teacher model, the model does not have
the possibility to somehow learn patch-level representations that are suitable for matching to text tokens.
Consequently, there really is no guidance for the model to learn matching its text tokens to the right image patches of the teacher model.

#figure(
  image(
  width: 100%,
  "../figures/t_cmli_examples.png"),
  caption: [
    Visualization of selected text tokens from the image's caption
    and the tokens top 5 most similar image patches under cosine similarity (left). The self-attention
    map (right) of the image patch with the highest similarity to the text token is shown next to the original image.
    We observe that matched image patches are often scattered across the image, are not part of the object the text token represents,
    and their self-attention map, indicating the composition of the token, does not always show a clear focus on the object the text token describes.
    The title shows a text token $j$, and the self-attention map with respect to image patch $m_j^("t2i")$ (i.e. the matched one).
    Image-text pairs taken from COCO test set @coco.
],
) <t_cmli_examples>


Lastly, @t_cmli_examples shows the problem based on text tokens that have a real-world meaning, and therefore a counterpart in images:
The text token "plane" can also be present in an image as an actual plane. However, text tokens like "a", "the", and even a
full stop ".", which is a valid token, are also matched to image patches. They are merely fill words or grammatic nuances in a sentence,
and do not carry any semantic information that can be mapped to an image patch. Because CMLI works over all text tokens, 
and only few text tokens actually have an object-level counterpart in images, like "plane", most of the matchings between text tokens
and image patches are not meaningful to begin with. Recall that this was one of the reasons why we decided to only regress the teacher's
$mono(["I_CLS"])$ token (see @multimodal_knowledge_distillation_challenges).

To come to a conclusion, even though Target-CMLI seems to be a promising approach to alleviate the mismatch between text tokens and image patches,
especially considering that some examples in @t_cmli_examples show a self-attention map that is focused on the object the text token describes,
the results are far from consistent, and are constrained to only a few text tokens that have a real-world counterpart in images, and even
this are not reliable.
