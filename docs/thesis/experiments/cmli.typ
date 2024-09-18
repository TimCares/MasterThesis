=== Target Cross-Modal Late Interaction <target_cross_modal_late_interaction>
==== Cross-Modal Late Interaction <cross_modal_late_interaction>
Until now, we used the global text and image representations $mono(["T_CLS"])$ and $mono(["I_CLS"])$ for
contrastive learning.
This has the disadvantage that only global information is utilized, and fine-grained, token/patch-specific, information is not considered.
This can make retrieval, and alignment in general, difficult, especially if real-world concepts described by an image and a text differ
in small, yet important, details. An example of this can be seen in (TODO: vis retrievals on full coco), where multiple retrievals
are incorrect, even though they are semantically very similar to the query. The differences between the query and the retrieved samples
are often so small that they are not captured by the global representations. This is one of the reasons why models like BEiT-3 @beit3
or VLMo @vlmo, which allow fine-grained alignment of text and image through cross-attention, perform better than models that only use
global representations, like CLIP @clip and our model.
To address the issue of fine-grained alignment, the authors of FILIP @filip
introduced Cross-Modal Late Interaction (CMLI) for contrastive learning, which showed improvements in retrieval performance even for a model
that already uses cross-attention.

As shown in @cmli, no cosine similarity between $mono(["T_CLS"])$ and $mono(["I_CLS"])$ is computed, but instead the cosine 
similarity between all image patches $[bold(h)_(v, l, k)]_(1 lt.eq k lt.eq N)$ and text tokens $[bold(h)_(w, l, j)]_(1 lt.eq j lt.eq M)$, 
with $N$ being the number of image patches, and $M$ being the number of text tokens. 
Specifically, $N$ and $M$ denote the number of patches/tokens in a sequence that are not the cls token
($mono(["I_CLS"])$/$mono(["T_CLS"])$) or padding token ($mono(["PAD"])$) @filip. The choice to exclude
padding tokens is obvious, as they do not carry any semantic information. The cls token is excluded, as it is not specific
to any token/patch, and is used to represent
global information. We additionally exclude the end-of-sequence token ($mono(["EOS"])$), as it is not specific to any token/patch either.
The result is the cosine similarity between all image patches and text tokens of an image-text pair.

The next step is to find for each image patch $k$, the text token with the maximum cosine similarity to this image patch.

$
m_k^("i2t") = op("argmax", limits: #true)_(1 lt.eq j lt.eq M) [bold(h)_(v, l, k)] [bold(h)_(w, l, j)]^T
$

Likewise, for each text token $j$, we get the image patch with the maximum cosine similarity to this text token
$
m_j^("t2i") = op("argmax", limits: #true)_(1 lt.eq k lt.eq N) [bold(h)_(v, l, k)] [bold(h)_(w, l, j)]^T
$

This has an intersting effect: For each image patch, the semantically most similar text token is found, and vice versa 
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
as e.g. a text token might have a maximum cosine similarity to another image patch, than the image patch that has its maximum
similarity to the text token @filip.
The process in illustrated in @cmli.
 
#figure(
  image(
  width: 75%,
  "../figures/cmli.png"),
  caption: [For a token/patch, CMLI finds the timestep with the highest semantic match from the other modality. This enables the model to associate small details of image and text with each other. Notice how through the $max$-operation patches containing grass are always associated with the word "grass", and the words "sheep" and "head" are matched with the head of the sheep (associations created through $max$ are shown in (2)). The cosine similarity is then the average of all associations between an image-text pair. Figure inspired and adapted from @filip.
  ],
) <cmli>

While this approach allows for a fine-grained alignment of image and text, its practical implementation is very computationally
and memory intensive. For standard constrastive learning, is is sufficient to compute the cosine similarity of the global representation
(cls token) between every possible image-text pair in a batch. If negative examples are gathered from all devices, then
the number of dot products to compute is defined as $(B*P)^2$, with $B$ being the batch size per device, 
and $P$ being the number of devices (in our case GPUs). As we use a batch size of $B=256$ per device, and use $P=2$ GPUs,
the number of dot products to compute is $(256*2)^2=262,144$. Considering that we perform this efficiently using matrix multiplication,
and the embedding size is 768 with float32 precision, we already need $262,144 * 768 * 4 "bytes" = 805.31 "MB" $ of GPU memory, which
is still manageable, since we have around 2 GB of GPU memory remaining for a step.

However, with CMLI, we need to compute the similarity between all possible image-text pairs, where the similarity
for one pair requires the computation of the cosine similarity between all image patches and text tokens of the image-text pair.
With a maximum text sequence length of 64 tokens @beit3, two of which are ignored as they are the cls and eos token,
and an image sequence length of 196 (without cls token), the number of dot products to compute for just
one image-text pair is $196*64=12,544$. With a batch size of 256 per device, and 2 GPUs, the number of dot products
increases from $262,144$ to $256*12,544=6,422,528$. Even if the embedding dimension is reduced to 256, which is a simplification
done in FILIP @filip, we need $6,422,528 * 256 * 4 "bytes" = 6.58 "GB"$ of additional GPU memory, just to store the result.
Consequently, this approach is not feasible in our setup.

==== Method <target_cmli_method>

What is feasible though, is to apply CMLI to a setting where the computation is more lightweight.
As the driving factor behind the memory requirements in contrastive learning is that the similarity between all
possible image-text pairs in a batch is computed, this is removed when just the similarity between
positive pairs is computed, which is what we call Target-CMLI.

Target-CMLI is not used for contrastive learning, but rather to alleviate the problem of regressing
patch-level information of the teacher model.
Recall that in the current setting, which is multimodal knowledge distillation, it is merely possible
to regress the global representations of the teacher model, and not the patch-level information.
This is because the teacher model only outputs patch-level predictions for the image modality, and not for the text modality.
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
image patches does not change between student and teacher image representations, however,
this would heavily bias the parameters of the shared Transformer block towards the image modality.

The definition of the loss changes as follows:

$
cal(L)_("KD")^("i2t") = 
op("MSE")(bold(H)'''^s_(v, K), bold(H)^t_(v, L_t)) =
sum_(n=1)^N ||bold(h)'''^s_(v, K, n) - bold(h)^t_(v, L_t, n)||^2_2
$

For a given text representation we first need to find $m_j^("t2i")$ for each text token $j$, and then define the loss as:

$
cal(L)_("KD")^("t2i") =
op("MSE")(bold(H)'''^s_(w, K), bold(H)^t_(v, L_t)) =
sum_(z=1)^Z ||g(bold(h)'''^s_(w, K, z)) - g(bold(h)^t_(v, L_t, m_j^("t2i")))||^2
$

We denote $g(dot)$ as a linear projection to reduce the dimensionality of the image representation from the teacher, and the text representation
from the student, to 32. This is done to (1) reduce memory consumption when computing the similarity between all image patches and text tokens
for all image-text pairs in a batch, and (2) to reduce the information that can be expressed by $g(bold(x))$. We motivate (2) by the fact
that matching individual image patches to text tokens is a very fine-grained task, and we want to aviod having pixel-level information
in the patch representations of the teacher model, which would, despite the matching via $op("argmax")$, cause problems with predicting those
representations, as we can't extract those pixel-level information from the text tokens.
The total Knowledge-Distillation loss remains the same, and is the mean of the two losses:

$
cal(L)_("KD") = 1/2 * (cal(L)_("KD")^("i2t") + cal(L)_("KD")^("t2i"))
$

We use the mean instead of the sum, as we also use the contrastive loss, and we want both losses to have the same weight.
We find $m_k^("t2i")$ using an embedding dimension of 32, which is achieved through the same projection $g(dot)$.
The hidden size of the model remains at 768.

What makes the implementation of Target-CMLI feasible is that we only need to compute the similarity between the positive pairs,
and not all possible image-text pairs in a batch. This is because we only need to find the most similar image patch for each text token
of a positive pair. For a per-device batch size of 256, Target-CMLI requires $12,544*256=3,211,264$ dot products to compute.
With an embedding dimension of 32, just $3,211,264*32*4 "bytes" = 411 "MB"$ of additional GPU memory
is required. This is feasible in our setup.

==== Results <target_cmli_results>

The results, as seen in @t_cmli_results, can be considered as disappointing. We lose more than 2 percentage points in average retrieval
on both MSCOCO and Flickr30K, while the performance on ImageNet-1K decreases by nearly 4 percentage points.

A look at a visualization (@t_cmli_examples) of matches between text tokens and their top-5 most similar image patches under cosine similarity
reveals that while there are some cases where the token has the highest similarity to an actual image patch belonging the the correct
object, the matching is far from perfect, and matches are often either completely unrelated, or scattered across almost random regions.
A matching that would be expected from a successful approach is illustrated by the authors of FILIP @filip, and can be seen in
@filip_cmli_examples. While the exaples of FILIP are based on CMLI for contrastive learning, and not our Target-CMLI for knowledge distillation,
the principle of matching text tokens to image patches remains the same, and the quality of the matches is expected to be similar.

In general, we observe that the patch with the highest similarity for an example text token is often a patch completely unrelated to the token,
and the matches do not contain enough information to accurately match the text token. For example, consider the image in the first row and
second column of @t_cmli_examples. While the matched patch token is part of a "bike", it is not large enough to actually contain the bike.
Minimizing the MSE between the representation of the text token "bike" and the representation of the matched patch will therefore
not result in a representation that is representative of the concept "bike". This is a general problem of the approach, and might be one reason
why our previous approach of regressing the global representations of the teacher model performed better.

While it is true that the representation of an image patch not only contains information about the patch itself, but also about its surrounding
patches, thanks to self-attention, this still does not explain the matches to completely unrelated patches that are not even part of the
correct object.

#figure(
  image(
  width: 100%,
  "../figures/t_cmli_examples.png"),
  caption: [
  Image-text pairs taken from COCO test set @coco.
],
) <t_cmli_examples>

#figure(
  image(
  width: 100%,
  "../figures/beit2_cls_attn_examples.png"),
  caption: [
  Image-text pairs taken from COCO test set @coco.
],
) <beit2_cls_attn_examples>


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
    caption: [C]
) <t_cmli_results>
#show table: set text(12pt)

==== Empty Target <target_cmli_empty_target>

A weakness of the aforementioned approach is that some not all text tokens carry semantic information that can be mapped
to image patches. An example of this can be seen in @cmli, where the tokens "A", "his", "to", and "some"
do not contains any information that can be related to an image patch, because they are merely fill words in a sentence.
However, with Target-CMLI there will be an image patch that is most similar to these tokens, even if the similarity is very low.
Consequently, the model will try to minimize the MSE between the representation of these tokens and the corresponding image
patches, which is not meaningful.

To address this, we propose the introduction of an empty target, which is a learnable token to which all text tokens are,
next to the image patches, compared to using cosine similarity. If the maximum similarity for a text token is the empty target,
then the loss for this token is set to zero, i.e. is ignored.

However, this approach will inevitably lead to a model collapse. This is because the model will try to find the easiest way
to minimize the loss ($cal(L)_("KD")$), and the easiest way to do that is to simply have all text tokens have the
highest similarity to the empty target. This will result in a loss of zero, as the loss for all text token is now ignored,
i.e. zero. Consequently, the model will not learn any meaningful representations. A potential solution to this is to
add another loss term that encourages the model minimize the number of text tokens that have the highest similarity to the empty target,
which will strike a balance between utilizing the empty target for meaningless tokens, and using actual image patches for meaningful tokens.
We define the loss $cal(L)_("Reg")$ as follows:

$
cal(L)_("Reg") = -log(p)
$

We denote $p$ as the percentage of tokens that have the highest similarity to an actual image patch, and not the empty target.
The more text tokens have the highest similarity to an image patch, the closer $cal(L)_("Reg")$ will be to zero, and vice versa.
This forces the model to utilize the empty target for as few tokens as possible.

The total loss is then:

$
cal(L)_("S-SMKE") = cal(L)_("KD") + cal(L)_("Reg") + cal(L)_("CL")
$