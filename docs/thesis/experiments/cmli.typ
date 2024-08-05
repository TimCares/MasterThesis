#set math.equation(numbering: "(1)")
===== Cross-Modal Late Interaction (CMLI) <cross_modal_late_interaction>
Until now, we used the global text and image representations $mono(["T_CLS"])$ and $mono(["I_CLS"])$, respectively, for
contrastive learning and the alignment loss of (TODO: cite removing itc).
This has the disadvantage that only global information is utilized, and fine-grained, token/patch-specific, information is not considered.
This can make retrieval, and alignment in general, difficult, especially if real-world concepts described by and image and text differ
in small, yet important, details.
To address this, the authors of FILIP @filip introduce Cross-Modal Late Interaction (CMLI) for a fine-grained comparison of text and image in contrastive learning.

As shown in @cmli, no cosine similarity between $mono(["T_CLS"])$ and $mono(["I_CLS"])$ is computed, but instead the cosine 
similarity between all image patches $[bold(v)_l^k]_(1 lt.eq k lt.eq N)$ and text tokens $[bold(w)_l^j]_(1 lt.eq j lt.eq M)$, 
with $N$ being the number of image patches, and $M$ being the number of text tokens. 
Specifically, $N$ and $M$ denote the number of patches/tokens in a sequence that are not the cls token
($mono(["I_CLS"])$/$mono(["T_CLS"])$) or padding token ($mono(["PAD"])$) @filip. The choice to exclude
padding tokens is obvious, as they do not carry any semantic information. The cls token is excluded, as it contains
"just" global information. The result is that we now have the cosine similarity between all image patches and text tokens of an image-text pair.

The next step is to find for each image patch $k$, the text token with the maximum cosine similarity to this image patch.

$
m_k^("i2t") = op("argmax", limits: #true)_(1 lt.eq j lt.eq M) [bold(v)^k] [bold(w)^j]^T
$

Likewise, for each text token $j$, we get the image patch with the maximum cosine similarity to this text token
$
m_j^("t2i") = op("argmax", limits: #true)_(1 lt.eq k lt.eq N) [bold(v)^k] [bold(w)^j]^T
$

This has an intersting effect: For each image patch, the semantically most similar text token is found, and vice versa 
for each text token - the result of this operation can be seen in (2) of @cmli.
Consequently, the model will be able to associate small details of an image with individual text tokens, and vise versa. 
The actual cosine similarity between an image-text pair is then the average of all associations between an image patch and a text token.

$
s^("i2t")_(bold(H)^v_(l),bold(H)^w_(l)) = 1/N sum_(k=1)^N [bold(v)_l^k] [bold(w)_l^(m_k^("i2t"))]^T
$

$
s^("t2i")_(bold(H)^v_(l),bold(H)^w_(l)) = 1/M sum_(j=1)^M [bold(v)_l^(m_j^("t2i"))] [bold(w)_l^j]^T
$

Here, for one image-text pair, $m_k^("i2t")$ denotes the index of the text token with the highest cosine similarity to image patch $k$,
and $m_j^("t2i")$ the index of the image patch with the highest cosine similarity to text token $j$. $s^("i2t")_(bold(H)^v_(l),bold(H)^w_(l))$
denotes the the similarity score between an image representation $bold(H)^v_(l)$ and text representation $bold(H)^w_(l)$.
Vice versa, $s^("t2i")_(bold(H)^v_(l),bold(H)^w_(l))$ denotes the similarity score between a text representation $bold(H)^w_(l)$ and an image representation $bold(H)^v_(l)$. $l$ can denote any layer of the model, but we will use, as done in FILIP @filip, the last layer of the model,
so if a model has $L$ layers, then $l=L$.

In contrast to the standard contrastive learning, this similarity measure is not necessarily symmetric,
as e.g. a text token might have a maximum cosine similarity to another image patch, than a image patch to the text token @filip.
The process in illustrated in @cmli.
 
#figure(
  image(
  width: 50%,
  "../figures/cmli.png"),
  caption: [For a token/patch, CMLI finds the semantic timestep with the highest match from the other modality. This enables the model to associate small details of image and text with each other. Notice how through the $max$-operation patches containing grass are always associated with the word "grass", and the words "sheep" and "head" are matched with the head of the sheep (associations created through $max$ are shown in (2)). The cosine similarity is then the average of all associations between an image-text pair. Figure inspired and adapted from @filip.
  ],
) <cmli>

While this approach allows for a fine-grained alignment of image and text, its practical implementation is very computationally
and memory intensive. For standard constrastive learning, is is sufficient to compute the cosine similarity of the global representation
(cls token) between every possible image-text pair in a batch. If negative examples are gathered from all devices, then
the number of dot products to compute is defined as $(B*P)^2$, with $B$ being the batch size per device, 
and $P$ being the number of devices (in our case GPUs). As we use a batch size of $B=256$ per device, and use $P=2$ GPUs,
the number of dot products to compute is $(256*2)^2=262,144$. Considering that we perform this efficiently using matrix multiplication,
and the embedding size is 768, with float32 precision, we already need $262,144 * 768 * 4 "bytes" = 805.31 "MB" $ of GPU memory, which
is still manageable, since we have around 2 GB of GPU memory remaining for a step.

However, with CMLI, we need to compute the similarity between all possible image-text pairs, where the similarity
for one pair requires the computation of the cosine similarity between all image patches and text tokens of the image-text pair.
With a maximum text sequence length of 64 tokens @beit3, two of which are ignored as they are the cls and eos token,
and an image sequence length of 196 (without cls token), the number of dot products to compute for just
one image-text pair is $196*64=12,544$. With a batch size of 256 per device, and 2 GPUs, the number of dot products
increases from $262,144$ to $256*12,544=6,422,528$. Even if the embedding dimension is reduced to 256, which is a simplification
done in FILIP @filip, we need $6,422,528 * 256 * 4 "bytes" = 6.58 "GB"$ of additional GPU memory, just to store the result.
Consequently, this approach is not feasible in our setup.

===== Target-CMLI <target_cmli>

What is feasible though, is to apply CMLI to a setting where the computation is more lightweight.
As in Contrastive Learning, the driving factor behind the memory requirements is that the similarity between all
possible image-text pairs in a batch is computed. However, this is not the case when just the similarity between
the positive pairs is computed, which is the case for what we call Target-CMLI.

Target-CMLI is not used for contrastive learning, but rather to alleviate the problem of regressing
patch-level information of the teacher model.
Recall that in the current setting, which is Multimodal Knowledge Distillation, it is merely possible
to regress the global representations of the teacher model, and not the patch-level information.
This is because the teacher model only outputs patch-level predictions for the image modality, and not for the text modality.
Consequently, the student model can replicate the output of the teacher model for the image modality, but not for the text modality,
as it is not possible to assign a text token to a specific image patch (without labeled data).
This is illustrated in (TODO: cite \@mm_kd_cls_token) of (TODO: cite \@differences_to_unimodal_knowledge_distillation).

However, as seen in @cmli, when computing the cosine similarity between all image patches and text tokens of an image-text pair,
and selecting the $op("argmax")$ over all image patches with respect to a text token, then a corresponding, or at least similar,
image patch can be found for each text token ((2) of @cmli).
This means that the student model can replicate the output of the teacher model for the text modality, by first selecting the most
similar image patch for each text token, and then minimizing the Mean Squared Error (MSE) between the teacher's representation of 
the selected image patch and the representation of the text token.

For the patch-level image representation of the student model that means that we can now also regress the patch-level information
of the teacher, and not only the global information. This was also possible in all previous experiments, as the order of the
image patches does not change, however, this would heavily bias the parameters of the shared Transformer block towards the image modality.
The definition of the loss changes as follows. For a given image representation
$
bold(H)^s_(v, L)=[bold(h)^s_(v, L, mono(["I_CLS"])), bold(h)^s_(v, L, 1), ..., bold(h)^s_(v, L, N)]
$
of the student, and the image representation
$
bold(H)^t_(v, L)=[bold(h)^t_(v, L, mono(["I_CLS"])), bold(h)^t_(v, L, 1), ..., bold(h)^t_(v, L, N)]
$ 
of the teacher, the loss is defined as:

$
cal(L)_("KD")^("i2t") = op("MSE")(bold(H)^s_(v, L), bold(H)^t_(v, L)) = 1/N sum_(k=1)^N ||bold(h)^s_(v, L, k) - bold(h)^t_(v, L, k)||^2
$

For a given text representation
$
bold(H)^s_(w, L)=[bold(h)^s_(w, L, mono(["T_CLS"])), bold(h)^s_(w, L, 1), ..., bold(h)^s_(w, L, M), bold(h)^s_(w, L, mono(["T_SEP"]))]
$
of the student, we first need to find $m_j^("t2i")$ for each text token $j$, and then define the loss as:

$
cal(L)_("KD")^("t2i") = op("MSE")(bold(H)^s_(w, L), bold(H)^t_(v, L)) = 1/N sum_(k=1)^N ||bold(h)^s_(w, L, k) - bold(h)^t_(v, L, m_k^("t2i"))||^2
$

The total Knowledge-Distillation loss remains the same, and is the mean of the two losses:

$
cal(L)_("KD") = 1/2 * (cal(L)_("KD")^("i2t") + cal(L)_("KD")^("t2i"))
$

We use the mean instead of the sum, as we also use the contrastive loss, and we want both losses to have the same weight.
As in FILIP @filip, we find $m_j^("t2i")$ using an embedding dimension of 256. The hidden size of the model stays at 768,
and we use a linear projection to reduce the dimensionality of the image representation from the teacher, and the text representation
from the student, to 256. The loss is still computed using the raw outputs, with 768 dimensions, so only CMLI is performed
in the lower-dimensional space.

What makes the implementation of Target-CMLI feasible is that we only need to compute the similarity between the positive pairs,
and not all possible image-text pairs in a batch. This is because we only need to find the most similar image patch for each text token
of a positive pair. For a per-device batch size of 256, Target-CMLI requires $12,544*256=3,211,264$ dot products to compute.
With an embedding dimension of 256 and half-precision float16, just $3,211,264*256*2 "bytes" = 1.64 "GB"$ of additional GPU memory
is required. This is feasible in our setup.

#bibliography("../references.bib")