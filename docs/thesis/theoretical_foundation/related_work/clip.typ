#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
=== CLIP
==== Method

CLIP is a method developed by OpenAI to train a vision-language model using contrastive learning. CLIP stands for
(#text(weight: "bold")[C]ontrastive #text(weight: "bold")[L]anguage-#text(weight: "bold")[I]mage #text(weight: "bold")[P]retraining).
The architecture consists of a separate image encoder $f$ and text encoder $g$, both of which can be any architecture,
and a linear projection (linear layer without bias and activation function) on top of the modality-specific encoders.

The forward pass works as follows:
For a batch of image-text pairs, the images $bold(v)$ are passed through the image encoder, resulting in an image representation $f(bold(v))$.
Similarly, the texts $bold(w)$ are passed through the text encoder, producing a text representation $g(bold(w))$.

Both the image and text representations produced by the encoders are in separate embedding spaces — one for text and one for images -
they are not related to each other initially.
However, for contrastive learning to be effective, the embeddings should exist in the same latent space, after all, the embedding for an image
and its corresponding text should be the same (or at least very close to each other).

In SHRE, discussed in the previous section, this shared latent space is achieved through a shared encoder on top of the modality-specific encoders,
and through a ranking loss @shre. CLIP maps the image and text representations into a shared latent space using linear projections $O_v$ and $O_t$
for image and text, respectively. These linear projections allow the model to map the image and text embeddings in a shared latent space, 
which is ensured by the contrastive loss.

The image representation in the shared embedding space is denoted as $bold(I) = ||O_v times f(bold(v))^T||_2$, and the text representation
as is given by $bold(T) = ||O_t times g(bold(w))^T||_2$. Since cosine similarity is used as the similarity metric in the contrastive loss,
the embeddings are normalized, which is indicated by the $||dot||_2$ around the result of the linear projections.
It is important to note that the superscript $T$ denotes the transpose of a matrix, not the batch of text representations, and that
$times$ denotes matrix multiplication.

To compute the cosine similarity between all possible image-text pairs in the batch, it is sufficient to perform matrix multiplication
of the normalized representations. The result is given by $bold(L) = exp(t) * (bold(I) times bold(T))$, with $bold(L) in RR^(B times B)$,
where $B$ is the batch size.

In the calculation, it is notable that the cosine similarities are scaled by $exp(t)$, where $t$ is a temperature parameter.
This parameter is used to control the smoothness of the softmax function, which is applied to the cosine similarities.
The concept of temperature was originally introduced in the context of Knowledge Distillation (TODO: cite KD section) to generate soft targets.

In Knowledge Distillation, the temperature was introduced as a tunable hyperparameter @kd_survey @shre.
However, in CLIP, it is a learnable parameter that is optimized during training, just like any other parameter in the model,
eliminating the need for manual tuning. The temperature $t$ is optimized in log-space, which is why the actual temperature
by which logits are scaled, is given by $exp(t)$ @clip.

Although the authors did not provide a specific reason for optimizing in log-space, it is likely that this approach
ensures that the temperature is always positive, since $exp(t)$ always returns a positive value. Optimizing in log-space may also
contribute to greater numerical stability (the logarithm grows at a low rate), resulting in less drastic changes in the
temperature during optimization and thereby making training more stable.

In the matrix $bold(L)$, the cosine similarity between image $i$ and text $j$ in the batch is denoted by $bold(L)_(i,j)$,
where the diagonal elements contain the similarity for positive pairs. To maximize the similarity between positive pairs $(i, i)$,
and minimize the similarity between negative pairs $(i, j)$, with $i eq.not j$, cross-entropy loss is used.

To calculate the probability that the correct caption belongs to the current image, the cosine similarity between them is
softmax-normalized with respect to the similarity of the image with all other captions in the batch. This means that each row
in the similarity matrix $bold(L)$ represents the similarities of one image to all texts. The loss is defined as the
negative log-likelihood of the probability that the caption belongs to the correct image. This loss is referred to as the image-to-text
(it2) loss $cal(L)_"CLIP"^("i2t")$, which is computed as mean of the negative log-likelihoods for all images (as described before).

$
cal(L)_"CLIP"^("i2t") = 1/B sum_(i=1)^B -log exp(bold(L)_(i, i))/(sum_(k=1)^B exp(bold(L)_(i, k)))
$

To get the text-to-image (t2i) loss, the same process is applied, the only difference is that the
softmax-normalization is with respect to the similarity of the text with all other images. This ensures that the cosine similarity
of a text with the correct image is maximized, while the similarity of the text with all other images (in the batch) is minimized.

$
cal(L)_"CLIP"^("t2i") = 1/B sum_(i=1)^B -log exp(bold(L)_(i, i))/(sum_(k=1)^B exp(bold(L)_(k, i)))
$

The final loss of CLIP is the mean of the image-to-text and text-to-image loss:

$
cal(L)_"CLIP" = 1/2 * (cal(L)_"CLIP"^("i2t") + cal(L)_"CLIP"^("t2i"))
$

CLIP only relies on contrastive learning to train a vision-language model, and therefore requires high batch size to achieve good results.
The authors use a very large batch size of 32,768 @clip. An abstract illustration of the end-to-end training process of CLIP is shown in
(TODO: cite figure) in the Appendix.

==== Zero-Shot Image Classification


#bibliography("../../references.bib")