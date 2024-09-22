=== CLIP <clip_section>
==== Method

CLIP is a method developed by OpenAI to train a vision-language model using contrastive learning. CLIP stands for
(#text(weight: "bold")[C]ontrastive #text(weight: "bold")[L]anguage-#text(weight: "bold")[I]mage #text(weight: "bold")[P]retraining).
The architecture consists of a separate image encoder $f(dot)$ and text encoder $g(dot)$, both of which can be any architecture,
and a linear projection (linear layer without bias and activation function) on top of the modality-specific encoders.

The forward pass works as follows:
For a batch of image-text pairs, the images ${bold(H)_((v, 0), i)}_(i=1)^B$ ($B$ denotes the batch size) 
are passed through the image encoder, resulting in an image
representation $bold(I) in RR^(B times D)$.
Similarly, the texts ${bold(H)_((w, 0), i)}_(i=1)^B$ are passed through the text encoder, producing a text representation $T$.
Recall that $bold(I)$ and $bold(T)$ correspond to the batched representations of the $mono(["I_CLS"])$ and $mono(["T_CLS"])$ tokens, 
as defined in (TODO: cite equ for I) and (TODO: cite equ for T), respectively.

Both the image and text representations produced by the encoders are in separate embedding spaces — one for images and one for text -
they are not related to each other initially.
However, for contrastive learning to be effective, the embeddings should exist in the same latent space. After all, the embedding for an image
and its corresponding text should be the same (or at least very close to each other).

In SHRE, discussed in the previous section, this shared latent space is achieved through a shared encoder on top of the modality-specific encoders,
and through a ranking loss @shre. CLIP maps the image and text representations into a shared latent space using linear projections $bold(o)_v$ and $bold(o)_w$
for image and text, respectively. These linear projections allow the model to map the image and text embeddings in a shared latent space, 
which is ensured by the contrastive loss. Note that the linear projections $bold(o)_v$ and $bold(o)_w$ can also be defined as functions, but
for consistency with the original paper we use dot product notation, as shown in the following.

The image representation in the shared embedding space is denoted as $bold(I)' = ||bold(o)_v bold(I)^T||_2$, and the text representation
as is given by $bold(T)' = ||bold(o)_w bold(T)^T||_2$. Since cosine similarity is used as the similarity metric in the contrastive loss,
the embeddings are normalized, which is indicated by the l2 norm $||dot||_2$ around the result of the linear projections.
It is important to note that the superscript $T$ denotes the transpose of a matrix, not the batch of text representations.

Then, it is sufficient to perform matrix multiplication
of the normalized representations in order to compute the cosine similarity between each pair. The result is given by:

$
bold(L) = exp(t) * bold(I)' bold(T)'^T, bold(L) in RR^(B times B)
$

The opteration is quite similar to the batched cosine similarity operation introduced in (TODO: cite vision-lang-contrast). However,
it is notable that the cosine similarities $bold(L)$ are scaled by $exp(t)$, where $t$ is a temperature parameter.
This parameter is used to control the smoothness of the softmax function, and is a scalar applied element-wise to the cosine similarities, which should be
a familiar concept from knowledge distillation (TODO: cite KD section).

In knowledge distillation, the temperature was introduced as a tunable hyperparameter @kd_survey @shre.
However, in CLIP it is a learnable parameter that is optimized during training, just like any other parameter in the model,
eliminating the need for manual tuning. The temperature $t$ is optimized in log-space, which is why the actual temperature
by which logits are scaled, is given by $exp(t)$ @clip.

Although the authors did not provide a specific reason for the optimization in log-space, it is likely that this approach
ensures that the temperature is always positive, since $exp(t)$ always returns a positive value. Optimizing in log-space may also
contribute to greater numerical stability (the logarithm grows at a low rate), resulting in less drastic changes in the
temperature during optimization and thereby making training more stable.

In the matrix $bold(L)$, the cosine similarity between image $i$ and text $j$ in the batch is denoted by $bold(L)_(i,j)$,
where the diagonal elements contain the similarity for positive pairs. To maximize the similarity between positive pairs $(i, i)$,
and minimize the similarity between negative pairs $(i, j)$, with $i eq.not j$, cross-entropy loss is used.

The loss for selecting the correct caption for each image and vice versa is
exacly the same as given in (TODO: cite vision-lang-contrast i2t) and (TODO: cite vision-lang-contrast t2i), respectively.
The final loss of CLIP is the vision-language contrastive loss, given in (TODO: cite vision-lang-contrast).

$
cal(L)_"CLIP" = cal(L)_"CL" = 1/2 * (cal(L)_"CL"^("i2t") + cal(L)_"CL"^("t2i"))
$

CLIP only relies on contrastive learning to train a vision-language model, and therefore requires a high batch size to achieve good results.
The authors use a very large batch size of 32,768 @clip. An abstract illustration of the end-to-end training process of CLIP is shown in
(TODO: cite figure) in the Appendix.

==== Zero-Shot Image Classification <clip_zero_shot_section>

What makes CLIP special is its method of zero-shot image classification using the trained model.
This capability is achieved through prompt engineering on the text encoder.
For each class in the dataset, where image classification is desired, the name of the class is injected into a prompt template.
The prompt template follows a structure like this: "a photo of a {class name}.".

CLIP uses 80 different prompts, so for each class in the dataset, 80 distinct prompts are generated (similar to the example shown above).
These 80 prompts are passed through the text encoder and text projection, resulting in 80 different text embeddings for one class.
These embeddings are then averaged and normalized, yielding a single embedding per class.
This embedding captures the semantic meaning of the class name, which the model learned through contrastive pretraining.

To classify an image, the image is passed through the image encoder and image projection, resulting in an image embedding.
The cosine similarity between this image embedding and all class embeddings is calculated.
The class corresponding to the text embedding with the highest similarity to the image representation is predicted as the class for the image,
as demonstrated in @clip_zero_shot.

#figure(
  image("../../figures/clip_zero_shot.png"),
  caption: [For zero-shot image classification, CLIP uses prompt engineering to create one classifier per image class to predict (2).
  The class whose classifier has the highest similarity (cosine) with the image representation is the predicted class (3) for the image @clip.],
) <clip_zero_shot>

The approach reaches a zero-shot accuracy of 76.2% on the validation set of ImageNet-1K @imagenet, with a top-5 accuracy of 95% @clip.
This is particularly impressive given that the model has never seen any images from the ImageNet-1K dataset during training,
nor has it been trained on any image classification task.
It merely achieves this accuracy through its cross-modal understanding between text and image. The model effectively “knows”
how the ImageNet-1K classes look visually.

However, it is important to note that these results were based on a vision Transformer following the ViT-L/14@336px architecture for
the image encoder. This architecture consists of 24 layers, 16 attention heads, a hidden size of 1024, and processes images
at a resolution of 336x336 @clip. For the text encoder, a 12-layer Transformer was used, consisting of 12 attention heads
and a hidden size of 768 @clip. According to HuggingFace, the model is 428 million parameters
large #footnote[#link("https://huggingface.co/openai/clip-vit-large-patch14")].
Additionally, the model was trained on a custom dataset specifically developed for CLIP, consisting of 400 million image-text pairs @clip.
