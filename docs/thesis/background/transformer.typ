== Transformer <transformer_section>

While we previously introduced concepts to train smaller models in a cost-efficient way, we
now introduce the architecture of the models used in this work. We will exclusively make
use of the Transformer architecture @transformer, which has been shown to be highly effective
across vision @beit and language @bert. Consequently, we will also define the interpretation of
image and text in the context of the Transformer.

=== Language Transformer <language_transformer>
==== Text Representation <transformer_text_representation>
In Transformers, text is represented as a sequence of discrete tokens, which are, as described in word2vec @word2vec,
embedded into D-dimensional vectors using an embedding matrix.
A single token $i$, being a (sub)word like "hell", "hello", "a", is represented as $bold(e)^w_i in RR^D$, which is the embedding of the token
resulting from the embedding matrix. A text or sentence is represented as a sequence of $M$ token embeddings/representations
$bold(E)_w = [bold(e)^w_1, bold(e)^w_2, ..., bold(e)^w_M] in RR^(M times D)$. $w$ denotes the modality being text.

In addition, each sequence is prepended
with a $mono(["T_CLS"]) in RR^D$ token, and appended with a $mono(["T_SEP"]) in RR^D$ token, often referred to as the cls and
end-of-sequence token, respectively. As with all word embeddings, they are learnable and part of the embedding matrix.
The purpose of the $mono(["T_CLS"])$ token is to aggregate the global information/content of the text, while
the $mono(["T_SEP"])$ token is used to indicate the end of the text sequence. The resulting text representation is:

$
bold(E)_w = [bold(e)^w_mono(["T_CLS"]), bold(e)^w_1, bold(e)^w_2, ..., bold(e)^w_M, bold(e)^w_mono(["T_SEP"])] in RR^((M+2) times D)
$

The sequence length $M$ is not fixed, but is usually set to a specific value when training a Transformer model, a popular choice being
$M+2 = 512$.

As will be explained later, unlike RNNs, Transformers do not process a sequence step by step, but all at once. As a result,
the Transformer does not have any sense of order in the sequence. It is therefore necessary to add a so called positional encoding
to the text embeddings, giving the Transformer a sense of order in the text sequence.
This positional encoding $bold(T)^"pos"$ is a sequence of $D$-dimensional vectors with the same length as $bold(E)_w$, and can
therefore simply be added to the text embeddings. The positional encoding, i.e. the embedding of each time step, is either learned or fixed,
and we refer to the original "Attention is all you need paper" @transformer for more details.

$
bold(T)^"pos"_w &= [bold(t)^w_"pos"_mono(["T_CLS"]), bold(t)^w_"pos"_1, bold(t)^w_"pos"_2, ..., bold(t)^w_"pos"_M, bold(t)^w_"pos"_mono(["T_SEP"])]
$

The final text representation is defined as:

$
bold(H)_(w, 0)=[bold(h)_(w, 0, mono(["T_CLS"])), bold(h)_(w, 0, 1), ..., bold(h)_(w, 0, M), bold(h)_(w, 0, mono(["T_SEP"]))] = bold(E)_w + bold(T)^"pos"_w
$ <text_representation>

We refer to $0$ as the layer number of the Transformer, which we will clarify in the next section.

==== Transformer Layer
The actual Transformer consists of a set of $L$ layers, also referred to as blocks, which are stacked on top of each other.
The input to a Transformer layer $l$ is a sequence of token embeddings $bold(H)_(w, l-1)$ produced by the previous layer, and
the input to the first layer $l=1$ is therefore the text representation $bold(H)_(w, 0)$ as defined in @text_representation.
Correspondingly, the output of a Transformer layer $l$ is denoted as $bold(H)_(w, l)$, and the length of the sequence does not change
across layers.
The architecture of a Transformer is displayed in @transformer_encoder.

#figure(
  image("../figures/transformer_encoder.png", width: 25%),
  caption: [The architecture of the Transformer encoder. The original architecture also includes a Transformer decoder, which is not
  relevant for our work and therefore omitted. The encoder consists of $N$ (we use $L$) Transformer layers. The output of a layer
  is the input to the subsequent layer. Figure is adjusted from the original Transformer paper @transformer.
  ],
) <transformer_encoder>

Independent of modality, we define the operations performed by one Transformer layer as follows:

$
bold(H)'_l &= op("LN")(op("MHA")(bold(H)_(l-1)) + bold(H)_(l-1)) \
bold(H)_l &= op("LN")(op("FFN")(bold(H)'_l) + bold(H)'_l)
$

$op("LN")(dot)$ denotes layer normalization, or short LayerNorm, as defined in @layer_norm.
Simply put, it normalizes each embedding (or time step respectively) of a sequence individually, so that the mean is zero and the variance is one.

$
h'_j = (h_j - mu_j) / sqrt(sigma_j^2 + epsilon), wide mu_k = 1/D sum_(d=0)^(D-1) h_d, wide sigma_k^2 = 1/D sum_(d=0)^(D-1) (h_d - mu_k)^2
$ <layer_norm_eq>

$epsilon$ is a small constant to avoid division by zero, and $D$ is the embedding dimension of the tokens. The operation is applied to each time step $bold(h)$ individually, and its relation to other normalization techniques, like BatchNorm, is shown in @norm_comparison.

The addition between two sequences of embeddings is a residual connection originally introduced in ResNets @resnet, and adds
each time step of one sequence to the corresponding time step of the other sequence, also seen in @text_representation when adding
the positional encoding to the text embeddings.

$op("FFN")(dot)$ denotes a timestep-wise application of a feed-forward network. In essence, it is a 2-layer MLP with Layer Norm
and GELU activation, where the first layer
usually consists of $D_"ff"$ neurons, and the second layer of $D$ neurons, with $D$ being the embedding dimension of the tokens.
A popular choice for the intermediate dimension $D_"ff"$ is $D_"ff"=4 times D$. The operation for a single time step $bold(h)$, or embedding
respectively, is defined as:

$
op("FFN")(bold(h)) = op("LN")(op("GELU")(bold(h)bold(W)_1^T + bold(b)_1))bold(W)_2^T + bold(b)_2
$ <transformer_ffn_eq>

For a sequence of embeddings $bold(H)$, the operation is applied to each time step $bold(h)$ individually, which includes the $mono(["T_CLS"])$
and $mono(["T_SEP"])$ token. Since the output of $op("FFN")(dot)$ is also in $RR^D$, the embedding
dimension of the tokens, referred to as the _hidden size_ or _hidden dim_ of a Transformer, does not change across layers.

$
op("FFN")(bold(H)) = [op("FFN")(bold(h)_mono(["T_CLS"])), op("FFN")(bold(h)_1), op("FFN")(bold(h)_2), ..., op("FFN")(bold(h)_M),
op("FFN")(bold(h)_mono(["T_SEP"]))]
$ <point_wise_ffn>

For the definition of Multi-Head Attention $op("MHA")(dot)$, performing self-attention between the tokens,
we refer to the original Transformer paper @transformer for a detailed explanation.

The output of the last layer $L$ is the final text representation $bold(H)_(w, L)$. To use this representation for downstream tasks
like classification, the sequence of embeddings has do be reduced to a single representation of the input. This is done
by taking the representation of the $mono(["T_CLS"])$ token, which is then passed to a subsequent classification head, represented by
a single linear layer:

$
bold(hat(y))_w = bold(h)_(w, L, mono(["T_CLS"]))bold(W_mono(["T_CLS"]))^T + bold(b)_mono(["T_CLS"]) in RR^C
$ <transformer_classification_head>

=== Vision Transformer <vision_transformer>

After the success of the Transformer architecture in NLP, breaking various benchmarks with architectures like BERT @bert,
which led to its widespread adoption especially in Large Language Models (LLMs) like GPT-2 @gpt2, researchers explored
the potential of this architecture beyond text. This exploration led to the development of the vision Transformer (ViT) @vit,
marking a significant shift in computer vision.

Different from traditional Convolutional Neural Networks (CNNs), which have been the dominant architecture in computer vision 
before the ViT, the ViT processes images as 1D sequences of patches, instead of 2D grids of pixels.

We define an image as a 3-dimensional tensor $bold(v) in RR^(C times H times W)$. Throughout this work, we will exclusively
use an image size of $224 times 224$ pixels and 3 color channels, so $bold(v) in RR^(3 times 224 times 224)$. Before being passed to the
Transformer layers, the image first needs to be converted into a sequence of embeddings, similar to text. This is done by
first dividing the image into $14 times 14$ patches, each being a square of size $16 times 16$ pixels. Each patch $i$ is then
flattened into a 256-dimensional vector, and projected into a 768-dimensional embedding $bold(e)^v_i in RR^768$ using a fully connected layer.
$v$ denotes the image modality.

Similar to text, the resulting sequence of patches is prepended with a special learnable $mono(["I_CLS"]) in RR^768$ token, which is used
to aggregate the global information/content of the image. The image representation is defined as a sequence of N patch embeddings
and the $mono(["I_CLS"])$ token:

$
bold(E)_v = [bold(e)^v_mono(["I_CLS"]), bold(e)^v_1, bold(e)^v_2, ..., bold(e)^v_N]
$

To again give the Transformer a sense of order in the image patches, a unique positional encoding is added to each patch embedding,
which is either learned or fixed and also represented as a sequence of 768-dimensional vectors:

$
bold(T)^"pos"_v &= [0, bold(t)^v_"pos"_1, bold(t)^v_"pos"_2, ..., bold(t)^v_"pos"_N]
$

Notice that the positional encoding for the $mono(["I_CLS"])$ token is set to zero. This is because the $mono(["I_CLS"])$ token is not 
actually part of the image, and can be seen as a type of meta token. The input to a ViT is defined as:

$
bold(H)_(v, 0)=[bold(h)_(v, 0, mono(["I_CLS"])), bold(h)_(v, 0, 1), ..., bold(h)_(v, 0, N)] = bold(E)_v + bold(T)^"pos"_v
$ <image_representation>

For an image size of $224 times 224$ pixels, and a patch size of $16 times 16$ pixels, the number of patches $N$ is $196$.

The architecture of a vision Transformer layer is almost identical to a language Transformer layer, with the only difference being
that the LayerNorm operation $op("LN")(dot)$ is applied before Multi-Head Attention $op("MHA")(dot)$ @vit, as illustrated in @vit_img.
This type of Transformer is referred to as the Pre-LN Transformer, and the operations performed by one layer changes to:

$
bold(H)'_l &= op("MHA")(op("LN")(bold(H)_(l-1))) + bold(H)_(l-1) \
bold(H)_l &= op("FFN")(op("LN")(bold(H)'_l)) + bold(H)'_l
$ <vit_full_forward_eq>

#figure(
  image("../figures/vit.png", width: 75%),
  caption: [The architecture of the vision Transformer. Its key difference to the language Transformer is the patch embedding
  and the position of LayerNorm in a layer @vit.
  ],
) <vit_img>

For downstream tasks like classification, the ViT follows the procedure of the original Transformer, and passes the representation
of the $mono(["I_CLS"])$ token to a classification head, which is a single linear (i.e. feed-forward)
layer @vit (see @transformer_classification_head).

With the introduction of the vision Transformer came the division into three different model variants, each having the same architecture
but different scales. The smallest model is the ViT-B/16, followed by the ViT-L/16. The largest model is the ViT-H/14. The number
of layers $L$ and the hidden size $D$ are different for each variant, and the ViT-B/16 has $L=12$ layers and $D=768$ hidden dim, the ViT-L/16
has $L=24$ layers and $D=1024$ hidden dim, and the ViT-H/14 has $L=32$ layers and $D=1280$ hidden dim @vit. 
Both ViT-B/16 and ViT-L/16 have a patch size of $16 times 16$ pixels, while the ViT-H/14 has a patch size of $14 times 14$ pixels.
As might have come apparent,
our explanation of the Transformer architecture is based on the ViT-B/16 model, which is the model we will make use of in this work.

== Multimodal Models <multimodal_models>

Multimodal models are characterized by their ability to process multiple modalities, such as text, images,
audio, or video, within a single model. The motivation behind these models lies in the idea that
models should be able to understand real-world concepts in a way similar to humans. Humans can express the same concept
across different modalities, “a cat”, for example, can be represented in text, image, or audio, and regardless of how the concept
is expressed, the interpretation and understanding remains the same.

Please note that since our focus is on vision-language models, all further explanations of multimodality will be
in the context of vision and language.

In the context of deep learning this means that the representations, the embedding/representation
of e.g. the $mono(["I_CLS"])$ and $mono(["T_CLS"])$ token, of a concept should be the same (or at least close to each other),
no matter if is expressed through image or text, which is also called alignment.
However, in most existing models this is not the case. These models are typically unimodal, meaning they process only one modality,
making alignment of multiple modalities impossible.
A naive approach would be to pass an image into an image model, and its caption into a (separate) text model. Even though the generated representations
describe the same concept, they will not be the same, as both models are not related to each other.
Each model will have a separate latent space, as there has been no incentive for the models to learn a representation that
is aligned across modalities (@different_latent_spaces), resulting in different representations for the same concept.
While it is possible to compare the representations of two unimodal models, e.g. through cosine similarity,
a similarity close to 1 (the maximum) does not necessarily mean that the concepts expressed in the representations are the same.
There simply is no semantic relationship between the representations of the same concept produced by two unimodal models.
A practical example of this will be shown in @first_results_transformer_shre.

To overcome this limitation, a model is required that can produce modality-invariant representations, i.e. representations that are
independent of the modality of the input.
They should map the input of different modalities into a common representation space, where the representations
of the same concept are aligned, i.e. close to each other, since they describe the same concept.

#figure(
  image(
  width: 75%,
  "../figures/different_latent_spaces.png"),
  caption: [A multimodal model maps multiple modalities into a common representation space, where the representations of the same concept are aligned. In contrast, unimodal models map the input of a single modality into a modality-specific representation space. There is no alignment
  between the representations of the same concept produced by two unimodal models (indicated by the double slashed [\//] arrow).
  While a comparison between the representations of two unimodal
  models is numerically possible, e.g. through cosine similarity, the similarity cannot be interpreted in a meaningful way.],
) <different_latent_spaces>

Multimodal models consist of both unimodal encoders and multimodal components. Unimodal encoders are needed because of the
inherent differences between modalities, e.g. image and text: Images are 2D and composed of pixels, while text is 1D and composed of words.
Unimodal encoders encode the input into a modality-specific representation space, so they are normal
unimodal models, e.g. a ResNet for images. In this work, all encoders will be based on the Transformer architecture.

Multimodal models require components that enforce a shared representation space for the modalities.
There are two options: A multimodal (or shared) encoder, or a loss function (training objective).

The multimodal encoder is responsible for mapping the modality-specific representations into a unified/shared representation space,
where representations should be independent of the modality. That means, the representations should not contain any modality-specific information,
e.g. pixel information in images or single-word information in text. Only then representations of the same concept can be aligned, or close
to each other under some distance metric, respectively.

To actually ensure that the representations of the same concept are aligned, and not only in the same space,
a training objective is needed that pushes the representations of the same concept closer together in representation space,
while pushing the representations of different concepts further apart. For vision-language models this translates to
pushing the representations of an image and its caption closer together, while pushing the representations of an image and an unrelated caption
(or vice versa) further apart. To quantify the similarity between two representations,
a distance metric is used, e.g. cosine similarity.
The loss function is usually the contrastive loss, and its implementation for vision-language models will be introduced in @vision_language_contrast.

#figure(
  image(
  width: 75%,
  "../figures/multimodal_model.png"),
  caption: [An abstract illustration of a vision-language model. Image and text are first passed through unimodal
  Transformer encoders, the cls tokens are extracted and passed separately into the MLP that maps the
  modality-specific representations into a common representation space.
  A contrastive loss ensures the alignment and repulsion of similar and dissimilar concepts, respectively. We indicate this through
  purple arrows.],
) <multimodal_model>

When it comes to the actual implementation of multimodal models, Transformers are a very suitable choice, as they can be used for
both vision and language, and even other modalities not covered in this work, like audio. Furthermore, both the language and vision
Transformer require their input to be a sequence of embeddings, which makes alignment more straightforward: For each unimodal encoder
of a multimodal (vision-language) model one distinct Transformer can be used, and the output of an unimodal encoder, which is the respective
cls token ($mono(["I_CLS"])$ or $mono(["T_CLS"])$) can then be passed (separately) to a shared encoder, which can be implemented as a simple
feed forward network (MLP).
The output of the shared encoder is then still _one_ representation for the image, and _one_ for the text. However,
both representations will be close to each other under cosine similarity, which is useful
for multimodal tasks like image-text retrieval, introduced in @image_text_retrieval. An abstract illustration of a vision-language model
with Transformer encoders and a shared encoder MLP is shown in @multimodal_model.
