== Transformer <transformer_section>

While we previously introduced concepts to train smaller models in a cost-efficient way, we
now introduce the architecture of the models used in this work. We will exclusively make
use of the Transformer architecture @transformer, which has been shown to be highly effective
across vision @beit and language @bert. Consequtively, we will define the interpretation of
image and text in the context of the Transformer.

=== Language Transformer <language_transformer>
==== Text Representation <transformer_text_representation>
In Transformers, text is represented as a sequence of discrete tokens, which are, as described in word2vec @word2vec,
embedded into D-dimensional vectors using an embedding matrix.
A single token $i$, being a (sub)word like "hell", "hello", "a", is represented as $bold(e)^w_i in RR^D$, which is the embedding of the token
resulting from the embedding matrix. A text or sentence is represented as a sequence of $M$ token embeddings/represenations
$bold(E)_w = [bold(e)^w_1, bold(e)^w_2, ..., bold(e)^w_M] in RR^(M times D)$.

In addition, each sequence is prepended
with a $mono(["T_CLS"]) in RR^D$ token, and appended with an $mono(["T_SEP"]) in RR^D$ token, often referred to as the cls and
end-of-sequence token, respectively. As with all word embeddings, they are learnable and part of the embedding matrix.
The purpose of the $mono(["T_CLS"])$ token is to aggregate the global information/content of the text, while
the $mono(["T_SEP"])$ token is used to indicate the end of the text sequence. The resulting text representation is:

$
bold(E)_w = [bold(e)^w_mono(["T_CLS"]), bold(e)^w_1, bold(e)^w_2, ..., bold(e)^w_M, bold(e)^w_mono(["T_SEP"])] in RR^((M+2) times D)
$

The sequence length $M$ is not fixed, but is usuall set to a specific value when training a Transformer model, a popular choice being
$M+2 = 512$.

As will be explained later, unlike RNNs, Transformers do not process a sequence step by step, but all at once. As a result,
the Transformer does not have any sense of order in the sequence. It is therefore necessary to add a so called positional encoding
to the text embeddings, giving the Transformer a sense of order in the text sequence.
This positional encoding $bold(T)^"pos"$ is a sequence of $D$-dimensional vectors with the same length as $bold(E)_w$, and can
therefore simply be added to the text embeddings. The positional encoding, i.e. the embedding of each time step, is either learned or fixed,
and we refer to the original Attention is all you need paper @transformer for more details.

$
bold(T)^"pos"_w &= [bold(t)^w_"pos"_mono(["T_CLS"]), bold(t)^w_"pos"_1, bold(t)^w_"pos"_2, ..., bold(t)^w_"pos"_M, bold(t)^w_"pos"_mono(["T_SEP"])]
$

The final text representation is defined as:

$
bold(H)_(w, 0)=[bold(h)_(w, 0, mono(["T_CLS"])), bold(h)_(w, 0, 1), ..., bold(h)_(w, 0, M), bold(h)_(w, 0, mono(["T_SEP"]))] = bold(E)_w + bold(T)^"pos"_w
$ <text_representation>

We refer to $0$ as the layer number of the Transformer, which we will clarify in the next section, and $w$ as the modality being text
or language, respectively.

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
  relevant for our work and therefore omitted. The encoder consists of $N$ (or $L$) Transformer layers. The output of a layer
  is the input to the subsequent layer. Figure is adjusted from the original Transformer paper @transformer.
  ],
) <transformer_encoder>

Independent of modality, we define the operations performed by one Transformer layer as follows:

$
bold(H)'_l = op("LN")(op("MHA")(bold(H)_(l-1)) + bold(H)_(l-1))
$
$
bold(H)_l = op("LN")(op("FFN")(bold(H)'_l) + bold(H)'_l)
$

$op("LN")(dot)$ denotes layer normalization, or short LayerNorm. as defined in @layer_norm.
Simply put, it normalizes each embedding (or time step respectively) of a sequence individually, so that the mean is zero and the variance is one.

$
h'_j = (h_j - mu_j) / sqrt(sigma_j^2 + epsilon), wide mu_k = 1/D sum_(d=1)^D h_d, wide sigma_k^2 = 1/D sum_(d=1)^D (h_d - mu_k)^2
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
op("FFN")(bold(h)) = op("LN")(op("GELU")(bold(h)bold(W)_1 + bold(b)_1))bold(W)_2 + bold(b)_2
$

For a sequence of embeddings $bold(H)$, the operation is applied to each time step $bold(h)$ individually, which includes the $mono(["T_CLS"])$
and $mono(["T_SEP"])$ token, though not displayed in @point_wise_ffn. Since the output of $op("FFN")(dot)$ is also in $RR^D$, the embedding
dimension of the tokens, referred to as _hidden size_ or _hidden dim_ of a Transformer, does not change across layers.

$
op("FFN")(bold(H)) = [op("FFN")(bold(h)_1), op("FFN")(bold(h)_2), ..., op("FFN")(bold(h)_M)]
$ <point_wise_ffn>

For the definition of the Multi-Head Attention $op("MHA")(dot)$, performing Self-Attention,
we refer to the original Transformer paper @transformer for a detailed explanation.

The output of the last layer $L$ is the final text representation $bold(H)_(w, L)$. To use this representation for downstream tasks
like classification, the sequence of embeddings has do be reduced to a single representation of the input. This is done
by taking the representation of the $mono(["T_CLS"])$ token, which is then passed to a subsequent classification head, represented by
a single linear layer:

$
bold(hat(y))_w = bold(h)_(w, L, mono(["T_CLS"]))bold(W_mono(["CLS"])) + bold(b)_mono(["CLS"]) in RR^C
$

#figure(
  image("../figures/pre_layer_norm.png", width: 50%),
  caption: [Comparison of a Post-Norm Transformer block/layer (a), and a Pre-Norm Transformer block/layer (b).
  (a) is the architecture as defined in the original "Attention is all you need" paper @transformer. We follow the Pre-Norm architecture @pre_layer_norm.
  ],
) <pre_layer_norm_fig>

One Transformer block performs the following operations:

$
bold(H)'_l = op("MHA")(op("LN")(bold(H)_(l-1))) + bold(H)_(l-1)
$
$
bold(H)_l = op("FFN")(op("LN")(bold(H)'_l)) + bold(H)'_l
$

We denote $op("LN")$ as LayerNorm, $op("MHA")$ as Multi-Head Attention, and $op("FFN")$ as a 2 layer MLP, all following the original Transformer
of @transformer. As previously mentioned, the only difference is the order of
operations @pre_layer_norm. $bold(H)^s_(v, l)$ and $bold(H)^s_(w, l)$ can be used as a drop-in replacement for image and text, respectively.
Both equations are, with slight adjustment, taken from VLMo @vlmo.

We define a Transformer as multiple Transformer blocks stacked on top of each other.

=== Vision Transformer <vision_transformer>

We define an image as a 3-dimensional tensor $bold(v) in RR^(C times H times W)$. Because we will use the base variant of the vision Transformer,
ViT-B/16 @vit, the image is patchified into 14x14 patches, each being a square of size 16x16 pixels. Each image patch represents one timestep in the
sequence, and the number of patches $N$ is given by $N = H times W / P^2$, with $P$ being the number of patches per dimension, and $P=14$.
Since we use an image size of 224x224 pixels, so $bold(v) in RR^(3 times 244 times 244)$, we will have $N=244 times 244 / 14^2 = 196$
patches, or timesteps respectively. Each patch is flattened into a 256-dimensional vector, 
and then projected into a 768 dimensions $bold(e)^v_i in RR^768$, using a fully connected layer. 
The image sequence is prepended with a special learnable $mono(["I_CLS"]) in RR^768$ token,
which is, following @vit, used to aggregate the global information/content of the image.
The result is a sequence of patch embeddings, which we define as $bold(E)_v$, where $v$ indicates the image modality:

$
bold(E)_v = [bold(e)^v_mono(["I_CLS"]), bold(e)^v_1, bold(e)^v_2, ..., bold(e)^v_N]
$

To give the Transformer a sense of order in the image patches/timestep, a unique positional encoding is added to each patch embedding.
This can either be learned or fixed, with the latter being for example a sinusoidal positional encoding @transformer.
This positional encoding is also represented as a sequence of 768-dimensional vectors:

$
bold(T)^"pos"_v &= [0, bold(t)^v_"pos"_1, bold(t)^v_"pos"_2, ..., bold(t)^v_"pos"_N]
$

#figure(
  image("../figures/vit.png", width: 75%),
  caption: [@vit.
  ],
) <vit_img>

Since the $mono(["I_CLS"])$ token is not part of the image, the positional encoding for the $mono(["I_CLS"])$ token is set to zero,
so nothing is added to it. An image representation is defined as:

$
bold(H)^s_(v, l)=[bold(h)^s_(v, l, mono(["I_CLS"])), bold(h)^s_(v, l, 1), ..., bold(h)^s_(v, l, N)]
$ <image_representation>

In @image_representation, $l$ denotes the layer of the Transformer block that returned the image representation, and $v$ indicates that the representation is an image. Since we use Knowledge Distillation (KD) in some parts of this thesis, representations will be,
if neccessary, superscripted with $s$ or $t$, for a student and teacher representation, respectively.

We define $l=0$ as the input to the Transformer, and $l=L$ as the output of the Transformer, where $L$ is the number of layers in the Transformer.
Consequently, the image input to the Transformer is defined as:

$
bold(H)^s_(v, 0)=[bold(h)^s_(v, 0, mono(["I_CLS"])), bold(h)^s_(v, 0, 1), ..., bold(h)^s_(v, 0, N)] = bold(E)_v + bold(T)^"pos"_v
$ <image_representation_input>

The output of the Transformer is defined as:

$
bold(H)^s_(v, L)=[bold(h)^s_(v, L, mono(["I_CLS"])), bold(h)^s_(v, L, 1), ..., bold(h)^s_(v, L, N)]
$ <image_representation_output>

=== Vision-Language Transformer <vision_language_transformer>