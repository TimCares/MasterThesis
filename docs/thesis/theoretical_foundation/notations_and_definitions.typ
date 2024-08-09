#set math.equation(numbering: "(1)")
== Notations and Definitions <notions_and_definitions>

The architectures used in the experiments of this thesis are based on the Transformer @transformer, and vision Transformer @vit architecture.
Therefore, both image and text are represented as sequences of embeddings, which are processed by the Transformer blocks.

=== Image Representation

We define an image as a 3-dimensional tensor $bold(v) in RR^(C times H times W)$. Because we will use the base variant of the vision Transformer,
ViT-B/16 @vit, the image is patchified into 14x14 patches, each being a square of size 16x16 pixels. Each image patch represents one timestep in the
sequence, and the number of patches $N$ is given by $N = H times W / P^2$, with $P$ being the number of patches per dimension, and $P=14$.
Since we use an image size of 224x224 pixels, so $bold(v) in RR^(3 times 244 times 244)$, we will have $N=244 times 244 / 14^2 = 196$
patches, or timesteps respectively. Each patch is flattened into a 256-dimensional vector, 
and then projected into a 768 dimensions $bold(e)^v_i in RR^768$, using a fully connected layer. 
The image sequence is prepended with a special learnable $mono(["I_CLS"]) in RR^768$ token,
which is used to aggregate the global information/content of the image, and following @vit.
The result is a sequence of patch embeddings, which we define as $bold(E)_v$, where $v$ indicates an image:

$
bold(E)_v = [bold(e)^v_mono(["I_CLS"]), bold(e)^v_1, bold(e)^v_2, ..., bold(e)^v_N]
$

To give the Transformer a sense of order in the image patches/timestep, a unique positional encoding is added to each patch embedding.
This can either be learned or fixed, with the latter being for example a sinusoidal positional encoding @transformer.
This positional encoding is also represented as a sequence of 768-dimensional vectors:

$
bold(T)^"pos"_v &= [0, bold(t)^v_"pos"_1, bold(t)^v_"pos"_2, ..., bold(t)^v_"pos"_N]
$

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

=== Text Representation
We define a text as a sequence of discrete tokens, which are, similiar to image patches, embedded into 768-dimensional vectors
using an embedding matrix.
A single token $i$ is represented as $bold(e)^t_i in RR^768$, and the sequence of tokens, representing the text, is prepended
with a start-of-sequence token $mono(["T_CLS"]) in RR^768$, and appended with an end-of-sequence token $mono(["T_SEP"]) in RR^768$.
The purpose of the $mono(["T_CLS"])$ token is, as with $mono(["I_CLS"])$, to aggregate the global information/content of the text.
The $mono(["T_SEP"])$ token is used to indicate the end of the text sequence. A text sequence 
consists of $M$ tokens, and we use $w$ to denote a text sequence:

$
bold(E)_w = [bold(e)^w_mono(["T_CLS"]), bold(e)^w_1, bold(e)^w_2, ..., bold(e)^w_M, bold(e)^w_mono(["T_SEP"])]
$

The maximum text sequence length $M$ is not fixed, and will be defined when neccessary in the experimental part of this work.

As mentioned (TODO: cite data preparation), to obtain discrete tokens, a sentence is tokenized into subwords using the GPT-2 byte-pair encoder,
so one token does not necessarily represent a whole word.

A positional encoding is also added to the text embeddings, to give the Transformer a sense of order in the text sequence.
Since the special token $mono(["T_SEP"])$ denotes the end of the text sequence, it is part of the sequence, and therefore has a positional encoding.
The latter does not hold for the $mono(["T_CLS"])$ token, as it is used to aggregate the global information/content of the text.

$
bold(T)^"pos"_w &= [0, bold(t)^w_"pos"_1, bold(t)^w_"pos"_2, ..., bold(t)^w_"pos"_M, bold(t)^w_"pos"_mono(["T_SEP"])]
$

A text representation is defined as:

$
bold(H)^s_(w, l)=[bold(h)^s_(w, l, mono(["T_CLS"])), bold(h)^s_(w, l, 1), ..., bold(h)^s_(w, l, M), bold(h)^s_(w, l, mono(["T_SEP"]))]
$ <text_representation>

@text_representation denotes the representation denoted by a student model $s$, but it can also be a teacher representation $t$.

The input to the Transformer for text is @text_representation with $l=0$, and the output of the Transformer is @text_representation with $l=L$.

=== Transformer Block
Unless we use pretrained architectures that follow a different architecture, which we will then specify, we follow the Pre-LayerNorm definition of the Transformer block as given in @pre_layer_norm. As the name suggests, it applies LayerNorm before the Multi-Head Attention, instead of after.

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
bold(H)_l = op("FFN")(bold(H)_(l-1)) + bold(H)'_l
$

We denote $op("LN")$ as LayerNorm, $op("MHA")$ as Multi-Head Attention, and $op("FFN")$ as a 2 layer MLP, all following the original Transformer
of @transformer. As previously mentioned, the only difference is the order of
operations @pre_layer_norm. $bold(H)^s_(v, l)$ and $bold(H)^s_(w, l)$ can be used as a drop-in replacement for image and text, respectively.
Both equations are inspired by VLMo @vlmo.

We define a Transformer as multiple Transformer blocks stacked on top of each other.

#bibliography("../references.bib")
