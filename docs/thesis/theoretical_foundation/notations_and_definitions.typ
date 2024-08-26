#set math.equation(numbering: "(1)")
== Notations and Definitions <notions_and_definitions>
Throughout this work we will make use of various concepts and their notations,
which we will define here for easier reference, and to avoid redundancy.
Bold symbols (e.g. $bold(v)$) denote vectors, and $v_i$ the $i$-th element of the respective vector.
Upper-cased bold symbols (e.g. $bold(M)$) denote matrices, and $M_(i j)$ the element in the $i$-th row and $j$-th column of the respective matrix.

=== Loss Functions
==== Mean Squared Error (MSE)
The Mean Squared Error (MSE) is a loss function used in regression tasks, and
describes the average of the squared differences between the prediction $bold(hat(y)) in RR^d$ and the target $bold(y) in RR^d$. Since
in this work the predictions and targets will exclusively be in the form of d-dimensional vectors, the MSE is defined as:

$
cal(L)_("MSE")(bold(y), bold(hat(y))) = ||bold(y)- bold(hat(y))||_2^2 = frac(1, d) sum_(j=1)^d (y_j - hat(y)_j)^2
$ <mean_squared_error>

==== Kullback-Leibler Divergence (KL-Divergence)
The Kullback-Leibler Divergence (KL-Divergence) is used to measure the difference between two probability distributions.
Specifically, in the context of Machine Learning, we are comparing a predicted probability distribution $bold(q) in RR^d$
with a target distribution $bold(p) in RR^d$.
Since we are using the KL-Divergence in the context of classification tasks, which are
discrete distributions over classes, the KL-Divergence is defined as:

$
cal(L)_("KD")(bold(p) || bold(q)) = D_("KL")(bold(p) || bold(q)) = sum_(j)p_j log frac(p_j, q_j)
$ <kl_divergence>

$p_j$ and $q_j$ are the probabilities of class $j$ according to the target and
predicted distribution, respectively.
For both distributions, there are potentially multiple classes with a non-zero probability:

$
forall j (p_j in [0, 1]) and sum_(j) p_j = 1
$ <kl_constraint>

==== Cross-Entropy Loss (CE)
The Cross-Entropy Loss (CE) is quite similar to the KL-Divergence in that it compares two probability distributions in
classification tasks. It is defined as:

$
cal(L)_("CE")(bold(p), bold(q)) = H(bold(p), bold(q)) = H(bold(p)) + D_("KL")(bold(p) || bold(q))
= -sum_(j)p_j log p_j + sum_(j)p_j log frac(p_j, q_j)
$ <cross_entropy>

Here $H(bold(p))$ denotes the entropy of the target distribution $bold(p)$, and $D_("KL")(bold(p) || bold(q))$
the KL-Divergence between the target and predicted distribution.

The difference between KL-Divergence and cross-entropy is that the latter is used in traditional classification tasks,
where the target distribution $bold(p)$ is fixed and one-hot encoded,
meaning that there is only one correct class:

$
exists! j (p_j=1) and forall k(k eq.not j -> p_k=0)
$ <cross_entropy_constraint>

This strengthens the condition of the KL-Divergence, which we defined previously in @kl_constraint.
Since the goal is to minimize the cross-entropy loss $H(bold(p), bold(q))$ and $bold(p)$ is fixed,
the entropy of the target distribution $H(bold(p))$ is a constant, and does not affect the minimization.
Moreover, given the constraint in @cross_entropy_constraint, only one term in the sum of the KL-Divergence is non-zero.
Consequently, we can simplify the cross-entropy loss, so that the training objective for classification tasks is:

$
min H(bold(p), bold(q)) &= H(bold(p)) + D_("KL")(bold(p) || bold(q)) \
&= D_("KL")(bold(p) || bold(q)) \
&= sum_(j)p_j log frac(p_j, q_j) \
&= log frac(1, q_i) \
&= -log q_i
$ <cross_entropy_minimization>

The cross entropy loss therefore minimizes the negative log-likelihood of the correct class $i$.

Often times, the prediction of a model $bold(x)$ is returned as raw logits, and not as probabilities.
To convert logits into probabilities, the softmax function is used. For ease of use, without having to mention a 
softmax-normalization every time we make use of the cross-entropy loss, we redefine the cross-entropy loss
_actually used in this work_ as:

$
cal(L)_("CE")(bold(p), bold(x)) = H(bold(p), bold(x)) = - log exp(x_i)/(sum_(j) exp(x_j))
$ <cross_entropy_single_class_exp>

We denote $bold(x)$ as the raw logits (the model prediction), and $bold(p)$
as the one-hot encoded target distribution. $i$ is the index of the correct class,
and hence each element in $bold(x)$ corresponds to the raw logit for one class.

A comparison between the target distribution predicted using KL-Divergence, and another predicted by cross-entropy
is shown in the following figure.

#figure(
  image("../figures/target_dist.png", width: 75%),
  caption: [
    Comparison between the distributions with 10 classes. The one-hot distribution (left)
    is used for classification tasks with the cross-entropy loss. The KL-Divergence is used
    when predicting a smooth distribution (right). A smooth distribution usually results from a model prediction,
    and is a popular target distribution for knowledge distillation, introduced in a later section.
  ],
) <target_dist>

=== Modality Representations
Since the architectures used in the experiments of this work are based on the Transformer @transformer
and vision Transformer @vit architecture, both image and text are represented as sequences of embeddings, 
which are processed by the Transformer blocks.

=== Image Representation

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
bold(H)_l = op("FFN")(op("LN")(bold(H)'_l)) + bold(H)'_l
$

We denote $op("LN")$ as LayerNorm, $op("MHA")$ as Multi-Head Attention, and $op("FFN")$ as a 2 layer MLP, all following the original Transformer
of @transformer. As previously mentioned, the only difference is the order of
operations @pre_layer_norm. $bold(H)^s_(v, l)$ and $bold(H)^s_(w, l)$ can be used as a drop-in replacement for image and text, respectively.
Both equations are, with slight adjustment, taken from VLMo @vlmo.

We define a Transformer as multiple Transformer blocks stacked on top of each other.

#bibliography("../references.bib")
