#set math.equation(numbering: "(1)")

== Unimodal Knowledge Distillation
To validate whether unimodal knowledge distillation even works, which is undoubtedly a simpler task than the multimodal knowledge distillation we
are trying to develop, we will first conduct experiments on unimodal knowledge distillation. That is, the usual knowledge distillation introduced
in section (TODO: cite kd).
=== Vision

==== Method
Our approach to vision KD involves using BEiTv2, self-supervised pretrained on ImageNet-1K @imagenet @beitv2, as the teacher model and training
a shallow version of the vision variant from Data2Vec2 @data2vec2 as the student model. This approach might be seen as unusual by some, as
a more intuitive choice would be to construct a small BEiTv2 variant for the student model, and then 
training it to mimic the large model. 
DistilBERT @distilbert for example select half of the layers from a pretrained BERT @bert model, and organizes them into
a smaller model, which is then trained to replicate the output of the pretrained BERT.
However, we use layers of a different pretrained model and organize them into our student, as this step will be inevitable in the multimodal case.
This is becuase the multimodal student model will later not only be far more complex, but its multimodal nature also requires a
different architecture than the teacher. We believe that
taking a similar approach here will give us valuable insights about the feasibility (of multimodal KD) on a comparatively simple exampe, and
provide us with a foundation on which we can build our approach to multimodal KD.

BEiTv2 is a self-supervised model, and therefore does not provide a probability distribution over classes that can be predicted using KL-Divergence.
Instead, we only have access to the model's activations for each layer, so we have to resort to feature-based knowledge distillation.
One option would be to predict the teacher's output for the cls token $bold(h)^t_(v, L, mono(["I_CLS"]))$, which aggregates the high level
content of the image, and then use the mean squared error as the loss function.
However, this neglects the activations for individual image patches and activations of intermediate layers.

This argument is quite similar to that of Data2Vec, a general framework for self-supervised pretraining of unimodal image,
text and audio models @data2vec. The authors introduce "contextualized representations", which are the activations of all layers of a model
for each time step of the input. Because of Self-Attention in Transformers, the activations for each image patch (time step) are influenced by
all other image patches, and therefore not only encode information about a patches content, but also about its context in the image, i.e. the
relationship to other patches. Consequently, contextualized representations are more informative than a single cls token, as they encode
information about the image at different levels of abstraction, and how the model aggregates low level features to high level concepts.
Since the goal of KD is to "mimic" the behavior of a teacher model for a given input in a compressed way, this is the exact information
that should be transferred from the teacher to the student. Simply predicting the cls token would only "mimic" what information the teacher
extracts from the image, but not how the information is extracted.

While the dimensions of our student model match those of the teacher model, they both have a hidden size of $d=768$ and intermediate size of $d_("ff")=3072$
for the feed-forward layers in the Transformer blocks, the number of layers in the student model ($L_s=12$) is only half of that of the teacher model ($L_t=12$).
It is therefore not possible for each layer of the student model to mimic the behavior of the corresponding layer in the teacher model.
Fortunately, experiments of the Data2Vec authors show that predicting the mean of all layer activations for each time step works as well as predicting
the activations of each layer individually @data2vec. This suits our approach well, as the only mismatch between the teacher and student model is the number
of layers, which is irrelevant when predicting the mean of all layer activations for each time step.
The authors apply instance normalization to the activations of each layer before averaging. Instance normalization is a normalization technique
that works on each dimension of a sequence independently. For a sequence of embeddings/representations with length $T$, instance normalization
is defined as follows:

$
h'_(j,d) = (h_(j,d) - mu_d) / sqrt(sigma_d^2 + epsilon), wide mu_k = 1/T sum_(t=1)^T h_(t,k), wide sigma_k^2 = 1/T sum_(t=1)^T (h_(t,k) - mu_k)^2
$

Even though the formula might look complicated, it is quite simple in practice. For each embedding dimension $d$, the mean $mu_d$ and standard deviation
$sigma_d$ are calculated over all time steps $T$. In the case of an embedding dimension of $D=768$, this means for one sample (e.g. 
a sequence representing an image) 768 means and
standard deviations are calculated, one for each embedding dimension. Then, for each time step $j$, the embedding at time step $i$ is normalized
by normalizing each dimension of the embedding independently, using the corresponding mean and standard deviation computed for that dimension @instance_norm.
During the normalization, a small epsilon, e.g. $1e^(-8)=10^(-8)$, is added to the standard deviation to prevent division by zero.
For an illustrative comparison between instance normalization, batch normalization and layer normalization, see (TODO: cite normalization) in the appendix.
We define the operation $op("InstanceNorm")(dot)$ as instance normalization on a sequence of embeddings $bold(H)$.
$
op("InstanceNorm")(bold(H)) = [bold(h)'_1, bold(h)'_2, ..., bold(h)'_T]
$

After instance norm and averaging, parameter-less layer normalization is performed.
We perform all three operations likewise.
The target and prediction are therefore given by:

$
bold(H)'^t_(v, l) &= op("InstanceNorm")(bold(H)_(v, l)^t), l in {1, 2, ..., L_t} \
bold(H)'^s_(v, l) &= op("InstanceNorm")(bold(H)_(v, l)^s), l in {1, 2, ..., L_s}
$

$
bold(hat(H))^t_v &= 1/L_t sum_(l=1)^(L_t) bold(H)'^t_(v, l) \
bold(hat(H))^s_v &= 1/L_s sum_(l=1)^(L_s) bold(H)'^s_(v, l)
$

$
bold(Y) &= [bold(y)_mono(["I_CLS"]), bold(y)_1, ..., bold(y)_N] = op("LayerNorm")(bold(hat(H))^t_v) \
bold(hat(Y)) &= [bold(hat(y))_mono(["I_CLS"]), bold(hat(y))_1, ..., bold(hat(y))_N] = op("LayerNorm")(bold(hat(H))^s_v)
$

The loss for a single sample (image) is defined in the following:

$
cal(L)_("KD")(bold(Y), bold(hat(Y))) = ||bold(Y) - bold(hat(Y))||_2^2 = 1/(N+1) ( sum_(n=1)^N cal(L)_("MSE")(bold(y)_n, bold(hat(y))_n)
+ cal(L)_("MSE")(bold(y)_mono(["I_CLS"]), bold(hat(y))_mono(["I_CLS"])))
$

We denote $bold(y)_i$ and $bold(hat(y))_i$ as the average representation for image patch $i$ over all layers from the teacher and student model, respectively.
This includes instance norm before averaging, and layer norm after averaging.
For the definition of $op("LayerNorm")(dot)$, see (TODO: cite notation).
$cal(L)_("MSE")(dot, dot)$ is the mean squared error between two d-dimensional vectors, defined in (TODO: cite equation).

==== Pretraining

==== Finetuning
To get a sense of how well the student model has learned from the teacher, we evaluate the student model by finetuning it on the downstream 
image classification tasks of CIFAR-10 @cifar_10_100 and CIFAR-100 @cifar_10_100 and ImageNet-1K @imagenet.
For that, we load the trained student model and add Layer Normalization and a linear classifier on top of it.
The output of the student model is a sequence of embeddings, one embedding for each image patch, and one cls token embedding.
We follow the approach of Data2Vec @data2vec @data2vec2 and BEiTv2 @beitv2, and take the mean over all
patch embeddings as the output of the student model, which is then
passed to the layer normalization and linear classifier.
For all three tasks we perform full finetuning, i.e. we finetune all layers of the student model on the target task, and
linear probing, i.e. we only traing the added linear classifier on top of the student model. For pytorch pseudocode of linear probing
and full finetuning, see (TODO: cite pseudocode).

For data augmentation during finetuning, we use RandAugment @randaugment, mixup @mixup and cutmix @cutmix augmentation, and random erasing @randerase.
The hyperparameters for these augmentations are provided in (TODO: cite hyperparameters), and have been selected based on the values used in
BEiTv2 @beitv2, Data2Vec @data2vec, and Data2Vec2 @data2vec2. We refrain from explaining the augmentations in detail here, as they are
well documented in the respective papers. As for distillation, we use one RTX 4090 24GB GPU for finetuning.

For finetuning of ImageNet-1K, we use a base learning rate of 1-e3 in combination with layer decay. Layer decay is a technique to reduce the
base learning rate for each layer of the model by a certain factor. The goal is to have lower learning rates for layers closer to the input,
and higher learning rates for layers closer to the output. This ensures that low-level features learned during pretraining or distillation
are not destroyed during finetuning. We use a decay factor of 0.81, which is derived by scaling the layer decay used in Data2Vec2 @data2vec2,
from which we extract layers for the student model, by the square root. We use scaling instead of the value used in Data2Vec2, which is 0.65
($sqrt(0.65) approx 0.81$),
as we only have half the number of layers in the student model, and can therefore afford larger learning rates for the lower layers.
The actual learning rates a layer $l$ is then calculated by:

$
"lr"_(l) = "base_lr" * "layer_decay"^(L_s + 1 - l)
$

The learning rates can be seen in the following table:

#figure(
  table(
    columns: 9,
    stroke: none,
    table.hline(),
    table.header(
      [*Layer no.*],
      [0],
      [_1_],
      [_2_],
      [_3_],
      [_4_],
      [_5_],
      [_6_],
      [7],
    ),
    table.hline(stroke: .6pt),
    [*Learning rate*], [2.3e-4], [2.8e-4], [3.5e-4], [4.3e-4], [5.3e-4], [6.6e-4], [8.1e-4], [1e-3],
    table.hline(),
  ),
  caption: [Cursive layer numbers indicate Transformer blocks. The learning rates are calculated using a base learning rate of 1e-3 and a layer decay of 0.81.],
)<unimodal_kd_imagenet_finetuning_layer_decay_lr>

=== Language
==== Method
==== Pretraining
==== Finetuning

#bibliography("../references.bib")