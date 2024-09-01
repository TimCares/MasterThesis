#set math.equation(numbering: "(1)")

== Unimodal Knowledge Distillation
To validate whether traditional unimodal knowledge distillation, an undoubtedly simpler task than multimodal knowledge distillation, even works,
we will first conduct experiments on unimodal knowledge distillation. We will then build on the results to develop a multimodal knowledge distillation.
=== Vision

==== Method
Our approach to vision KD involves using a pretrained Data2Vec2 @data2vec2 image model as the teacher model, and distilling
a shallow version of this model, which is the student. 
We attribute our choice of Data2Vec2 to its effectiveness and consistency in self-supervised learning across image, text and audio.
Data2Vec2 is a general framework to pretrained _unimodal_ image, text, and audio models using self-supervised learning @data2vec @data2vec2, which
fits our philosophy of aligning modalities.

We approach the distillation by taking the first 6 Transformer blocks of the _pretrained_ teacher model, which are exactly half
of the 12 Transformer blocks in the teacher,
and organize them into a smaller model. This smaller model also includes the pretrained cls token, patch projection, and positional encodings.
Consequently, the student model is not trained from scratch, but already initialized with a subset of the teacher's weights.
This is inspired by DistilBERT @distilbert, a small BERT variant distilled from the normal BERT model,
selecting every second layer from a pretrained BERT @bert model and organizing them into
a smaller model.
As mentioned before, we use the first 6 Transformer blocks of the teacher model, which we found leads to better results than using every second layer.
The resulting student model is with 43.1M parameters almost half the size of the teacher model, which has 85.9M parameters.

Data2Vec2 is a self-supervised model @data2vec2, and therefore does not provide a probability distribution over classes
that can be predicted using KL-Divergence.
Instead, we only have access to the model's activations for each layer, so we have to resort to feature-based knowledge distillation.
One option would be to predict the teacher's output for the cls token $bold(h)^t_(v, L, mono(["I_CLS"]))$, which aggregates the high level
content of the image, and then use the mean squared error as the loss function.
However, this neglects the activations for individual image patches and activations of intermediate layers.

This argument is quite similar to that of Data2Vec. The authors introduce "contextualized representations", which are the activations of all layers of a model
for each time step of the input. Because of Self-Attention in Transformers, the activations for each image patch (time step) are influenced by
all other image patches, and therefore not only encode information about a patches content, but also about its context in the image, i.e. the
relationship to other patches @data2vec @data2vec2. Consequently, contextualized representations are more informative than a single cls token, as they encode
information about the image at different levels of abstraction, and how the model aggregates low level features to high level concepts.
Since the goal of KD is to "mimic" the behavior of a teacher model for a given input in a compressed way, this is the exact information
that should be transferred from the teacher to the student. Simply predicting the cls token would only "mimic" what information the teacher
extracts from the image, but not how the information is extracted.

While the dimensions of our student model match those of the teacher model, they both have a hidden size of $D=768$ and intermediate size of $D_("ff")=3072$,
the number of layers in the student model ($L_s=6$) is only half of that of the teacher model ($L_t=12$).
It is therefore not possible for each layer of the student model to mimic the behavior of the corresponding layer in the teacher model.
Fortunately, experiments of the Data2Vec authors show that predicting the mean of all layer activations for each time step (or image patch, respectively)
works as well as predicting
the activations of each layer individually @data2vec. This suits our approach well, as the only mismatch between the teacher and student model is the number
of layers, which is irrelevant when predicting the mean of all layer activations for each time step.
The authors apply instance normalization to the activations of each layer before averaging, which is a normalization technique
that works on each dimension of a sequence independently.

For a sequence of embeddings/representations with length $T$, instance normalization
is defined as follows:

$
h'_(j,d) = (h_(j,d) - mu_d) / sqrt(sigma_d^2 + epsilon), wide mu_k = 1/T sum_(t=1)^T h_(t,k), wide sigma_k^2 = 1/T sum_(t=1)^T (h_(t,k) - mu_k)^2
$

Even though the formula might look complicated, it is quite simple in practice. For each embedding dimension $d$, the mean $mu_d$ and standard deviation
$sigma_d$ are calculated over all time steps $T$. In the case of an embedding dimension of $D=768$, this means for one sample (e.g. 
a sequence representing an image) 768 means and
standard deviations are calculated, one for each embedding dimension. Then, for each time step $j$, the embedding at time step $j$ is normalized
by normalizing each dimension of the embedding independently, using the corresponding mean and standard deviation computed for that dimension @instance_norm.
During the normalization, a small epsilon, e.g. $1e^(-8)=10^(-8)$, is added to the standard deviation to prevent division by zero.
For an illustrative comparison between instance normalization, batch normalization and layer normalization, see (TODO: cite normalization) in the appendix.
We define the operation $op("InstanceNorm")(dot)$ as instance normalization on a sequence of embeddings $bold(H)$.
$
op("InstanceNorm")(bold(H)) = [bold(h)'_1, bold(h)'_2, ..., bold(h)'_T]
$

After instance norm and averaging, parameter-less layer normalization is performed @data2vec @data2vec2.
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
$ <unimodal_kd_data2vec2_loss>

We denote $bold(y)_i$ and $bold(hat(y))_i$ as the average representation for image patch $i$ over all layers from the teacher and student model, respectively.
This includes instance norm before averaging, and layer norm after averaging.
For the definition of $op("LayerNorm")(dot)$, see (TODO: cite notation).
$cal(L)_("MSE")(dot, dot)$ is the mean squared error between two vectors, defined in (TODO: cite equation).

==== Distillation
We distill the student model by minimizing the loss defined in @unimodal_kd_data2vec2_loss
using the AdamW optimizer @adamW with a base learning rate of 5e-4. We train for 10 epochs with a batch size of 256 on the training set
of ImageNet-1K @imagenet, and run validation after every epoch on the validation set of ImageNet-1K. As Data2Vec2
our approach does not involve labels, we use the loss defined in @unimodal_kd_data2vec2_loss also for validation.
The total number of parameters involved in the distillation process is 129M, of which 43.1M trainable
belong to the student model, and 85.9M frozen parameters to the teacher model.

Since we use the same architecture as Data2Vec2 for our teacher model, the images, being of size 224$times$224, which is the size we will
consistently use for all experiments, are split into 16x16 patches, which results in a sequence length of $N=196$. A cls token is added to the sequence,
which results in a total sequence length of $N+1=197$. All embeddings have a dimension of $D=768$.

For data augmentation we decide to use the same augmentation strategy using during the training of the teacher model. This ensures that
we get the training targets from the same distribution the teacher has seen, and that we do not generate data for which the teacher might
give inaccurate representations. The augmentations involve (1) cropping a random portion of an image and resizing it back to the original
image resolution (224$times$224), (2) performing a random horizontal flip with probability 0.5, and (3) normalizing the image RGB channels
with the mean and standard deviation of the ImageNet-1K dataset @data2vec2.

Detailed hyperparameters are provided in (TODO: cite hyperparameters).

We show the evolution of the training and validation loss during training in @distil_d2v2_loss. We observe the traditional convergence behavior
of a model during training, and the validation loss is consistently lower than the training loss, which is a sign of good generalization.
There are some peaks in the training loss, which are likely due to a high learning rate, but they do not affect the validation loss, which is
why do not investigate them further.
As we do not have access to any other metric than the MSE loss during training we have to evaluate the student by finetuning
on downstream tasks, which follows in the next section. This will answer whether the distillation actually yields a model
competetive in performance to the teacher model and if the knowledge transfer was successful.


#figure(
  image(
  width: 50%,
  "../figures/distil_d2v2_loss.png"),
  caption: [Training loss vs. validation loss during distillation of the Data2Vec2 image model.
],
) <distil_d2v2_loss>


==== Finetuning
To get a sense of how well the student model has learned from the teacher, we evaluate the student model by finetuning it on the downstream 
image classification tasks of CIFAR-10 @cifar_10_100, CIFAR-100 @cifar_10_100 and ImageNet-1K @imagenet.
For that, we load the trained student model and add Layer Normalization and a linear classifier on top of it.
The output of the student model is a sequence of embeddings, one embedding for each image patch, and one cls token embedding.
We follow the approach of Data2Vec @data2vec @data2vec2 and BEiTv2 @beitv2, and take the mean over all
patch embeddings as the output of the student model, which is then
passed to the layer normalization and linear classifier (cls token embedding is ingnored).
For all three tasks we perform full finetuning, i.e. we finetune all layers of the student model on the downstream task, and
linear probing, we only train the added layer norm and linear classifier on top of the student model. For pytorch pseudocode of linear probing
and full finetuning see (TODO: cite pseudocode).

For data augmentation during finetuning we use RandAugment @randaugment, mixup @mixup and cutmix @cutmix augmentation, and random erasing @randerase.
The hyperparameters for these augmentations are provided in (TODO: cite hyperparameters), and have been selected based on the values used in
BEiTv2 @beitv2, Data2Vec @data2vec, and Data2Vec2 @data2vec2. We refrain from explaining the augmentation techniques in detail here, as they are
well documented in the respective papers.

For finetuning on ImageNet-1K we use a base learning rate of 1-e3 in combination with layer decay. Layer decay is a technique to reduce the
base learning rate for each layer of the model by a certain factor. The goal is to have lower learning rates for layers closer to the input,
and higher learning rates for layers closer to the output. This ensures that low-level features learned during pretraining or distillation
are not destroyed during finetuning. We use a decay factor of 0.81, which is derived by scaling the layer decay used in Data2Vec2 @data2vec2,
from which we extract layers for the student model, by the square root. We use scaling instead of the value used in Data2Vec2, which is 0.65
($sqrt(0.65) approx 0.81$),
as we only have half the number of layers in the student model, and can therefore afford larger learning rates for the lower layers.
The actual learning rate for a layer $l$ is then calculated by:

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
  caption: [Learning rates for different blocks/layers when finetuning on ImageNet-1K.
  Cursive layer numbers indicate Transformer blocks. The learning rates are calculated using a base learning rate of 1e-3 and a layer decay of 0.81.],
)<unimodal_kd_imagenet_finetuning_layer_decay_lr>

We show the learning rates for 8 layers in total, even though the student model only has 6 Transformer blocks. This is because we count all weights
before the first Transformer block as layer 0, which includes the weights used for projecting patches to embeddings, the cls token, and the positional
encodings. Correspondingly, layer 7 includes the weights for the layer norm and linear classifier on top of the student model, which are initialized
randomly and can be assigned a higher learning rate than the other layers.

For all hyperparameters used on the downstream tasks, see (TODO: cite hyperparameters).

The results, displayed in @distil_d2v2_imagenet_results and @distil_d2v2_cifar_results, show that while
the student model is not able to outperform the teacher model,
as well as all other models we compare to, it is able to achieve acceptable performance on all 6 evaluations considering both BEiTv2 and
Data2Vec2 are based on the ViT-B/16 architecture @data2vec2 @beitv2, which is twice as large as the student model. We also compare to the ResNet-101 model
from the original ResNet paper @resnet, which has 44.5M parameters, and is therefore comparable in size to the student model, but has been trained
only supervised.

#figure(
  table(
    columns: 3,
    stroke: none,
    table.hline(),
    table.header(
      [*Method*],
      [*Finetune*],
      [*Linear Probe*],
      table.hline(stroke: .6pt),
    ),
    [Data2Vec2], [84.5],[-], 
    [BEiTv2], [85.0], [80.1],
    [ResNet-101], [80.1], [-], 
    [*DistilData2Vec2 (ours)*], [75.0], [75.0],
    table.hline(stroke: .6pt),
  ),
  caption: [Comparison of finetuning and linear probing results with SOTA self-supervised models on ImageNet-1K.],
)<distil_d2v2_imagenet_results>

#figure(
  table(
    columns: 3,
    stroke: none,
    table.hline(),
    table.header(
      [*Dataset*],
      [*Approach*],
      [*Accuracy*],
      table.hline(stroke: .6pt),
    ),
    table.cell(rowspan: 2, align: horizon, [CIFAR-10]), [Finetune],[-], 
    [Linear Probe],[-], 
    table.cell(rowspan: 2, align: horizon, [CIFAR-100]), [Finetune],[-], 
    [Linear Probe],[-], 
    table.hline(stroke: .6pt),
  ),
  caption: [Results of finetuning and linear probing of our distilled Data2Vec2 image model on CIFAR-10 and CIFAR-100.],
)<distil_d2v2_cifar_results>

=== Language

==== Method
For knowledge distillation of a language model, we decide against the intuitive choice of distilling the corresponding Data2Vec2 language model,
and opt for distilling a BERT model instead. We do this for two reasons.
First, for reasons later explained later, we will also use BERT as the text encoder in our multimodal model, so for consistency
we will use BERT for the unimodal distillation as well. Second, as mentioned before, there already exists a distilled version of BERT, DistilBERT @distilbert,
to which we can directly compare our results.

We use the same approach as for the image model, and distill a smaller version of BERT from a pretrained BERT model.
Different to DistilBERT, we again take the first 6 Transformer blocks of the teacher model, and organize them into a smaller model together with
the embedding layer and positional encodings. The student model is therefore, again, initialized with a subset of the teacher's weights.

The distillation loss is defined analogously to the distilled image model, and is defined in @unimodal_kd_data2vec2_loss. We do not need to change
anything, as the loss is applicable to any Transformer, regardless of the modality, making it a universal loss function for feature-based knowledge distillation.
This follows Data2Vec2, which uses the same loss @data2vec2.
==== Distillation

The BERT model is distilled on a subset of the OpenWebText dataset @openwebtext, introduced in (TODO: cite data), of which
the text is tokenized into subwords using the BERT tokenizer.
During training, the tokenized text of the dataset is split into sequences of 196 tokens, which is the maximum sequence length
that can fit on a single GPU with 24GB of memory (RTX 4090).

We validate the student model on the dedicated validation dataset of OWT we introduced in (TODO: cite data). The same loss as used as for training.
==== Finetuning

#bibliography("../references.bib")