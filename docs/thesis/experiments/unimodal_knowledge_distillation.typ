= Experiments
== Unimodal Knowledge Distillation <unimodal_knowledge_distillation>
To validate whether traditional unimodal knowledge distillation, an undoubtedly simpler task than multimodal knowledge distillation, even works,
we will first conduct experiments on unimodal knowledge distillation. We will then build on the results to develop a multimodal knowledge distillation.
=== Vision <unimodal_kd_vision>

==== Method <unimodal_kd_data2vec2_method>
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
For an illustrative comparison between instance normalization, batch normalization and layer normalization, see @norm_comparison in the Appendix.
We define the operation $op("IN")(dot)$ as instance normalization on a sequence of embeddings $bold(H)$.
$
op("IN")(bold(H)) = [bold(h)'_1, bold(h)'_2, ..., bold(h)'_T]
$

After instance norm and averaging, parameter-less layer normalization, denoted as $op("LN")(dot)$, is performed @data2vec @data2vec2.
We perform all three operations likewise.
The target and prediction are therefore given by:

$
bold(H)'^t_(v, l) &= op("IN")(bold(H)_(v, l)^t), l in {1, 2, ..., L_t} \
bold(H)'^s_(v, l) &= op("IN")(bold(H)_(v, l)^s), l in {1, 2, ..., L_s}
$

$
bold(hat(H))^t_v &= 1/L_t sum_(l=1)^(L_t) bold(H)'^t_(v, l) \
bold(hat(H))^s_v &= 1/L_s sum_(l=1)^(L_s) bold(H)'^s_(v, l)
$

$
bold(Y) &= [bold(y)_mono(["I_CLS"]), bold(y)_1, ..., bold(y)_N] = op("LN")(bold(hat(H))^t_v) \
bold(hat(Y)) &= [bold(hat(y))_mono(["I_CLS"]), bold(hat(y))_1, ..., bold(hat(y))_N] = op("LN")(bold(hat(H))^s_v)
$

The loss for a single sample (image) is defined in the following:

$
cal(L)_("KD")(bold(Y), bold(hat(Y))) &= ||bold(Y) - bold(hat(Y))||_2^2 = \
1/(N+1) ( sum_(n=1)^N cal(L)_("MSE")&(bold(y)_n, bold(hat(y))_n)
+ cal(L)_("MSE")(bold(y)_mono(["I_CLS"]), bold(hat(y))_mono(["I_CLS"])))
$ <unimodal_kd_data2vec2_loss>

We denote $bold(y)_i$ and $bold(hat(y))_i$ as the average representation for image patch $i$ over all layers from the teacher and student model, respectively.
This includes instance norm before averaging, and layer norm after averaging.
$cal(L)_("MSE")(dot, dot)$ is the mean squared error between two vectors, defined in @mean_squared_error.

==== Distillation <unimodal_kd_data2vec2_distillation>
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

Detailed hyperparameters are provided in @distil_data2vec2_hyperparameters.

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
and full finetuning see @image_downstream_forward_pseudocode.

For data augmentation during finetuning we use RandAugment @randaugment, mixup @mixup and cutmix @cutmix augmentation, and random erasing @randerase.
The hyperparameters for these augmentations are provided in @distil_data2vec2_imagenet_finetuning_hyperparameters,
and have been selected based on the values used in
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

For all hyperparameters used on the downstream tasks, see @distil_data2vec2_imagenet_finetuning_hyperparameters.

The results, displayed in @distil_d2v2_imagenet_results and @distil_d2v2_cifar_results, show that while
the student model is not able to outperform the teacher model (Data2Vec2),
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
    [*DistilData2Vec2 (ours)*], [75.0], [56.2],
    table.hline(),
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
    table.cell(rowspan: 2, align: horizon, [CIFAR-10]), [Finetune],[97.0], 
    [Linear Probe],[68.4], 
    table.hline(stroke: .3pt),
    table.cell(rowspan: 2, align: horizon, [CIFAR-100]), [Finetune],[85.1], 
    [Linear Probe],[46.2], 
    table.hline(),
  ),
  caption: [Results of finetuning and linear probing of our distilled Data2Vec2 image model on CIFAR-10 and CIFAR-100.],
)<distil_d2v2_cifar_results>

=== Language

==== Method
For knowledge distillation of a language model, we decide against the intuitive choice of distilling the corresponding Data2Vec2 language model,
and opt for distilling a BERT model instead. We do this for two reasons.
First, we will also use BERT as the text encoder in our multimodal model, so for consistency
we will use BERT for the unimodal distillation as well. Second, as mentioned before, there already exists a distilled version of BERT, DistilBERT @distilbert,
to which we can directly compare our results.

We use the same approach as for the image model, and distill a smaller version of BERT from a pretrained BERT model.
As with our image model, we take the first 6 Transformer blocks of the teacher model, and organize them into a smaller model together with
the embedding layer and positional encodings. The student model is therefore, again, initialized with a subset of the teacher's weights.

The distillation loss is defined analogously to the distilled image model, and is defined in @unimodal_kd_data2vec2_loss. We do not need to change
anything, as the loss is applicable to any Transformer, regardless of the modality, making it a universal loss function for feature-based knowledge distillation.
This follows the philosophy of Data2Vec2, which uses the same loss for all modalities @data2vec2.

==== Importance of Sequence Length

Setting the maximum sequence length as long as possible is crucial when regressing the mean activation of
all teacher layers for each token. Self-attention mechanisms enable each token to encode not only its own
information but also contextual information from other tokens in the sequence. This phenomenon, referred to
as "contextualized targets" by the authors of Data2Vec @data2vec @data2vec2, becomes more pronounced with
longer sequences. A larger number of tokens in the sequence increases the opportunities for tokens to attend
to one another, thereby enriching contextual representations and enhancing the model's ability to capture complex
token relationships.

For instance, in the short phrase "a dog", each token can attend to only one other token, which
limits the model's capacity to understand the broader context and relationships. In contrast, the
sentence "a dog is playing in the garden with a tennis ball" provides a richer context with more tokens.
This expanded context allows the model to capture a wider variety of relationships between tokens, offering
the student model more opportunities to learn from the teacher model's outputs. A visualization of the attention
between the different tokens of both sentences is provided in @bert_attn_examples. Here, we can see that a longer
sequence allows tokens to attend to a high variety of other tokens, leading to relationships that represent real-world
scenarios. The token "dog" attends mostly to tokens "playing", "garden", "tennis" and "ball", which combined represent
the context of the sentence and a real-world scenario. The representation of the token "dog" therefore encodes information
about a dog and what it is doing, which can be learned by the student model when regressing the teacher's outputs. In contrast,
the token "dog" in the short sentence "a dog" can only attend to the token "a" (and the special tokens
$mono(["CLS"])$ and $mono(["SEP"])$), which does not provide any context about the
dog.
However, the disadvantage of longer sequences is that they require more memory and computational resources, as the self-attention
mechanism has a quadratic complexity with respect to the sequence length.

A variable sequence length is less relevant for the image model in @unimodal_kd_vision,
as the size of an image is fixed to 224$times$244 pixels, so
there is no variable sequence length.

#figure(
  image(
  width: 90%,
  "../figures/bert_attn_examples.png"),
  caption: [
    Visualization of the attention between tokens in a short and longer sentence. Longer sequences allow for more complex
    relationships between tokens that represent the real-world context of the sentence. Attention scores have been taken from
    the first attention head of the first Transformer layer of the trained uncased BERT model used for distillation.
],
) <bert_attn_examples> 

==== Distillation

During training, we set the maximum sequence length to 256, which is the maximum sequence length
that can fit on a single GPU with 24GB of memory (RTX 4090).

The BERT model is distilled on a subset of the OpenWebText dataset @openwebtext, introduced in @unimodal_data, of which
the text is tokenized into subwords using the BERT tokenizer. We use the uncased variant, meaning all tokens are first converted to lowercase:
To the model, there is no difference between "Dog" and "dog".

We validate the student model on the dedicated validation dataset of OWT we introduced in @unimodal_data.

The model is trained for only one epoch, as we found the model to converge quickly, and because the text data we collect yields
almost 1M batches of size 256, with a sequence length of 256, which is very large. In comparison, the 1.2M images of ImageNet-1K @imagenet
yield 5004 batches of size 256 for a single epoch.

Other hyperparameters used, including the learning rate, are similar to that of the image model, and are provided in @distil_bert_hyperparameters.

Before the loss, defined in @unimodal_kd_bert_loss, is applied, we perform a similar preprocessing as for the image model.
However, following Data2Vec2 @data2vec2, we do not apply layer normalization on the averaged activations, and the loss
is only calculated for non-padding tokens. This was not relevant in the image model, as there is no padding in the image data.

$
bold(H)'^t_(w, l) &= op("IN")(bold(H)_(w, l)^t), l in {1, 2, ..., L_t} \
bold(H)'^s_(w, l) &= op("IN")(bold(H)_(w, l)^s), l in {1, 2, ..., L_s}
$

$
bold(hat(H))^t_w &= 1/L_t sum_(l=1)^(L_t) bold(H)'^t_(w, l) \
bold(hat(H))^s_w &= 1/L_s sum_(l=1)^(L_s) bold(H)'^s_(w, l)
$

$
bold(Y) &= [bold(y)_mono(["T_CLS"]), bold(y)_1, ..., bold(y)_M, bold(y)_mono(["SEP"])] = bold(hat(H))^t_w \
bold(hat(Y)) &= [bold(hat(y))_mono(["T_CLS"]), bold(hat(y))_1, ..., bold(hat(y))_M, bold(hat(y))_mono(["SEP"])] = bold(hat(H))^s_w
$

$
cal(L)'_("MSE")(bold(y)_m, bold(hat(y))_m) &:= cases(
  0 &"if" bold(e)^w_m = bold(t)_mono(["PAD"]) \
  cal(L)_("MSE")(bold(y)_m, bold(hat(y))_m) &"else",
)
$
$
cal(L)_("KD")(bold(Y), bold(hat(Y))) &= ||bold(Y) - bold(hat(Y))||_2^2 = \
1/(M+2) ( sum_(m=1)^M cal(L)'_("MSE")(bold(y)_m, bold(hat(y))_m)
+ cal(L)_("MSE")(bold(y)_mono(["T_CLS"])&, bold(hat(y))_mono(["T_CLS"]))
+ cal(L)_("MSE")(bold(y)_mono(["SEP"]), bold(hat(y))_mono(["SEP"])))
$ <unimodal_kd_bert_loss>

Here, $bold(e)^w_m$ denotes the embedding of token $m$ in the sequence, and the loss is ignored, i.e. 0, if the token is the
padding token, denoted $bold(t)_mono(["PAD"])$.

==== Finetuning
To get a sense of the language understanding capabilities of the trained/distilled student model, we finetune it on all
GLUE benchmark tasks @glue, which are described in @unimodal_data, and visualized in @glue_example.

To perform finetuning, we load the weights of the trained student model, and
add a RoBERTa @roberta classification head on top of it.
This classification head takes the representation of the $mono(["CLS"])$ token, and passes it through one linear layer followed
by a Tanh activation function, and then through another linear layer. The first linear layers retrains the input dimensionality of the
$mono(["CLS"])$ token, which is 768, and the second linear layer maps the representation to the number of classes of the downstream task.
If the task is a regression task, for example sentence similarity in [0, 5] for STS-B @stsb, the second linear layer returns a scalar value.
Pytorch pseudocode for the forward pass is provided in @text_downstream_forward_pseudocode.

Since for most tasks the amount of training data is marginal, e.g. CoLA @cola has only 8.5k training samples, we do not use the layer decay
technique used for the image model, and directly select a very low learning rate. Most of the hyperparameters are inspired by
BERT @bert, and Data2Vec @data2vec @data2vec2, and are provided in @distil_bert_glue_finetuning_hyperparameters. We increase the number
of epochs and lower the batch size if the dataset, like CoLA, is very small. This ensures that we have a sufficient number of updates
for the model to learn from the data.


#figure(
  table(
    columns: 10,
    stroke: none,
    table.hline(),
    table.header(
      [],
      [MNLI],
      [QNLI],
      [RTE],
      [MRPC],
      [QQP],
      [STS-B],
      [CoLA],
      [SST],
      [*Score*],
      table.hline(stroke: .6pt),
    ),
    [BERT @bert],[86.7], [91.8], [69.3], [88.6], [89.6], [92.7], [56.3], [92.7], [83.5],
    [DistilBERT @distilbert],[82.2], [89.2], [59.9], [87.5], [88.5], [86.9], [51.3], [91.3], [79.6],
    [Feature-DistilBERT (ours)],[-], [-], [-], [-], [-], [-], [-], [-], [-],
    table.hline(),
  ),
  caption: [Results for BERT and DistilBERT are taken from the DistilBERT paper @distilbert.],
)<distil_d2v2_glue_results>
