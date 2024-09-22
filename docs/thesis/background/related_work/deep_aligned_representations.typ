== Related Work <related_work>
=== Deep Aligned Representations <shre_section>

The motivation for the knowledge distillation driven approach in this work is
provided by the paper “See, Hear, and Read: Deep Aligned Representations”
by Aytar et al. (2017) @shre. For simplicity, this paper will be referred to as “SHRe”
(for #text(weight: "bold")[S]ee, #text(weight: "bold")[H]ear, #text(weight: "bold")[Re]ad) in this work.

In SHRe, the authors propose a method to align representations of image, text, and audio through knowledge distillation from a
supervised image model. The student model is a multimodal model with separate modality-specific encoders for image, text,
and audio, with a shared encoder on top. The approach utilizes 1D convolutions for audio and text, and 2D convolutions for images.
The output feature maps of these encoders are flattened and then passed separately through a shared encoder, consisting of
3 linear layers @shre.
The approach is generally independent of the specific architecture of the components (encoders), meaning that any architecture can be used.
Notice how the aforementioned is similar to the definition of a multimodal model as defined in @multimodal_models.

The teacher model was trained in a supervised manner, with the authors utilizing a model pretrained on ImageNet-1K, though it is 
not specified what exact model was used. The training objective is to minimize the KL-Divergence between the teacher and student models.

Specifically, the method involves using image-text ${bold(x)^v, bold(x)^w}$ and image-audio ${bold(x)^v, bold(x)^a}$ pairs.
$bold(x)^v$ is a 2D image, $bold(x)^w$ a sequence of text tokens, and $bold(x)^a$ a spectrogram of audio.
For each pair, the image $bold(x)^v$ is passed through the teacher model $g(dot)$, producing a probability distribution over the
ImageNet-1K classes (1000 classes), denoted as $g(bold(x)^v)$. The same image $bold(x)^v$ is also passed through the image encoder
$f_(v)(dot)$ of the student model, followed by the shared encoder $s(dot)$, also resulting in a probability distribution over the ImageNet-1K classes,
defined as $s(f_(v)(bold(x)^v))$.

The other part of the pair, for example, the text $bold(x)^w$ in an image-text pair,
is passed through the text encoder $f_(w)(dot)$ of the student model, and then through the shared encoder $s(dot)$.
Since the shared encoder is the same as the one used for the image, the output is, again, a probability distribution
over the ImageNet-1K classes, represented as $s(f_(w)(bold(x)^w))$.

The probability distribution generated by the teacher model for the image can be compared with the probability distribution
produced by the student model for the same image, using KL-Divergence. This is the usual, response based,
approach to knowledge distillation, defined in @response_based_knowledge_distillation.
What makes the approach unique, however, is that the probability distribution
of the teacher model for the image can be compared with the probability distribution of the student model for the text.
For a single image-text pair, the loss is defined as:

$
cal(L)_("KD")^("vw") = 1/2 * D_("KL")(g(bold(x)^v) || s(f_(v)(bold(x)^v))) + 1/2 * D_("KL")(g(bold(x)^v) || s(f_(w)(bold(x)^w)))
$

With $D_("KL")$ being the KL-Divergence. The loss changes accordingly for image-audio pairs, where the probability
distribution over audio is defined as $s(f_(a)(bold(x)^a))$.

$
cal(L)_("KD")^("va") = 1/2 * D_("KL")(g(bold(x)^v) || s(f_(v)(bold(x)^v))) + 1/2 * D_("KL")(g(bold(x)^v) || s(f_(a)(bold(x)^a)))
$


The goal of this approach is to make the probability distributions between teacher and student as similar as possible.
Since an image and its corresponding text in an image-text pair describe the same real-world concept, 
the distribution of the teacher model for the image, over the ImageNet-1K classes, can directly be transferred to the
caption of the image. That way, the model can learn
to output the same probabilities over the ImageNet-1K classes for both the image and the text. This enables
the alignment of modalities at the level of real-world objects. The same process can be applied to image-audio pairs,
allowing the model to align representations across multiple modalities. A visualization of this
is shown in @shre_coco_prob_dist.

Even though all modalities share the same shared encoder $s(dot)$, the output of the intermediate layers in the 
shared encoder will still differ for each modality.
This is because KL-Divergence only ensures alignment at the level of classes, which corresponds to the output
layer (the last fully-connected layer of the shared encoder outputs the probability distribution over ImageNet-1K classes). 
The internal representations in $s(dot)$, meaning the first two layers, can still be different
between the modalities of a pair.
They can vary, as long as the resulting probability distribution of the last fully-connected/linear layer
is the same as the teacher model's output.

#figure(
  image(
  width: 75%,
  "../../figures/shre.png"),
  caption: [Illustration of the SHRe approach. The model is trained to output the same probability
  distribution over ImageNet-1K classes between images, image-text pairs, and image-audio pairs. Internal representations
  are aligned using a ranking loss @shre.
  Image, text, and audio are always passed individually through the model.
  For each input, the model outputs a probability distribution over the ImageNet-1K classes
  (illustrated distributions are smaller for better visibility).
  The figure does not originate from the original paper, but is a
  custom visualization of the concept. Image and text example is taken from the MSCOCO train set @coco, the spectogram originates
  from the SHRe paper @shre.],
) <shre_fig>

However, the shared encoder is meant to have the same internal representation for e.g. an image and its caption/text:
Since they describe the same concept, the activations in the shared encoder should be similar,
which is, as described in @contrastive_learning, crucial for tasks such as retrieval.
To achieve this, the authors add a ranking loss to the training,
which functions similarly to a contrastive loss. This ranking loss drives the representations of inputs from the same pair
closer together, while pushing the representations of inputs from different pairs further apart. It is defined as:

$
cal(L)_("Rank") = sum_(i=1)^B sum_(j eq.not i) max{0, Delta - cos(bold(x)^v_i, bold(x)_i) + cos(bold(x)^v_i, bold(x)_j)}
$

Here, $B$ represents the batch size, $bold(x)^v_i$ is an image, and $bold(x)_i$ is the corresponding text or audio, depending
if an image-text or image-audio pair is used. $j$ iterates over negative samples in the batch ($j eq.not i$).

Different from contrastive loss, for a given input, e.g. an image, the ranking loss does not normalize the similarity scores
of a positive pair (e.g. image-text) with respect to all other possible pairings (all other texts) for a sample (image) in the batch.
The authors did not provide intuitions for the choice of the ranking loss over the contrastive loss, and we can only assume
that since the paper was published in 2017 @shre, the contrastive loss was not as widely adapted as it is today.

The final loss is a combination of the KL-Divergence loss and the ranking loss:

$
cal(L)_("SHRe") = cal(L)_("KD") + cal(L)_("Rank")
$

The authors evaluate SHRe on retrieval tasks, and the results (@shre_retrieval) show that SHRe performs significantly better than a random baseline.
Interestingly, even though the model is only trained on image-text and image-audio pairs, the alignment also generalizes to text-audio pairs,
and the model can retrieve text-audio pairs, albeit not as well as between the modalities it was trained on @shre. This indicates that the image
modality acts as an anchor between text and audio, enabling the model to align representations between modalities it was not explicitly trained on.
The alignment between modalities becomes transitive.

#figure(
  table(
  columns: 13,
    stroke: none,
    table.hline(),
    table.header(
      table.cell(rowspan: 4, colspan: 1, align:horizon, [*Model*]),
      table.cell(colspan: 4, [*MSCOCO*]),
      table.cell(colspan: 4, [*Flickr (Custom)*#footnote[Datasets used consists of videos collected from Flickr, from which frames were extracted and used as images with the corresponding audio @shre.]]),
      table.cell(colspan: 4, [*Unspecified*#footnote[Data has been collected and annotated using Amazon Mechanical Turk @shre @amazon_mechanical_turk. Where the data originates from is not specified in the paper.]]),
      table.cell(colspan: 2, [Image]),
      table.cell(colspan: 2, [Text]),
      table.vline(stroke: .4pt),
      table.cell(colspan: 2, [Image]),
      table.cell(colspan: 2, [Sound]),
      table.vline(stroke: .4pt),
      table.cell(colspan: 2, [Text]),
      table.cell(colspan: 2, [Sound]),
      table.cell(colspan: 2, [$arrow.b$]),
      table.cell(colspan: 2, [$arrow.b$]),
      table.cell(colspan: 2, [$arrow.b$]),
      table.cell(colspan: 2, [$arrow.b$]),
      table.cell(colspan: 2, [$arrow.b$]),
      table.cell(colspan: 2, [$arrow.b$]),

      table.cell(colspan: 2, [Text]),
      table.cell(colspan: 2, [Image]),
      table.cell(colspan: 2, [Sound]),
      table.cell(colspan: 2, [Image]),
      table.cell(colspan: 2, [Sound]),
      table.cell(colspan: 2, [Text]),
    ),
    table.hline(stroke: .4pt),
    [Random], table.cell(colspan: 2, [500]), table.cell(colspan: 2, [500]), table.cell(colspan: 2, [500]), table.cell(colspan: 2, [500]), table.cell(colspan: 2, [500]), table.cell(colspan: 2, [500]),
    [SHRe], table.cell(colspan: 2, [5.8]), table.cell(colspan: 2, [6.0]), table.cell(colspan: 2, [47.5]), table.cell(colspan: 2, [47.8]), table.cell(colspan: 2, [135.0]), table.cell(colspan: 2, [140.5]),
    table.hline(),
  ),
  caption: [Retrieval results of SHRe on different datasets. Each dataset contains 5k sample pairs (e.g. image-text pairs) for evaluation,
  and is splitted into 5 chunks of 1k samples each. Retrieval is then performed on each chunk, and metric used is
  the median rank of the correct pair in the ranked list. The median rank is averaged over all chunks for each datasets, so the results
  seen describe the average median rank over all chunks for each dataset. The results are taken from the SHRe paper @shre.],
)<shre_retrieval>

The approach is illustrated in @shre_fig. It is important to note that SHRe 
is only trained with image-text and image-audio pairs, and not, how it might seem from the figure, with image-text-audio triplets.

The SHRe approach is a crucial foundation for this work, as it demonstrates how the knowledge
from a *supervised* unimodal (image) model can be _extracted_ and _transferred_ to a multimodal model.
