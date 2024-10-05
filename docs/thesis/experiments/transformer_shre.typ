== Multimodal Knowledge Distillation <multimodal_knowledge_distillation>
=== Transformer SHRe <transformer_shre>
Before we develop an end-to-end self-supervised approach to multimodal knowledge distillation, we first follow the approach
of SHRe @shre and train a multimodal model with a probability distribution over the classes as the prediction
target. This allows us to closely observe the impact of switching from a supervised to a self-supervised teacher model on the
student model's performance. Moreover, it allows us to gradually increase the complexity of our approach and build on our
previous advancements.

==== Method
===== Architecture <transformer_shre_architecture>
What makes our approach different from SHRe is that we use a language Transformer as the text encoder, and
a vision Transformer as the image encoder, bringing us closer to a unified architecture.
In contrast, SHRe uses 2D convolutions and 1D convolutions for the image and text, respectively. For now, the shared encoder remains
a 3-layer MLP as in SHRe @shre. Since the output of the image and text encoder is a sequence of features, but we use a normal MLP as the shared encoder,
we have to reduce the sequence of features to a single represenation. This is done by taking the representation of the $mono(["I_CLS"])$
and $mono(["T_CLS"])$
token from the image and text encoder, respectively, and aligns with our first implementation of a multimodal model shown in @multimodal_model.
This represenation, namely $bold(h)_(v, L_s, mono(["I_CLS"]))$
and $bold(h)_(w, L_s, mono(["T_CLS"]))$, is then passed to the shared encoder, which produces the final multimodal representations.
Here, $L_s$ denotes the number of Transformer layers in the image and text encoder, respectively, which is
defined as $L_s=6$.

To keep the model size manageable, we will resort the the same approach as in our unimodal experiment, and use 6 Transformer layers
for the image and text encoder, respectively. This also gives us the opportunity
to directly compare the performance of the image and text encoder from our multimodal model to that of the unimodal models (@unimodal_knowledge_distillation). This can be done by evaluating for instance the image encoder of the multimodal model on image classification tasks, and
then comparing its performance to the results observed for the unimodal image KD model of @unimodal_kd_vision
on the same tasks.
A performance similar to that of the strictly unimodal models would indicate that multimodal pretraining
yields strong unimodal encoders as a byproduct. As argued by the authors of FLAVA @flava, the aforementioned is in fact even a
requirement for multimodal models. This can be attributed to the fact that an alignment in
the shared encoder is only possible if the unimodal encoders generate features from which the shared encoder
can extract modality-invariant information, like the semantic content of an image or text. If the unimodal encoders are not able to extract
high-level features, then neither will the shared encoder be able to extract modality-invariant information,
nor will the features be useful in the corresponding unimodal downstream tasks.

As done in our unimodal experiments, we initialize the image and text encoder using the embeddings, positional encodings,
and the first 6 Transformer layers from Data2Vec2 @data2vec2 and BERT @bert, respectively. The shared encoder will be initialized
randomly. For an illustration of the architecture
we refer to the same depiction used for SHRe, shown in @shre_fig. The only difference is that we only use image and text encoders,
which are now Transformers instead of CNNs.

The shared encoder, which is a 3-layer MLP, is implemented by using the same 2-layer MLP module as present in Transformer layers,
and adding an additional LayerNorm and linear layer on top of it. Given the output from the text encoder, the forward pass of the shared encoder
is then given in the following, and changes for the image encoder accordingly:

$
bold(h)'_(w, K) &= bold(h)_(w, L_s, mono(["T_CLS"]))bold(W)_1^T + bold(b)_1 \
bold(h)''_(w, K) &= op("LN")(op("GELU")(bold(h)'_(w, K)))bold(W)_2^T + bold(b)_2 \
$ <transformer_shre_shared_encoder_ffn_computation>

$
bold(bold(h))'''_(w, K) &= op("LN")(bold(h)''_(w, K))bold(W)_3^T + bold(b)_3 \
$ <transformer_shre_shared_encoder_head_computation>

We consider the 3-layer MLP as a single layer stacked on the image and text encoder, and therefore denote the layer
number as $K=L_s+1=7$.
The operation done in @transformer_shre_shared_encoder_ffn_computation is analogous to the definition of the
MLP layers in a Transformer layer, defined in @transformer_ffn_eq. We choose a different notation here
to also capture the result $bold(h)'_(w, K)$, or $bold(h)'_(v, K)$ for an image, of the first linear layer
(the latter is represented by parameters $bold(W)_1$ and $bold(b)_1$),
which is important when defining the loss in the next section.

The whole model has a total of around 114.2M parameters, which is broken down in @transformer_shre_param_table.

#figure(
  table(
    columns: 2,
    stroke: none,
    table.hline(),
    table.header(
      [*Component*],
      [*\# Params*],
      table.hline(stroke: .6pt),
    ),
    [Image Encoder], [42.3M],
    [Text Encoder], [66.4M], 
    [Shared Encoder], [5.5M],
    table.hline(stroke: .6pt),
    [*Total*], [114.2M],
    table.hline(),
  ),
  caption: [A summary of the number of parameters of the Transformer SHRe model.],
)<transformer_shre_param_table>

The 24M parameters the text encoder has more than the image encoder can be attributed to the embedding matrix of the text encoder, which
alone has 23.4M parameters. Since we use parts of a pretrained BERT model, we also have to resort to using the BERT tokenizer and
vocabulary. The vocabulary consists of 30522 (sub)words, and the embedding matrix has a dimensionality of 768 ($30522*768=23.4M$).
The remaining parameters are related to the BERT-specific implementation of positional encodings.

While 115M parameters can be considered as quite large, considering we strive to build efficient(/smaller) models, 
it is still significantly smaller than the vision-language models we compare to.
For example, VLMo @vlmo has 562M
#footnote[#link("https://github.com/microsoft/unilm/tree/master/vlmo")], CLIP @clip has more than 400M
#footnote[#link("https://huggingface.co/openai/clip-vit-large-patch14")], and BEiT-3 @beit3 has 1.9B parameters @beit3.  

===== Training Objective

Since we start with using a supervised teacher, the loss for knowledge distillation will remain KL-Divergence.
As the application of the KL-Divergence is two-fold, once for the prediction based on the image and once for the prediction based on the caption,
we provide a refined version of the loss function. As a preliminary step, we define the softmax normalization of a vector $bold(u)$ as follows:

$
pi(bold(u)) = bold(z) &= [z_0, z_2, dots, z_(C-1)] in RR^C \
z_i &= exp(u_i) / (sum_(j=0)^(C-1) exp(u_j))
$

This allows us to generate a probability distribution over the classes for the logits generated by the teacher for the image, and for the
logits generated by the student for image and caption. The loss for knowledge distillation is then given by:

$
cal(L)_("KD") &= \ 
1/2 * cal(L)_("KD")^v &+ 1/2 * cal(L)_("KD")^w = \
1/2 * D_("KL")(pi(bold(p)), pi(bold(bold(h))'''_(v, K))) &+ 1/2 * D_("KL")(pi(bold(p)), pi(bold(bold(h))'''_(w, K)))
$

$bold(p)$ denotes the logits generated by the teacher for the image, $bold(bold(h))'''_(v, K)$ the logits generated by the student for the image, and
$bold(bold(h))'''_(w, K)$ the logits generated by the student for the caption. All are in $RR^1000$ for the 1000 classes of ImageNet.

We decide to replace
the ranking loss of SHRe with the contrastive loss introduced by CLIP @clip, and
explained in @vision_language_contrast. We justify this decision with the fact that vision-language contrast has become the
de-facto standard for multimodal self-supervised learning, and has lead models like CLIP @clip, VLMo @vlmo, and CoCa @coca to reach
state-of-the-art results in image-text retrieval.

We apply this loss on the outputs of all three MLP layers of the shared encoder, as we want to enforce the shared encoder to generate
aligned representations in all layers. The refined contrastive loss is then given by:

$
cal(L)_("CL") &= \
1/3 * (cal(L)_("CL"') &+ cal(L)_("CL"'') + cal(L)_("CL"''')) = \
1/6cal(L)_("CL"')^("i2t") &+ 1/6cal(L)_("CL"')^("t2i") + \
1/6cal(L)_("CL"'')^("i2t") &+ 1/6cal(L)_("CL"'')^("t2i") + \
1/6cal(L)_("CL"''')^("i2t") &+ 1/6cal(L)_("CL"''')^("t2i")
$ <full_contrastive_loss_transformer_shre>

Let's break this down: The prime ($prime$) symbol defines on which outputs
from the shared encoder (@transformer_shre_shared_encoder_ffn_computation and @transformer_shre_shared_encoder_head_computation)
the contrastive loss is applied. Since we have three linear
layers in our shared encoder, and we want to enforce alignment in the whole shared encoder,
we apply the contrastive loss on all three layers, but seperately.
The superscripts $"i2t"$ and $"t2i"$ denote if we apply the contrastive loss
from image to text or from text to image, and should already be known from 
when we introduced vision-language contrast (@vision_language_contrast). To sum up, we apply the contrastive loss on all
three linear layers of the shared encoder, and we weight the loss equally for each layer. The contrastive loss 
itself weights image-to-text and text-to-image equally, which is why each component of the contrastive loss is weighted with $1/6$.
For each contrastive loss $cal(L)_("CL"')$, $cal(L)_("CL"'')$, and $cal(L)_("CL"''')$
we generate the matrix $bold(L)$ from @vision_language_contrast once.

Since we use the contrastive loss from CLIP, we also have to define a temperature for the contrastive loss on each layer. In total, we have
three temperature parameters, one for each linear layer of the shared encoder. As done in CLIP, we optimize them in log space and initialize
them with 0.07 @clip.

The total training objective is then given by:

$
min cal(L)_("KD") + cal(L)_("CL")
$

===== Training <transformer_shre_training>
For the teacher model we select an improved variant of the ResNet-50 @resnet, called ResNet-50-A1 @resnet_50_a1, which has
25.6M parameters but runs in inference mode, so no gradients are computed. The model was trained on ImageNet-1K @imagenet and is available
on HuggingFace#footnote[#link("https://huggingface.co/timm/resnet50.a1_in1k")].

We train the student model on all 3.3M image-text pairs we collected (@vl_dataset_summary) for 7 epochs, using a batch size of 256.
We do not train for longer, as (1) we want to keep the training time manageable, and (2) we use a lot of pretrained components, which
need less training time to converge. As done in prior experiments, we use the AdamW optimizer @adamW and a learning rate of 1e-4.
After every epoch, we validate the model on CLIP's zero-shot image classification approach,
introduced in @clip_zero_shot_section, and select the best model based on the achieved accuracy. The representations for the zero-shot
classification are generated by the shared encoder of the student model, which we define as $bold(h)'''_(v, K)$ and $bold(h)'''_(w, K)$.
The classification is performed on the validation set of ImageNet-1K @imagenet. At this point it is important to note that
the accuracy we report with CLIP zero-shot classification is actually not zero-shot. This is because the teacher model is trained
supervised on ImageNet-1K, and the student model is trained using the teacher's probability distribution over the ImageNet-1K classes.
Our student model therefore learns the ImageNet-1K classes even though we do not train on ImageNet-1K directly.
However, the accuracy we achieve still gives us a good indication of the quality of the student model's representations.

As mentioned before, we tokenize the text using the uncased BERT tokenizer. Again, uncased means that all text is lowercased before tokenization.
Inspired by BEiT-3, we set the maximum text length, the length of the captions, to 64 tokens @beit3, truncate longer captions and pad shorter captions.
This reduces the time required for a forward pass of the text encoder, and the captions of the data we collect are on average not larger than
15 tokens anyway (see @vl_dataset_summary).

For image augmentation, we use the same techniques as in the unimodal image KD experiment (@unimodal_kd_data2vec2_distillation).
However, we make one important distinction in the size of the random crop: As seen in @distil_data2vec2_hyperparameters, the range of the random crop
size is between 0.08 and 1.0 of the original image size. The lower bound is quite small, but because student and teacher receive the same
crop, even if it is very small, the student can still learn the teacher's representation for a small part of the image.
However, this gets problematic with image text pairs. If the crop is very small, then important semantic information of the image might be lost,
which is still present in the text. Therefore, the resulting probability distribution of the teacher for that small crop might not be representative
of the image's high-level content, which is described by the text. This could result in the student predicting a probability distribution that is
correct with respect to the whole image, but not with respect to the small crop. To avoid this, we set the lower bound of the random crop size
to 0.9, which is also the value used by VLMo @vlmo. This ensures that the crop is large enough to capture the high-level content of the image.
A visualization of too small crops is shown in @vl_crop_size_bad, and a visualization of a minimum crop size of 0.9 is shown in @vl_crop_size_good.

All hyperparameters are summarized in @transformer_shre_hyperparams, pytorch pseudocode for the forward pass can be found in @transformer_shre_forward_pseudocode.

===== Results <first_results_transformer_shre>
We report the results of
image-text retrieval on the test sets of COCO @coco and Flickr30k @flickr30k, which are shown in @image_text_retrieval_shre_transformer_first.
Note that we do not report results on unimodal downstream tasks like image classification
or text classification. This is because finetuning is expensive, which is why will refrain from doing so until we reach our final
approach.

While the results on image-text retrieval are significantly worse than the state-of-the-art, we can still observe that we are not far off
from the performance of FLAVA @flava. Considering that FLAVA was developed by a team of researchers from Meta AI, and that this is our first
iteration of a multimodal model, the results are promising. From CLIP, VLMo, and BEiT-3 we are still far off, but this can at least partly
be attributed to the fact that those model are significantly larger than ours, and that they have been trained on much more data. The
latter is shown in @models_data_size_comparison.
The increased performance on Flickr30K compared to COCO can be attributed to the fact that the Flickr30K dataset is smaller. For a given
image, there are 5 correct captions in only 5k possible captions (25k for COCO),
and for a given caption there are only 1k possible images (5k for COCO).
We use the representations $bold(h)'''_(v, K)$ and $bold(h)'''_(w, K)$ for retrieval.

We also proof our statement of @multimodal_models that using two unrelated unimodal models, one image and one text model, for image-text retrieval
fails, as the representations produced by the two models are not aligned. We do this by using the pretrained image and text variant from
Data2Vec2 @data2vec2 as the image and text encoder, respectively. Each image is encoded by the image encoder, and each caption is encoded
by the text encoder, after which the representation of the $mono(["I_CLS"])$ and $mono(["T_CLS"])$ token is extracted and used for retrieval
according to our formulation from @vision_language_contrast. This approach does not reach a score of more than 0.2% on any metric,
and is therefore inappropriate for image-text retrieval. Consequently, we can conclude that training a multimodal model is essential
for the alignment of modalities.

Transformer SHRe reaches 40.26% accuracy on the (zero-shot) ImageNet classification task, which is significantly worse that the 76.2% reached by
CLIP @clip, with CLIP being an actual zero-shot application (see previous section).


#show table: set text(8pt)
#figure(
  table(
  columns: (25%, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: none,
    table.hline(),
    table.header(
      table.cell(rowspan: 3, colspan: 1, align:horizon, [*Model*]),
      table.cell(colspan: 6, [*MSCOCO (5K test set)*]),
      table.cell(colspan: 6, [*Flickr30K (1K test set)*]),
      table.cell(colspan: 3, [Image $arrow.r$ Text]),
      table.cell(colspan: 3, [Text $arrow.r$ Image]),
      table.vline(stroke: .4pt),
      table.cell(colspan: 3, [Image $arrow.r$ Text]),
      table.cell(colspan: 3, [Text $arrow.r$ Image]),
      table.hline(start: 1, end: 4, stroke: .2pt),
      table.hline(start: 4, end: 7, stroke: .2pt),
      table.hline(start: 7, end: 10, stroke: .2pt),
      table.hline(start: 10, end: 13, stroke: .2pt),
      [R@1], [R@5], [R@10], [R@1], [R@5], [R@10], [R@1], [R@5], [R@10], [R@1], [R@5], [R@10]
    ),
    table.hline(stroke: .4pt),
    [Data2Vec2 @data2vec2], [0.02], [0.08], [0.19], [0.01], [0.10], [0.19], [0.02], [0.12], [0.18], [0.02], [0.06], [0.12],
    table.hline(stroke: .1pt),
    [FLAVA @flava], [42.74], [76.76], [-], [38.38], [67.47], [-], [67.7], [94.0], [-], [65.22], [89.38], [-],
    [CLIP @clip], [58.4], [81.5], [88.1], [37.8], [62.4], [72.2], [88.0],[98.7], [99.4], [68.7], [90.6], [95.2],
    [VLMo @vlmo], [83.1], [96.0], [98.2], [65.2], [86.5], [92.2], [96.8], [100], [100], [88.1], [98.4], [99.3],
    [BEiT-3 @beit3], [*84.8*], [*96.5*],[*98.3*], [*67.2*], [*87.7*], [*92.8*], [*98*], [*100*], [*100*], [*90.3*], [*98.7*], [*99.5*],
    table.hline(stroke: .3pt),
    [$"SHRe"_T$ (ours)], [37.06], [67.74], [79.7], [25.3], [54.73], [68.19], [49.9], [78.5], [88.5], [37.04], [67.34], [77.38],
    table.hline(),
  ),
  caption: [
    Results of image-text retrieval on the COCO and Flickr30K test sets for Transformer SHRe.
    The results are compared to FLAVA @flava, CLIP @clip, VLMo @vlmo, and BEiT-3 @beit3.
  ],
)<image_text_retrieval_shre_transformer_first>
#show table: set text(12pt)

Since we are using the same approach as SHRe, we also provide a comparison of the average median rank on the COCO test set.
How SHRe reports the average median rank is described in @shre_section. Unfortunately, the authors 
do not provide the exact pairs used to calculate the average median rank. We therefore opt for an approach that should closely
resemble that of SHRe: We select all 5k images from the COCO test set, and split them into 5 distinct sets of 1k images each.
For each set we do the following: For each image one caption, out of the 5 captions available, is selected.
We then have 5 splits of 1k image-caption pairs each. In each split, for one candidate there is only one correct target and 999 incorrect targets.
We then perform retrieval on each split, and calculate the median rank of the correct target. The average median rank is then the average
of the median ranks over all 5 splits. This procedure itself is repeated 5 times, so that each image is paired with each of its
5 correct captions exactly once. The result is 5 average median ranks, which are then averaged to
get the metric reported in @transformer_shre_amr_results.
Our approach significantly improves the baseline of SHRe, and the results indicate that the correct pair for a query is
in most cases either the first or second retrieved item, though the results do not account for outliers, since the median rank is used.
The reason for the almost perfect results, considering the minimum possible value is 1, can be attributed to the advances in
deep learning research since SHRe was published, which was in 2017 @shre. Some of the advances include the Transformer architecture,
and the use of contrastive learning. Furthermore, we can assume that the quality of the teacher model, which is a ResNet-50-A1 @resnet_50_a1,
is also higher than the teacher model used in SHRe, which they do not specify explicitly, but they mention AlexNet as an example teacher model.
Lastly, our approach is a vision-language model, while SHRe is a vision-language-audio model, which might make the task of alignment more difficult.

#figure(
  table(
  columns: 5,
    stroke: none,
    table.hline(),
    table.header(
      table.cell(rowspan: 3, colspan: 1, align:horizon, [*Model*]),
      table.cell(colspan: 4, [*MSCOCO*]),
      table.cell(colspan: 2, [Image $arrow.r$ Text]),
      table.cell(colspan: 2, [Text $arrow.r$ Image]),
    ),
    table.hline(stroke: .4pt),
    [Random], table.cell(colspan: 2, [500]), table.cell(colspan: 2, [500]),
    [SHRe @shre], table.cell(colspan: 2, [5.8]), table.cell(colspan: 2, [6.0]),
    [$"SHRe"_T$ (ours)], table.cell(colspan: 2, [*1.5*]), table.cell(colspan: 2, [*2.0*]),
    table.hline(),
  ),
  caption: [Comparison of the average median rank over 5 1k image-caption splits on the COCO test set.
  Our approach outperforms SHRe by a large margin. The best possible value is 1.],
)<transformer_shre_amr_results>

As mentioned in the introduction of SHRe @shre, the idea of not only predicting the probability distribution over the ImageNet-1K classes
for a given image, but also for its caption, works because the ImageNet-1K classes can also be used to describe the content of a caption.
The main reason for this is that the ImageNet-1K classes describe real-world objects, which are independent of the image modality, and
can therefore also be used to describe the content of a text. A visualization of that is shown on image-text pairs from COCO,
which is part of the data we use for training, in @shre_coco_prob_dist.

==== Larger Batch Sizes with DDP <larger_batch_sizes_ddp>
As mentioned in the introduction of contrastive learning (@vision_language_contrast), a large batch size is crucial
for the success of contrastive learning. Larger batch sizes allow for more negative samples, which makes the task of
finding the correct pair among those negatives more difficult.  Unfortunately, we are not able to exceed a batch size of 256,
as we run out of memory with our current GPU setup (1 $times$ NVIDIA RTX 4090).

To overcome this limitation, we utilize Distributed Data Parallel (DDP) @pytorch_ddp, which allows us to train our model on multiple GPUs.
Each GPU has its own replica (copy) of the model, and for a single forward pass each GPU processes a different batch of the data.
The forward pass and backward pass are then computed on each GPU seperately, based on the mini-batch of data it received. The resulting
gradients for each replica are then aggregated across all GPUs/replicas, and each replica updates its weights based on the aggregated gradients.
We are therefore able to increase the batch size by the number of GPUs we use, and update the weights of the model based on gradients
that have been computed on a larger batch size than a single model received in a forward pass @pytorch_ddp. An illustration of DDP with 2 GPUs
is shown in @ddp_illustration.

A side effect of DDP is that after the forward pass of each replica, there now exists not just a single batch of image-text representations,
but as many batches as there are replicas/GPUs. Those representations can, similar to the gradient accumulation, be communicated across
all replicas. If we use 2 GPUs with a batch size of 256, then all representations on the first GPU are communicated to the second GPU, and
vice versa. Consequently, we now have 512 image-text pairs on which the contrastive loss is computed, which is equivalent 
to using a batch size of 512 on a single GPU.

This not only increases the effectiveness of the contrastive loss, but also reduces the time required to train the model. 
This is because when using DDP,
the whole dataset is split across all GPUs (see @ddp_illustration),
and each GPU always only processes its own part of the dataset. Since the forward pass on each GPU
is done in parallel, and the number of steps per epoch is reduced by a factor equal to the number of GPUs used, the training time is reduced.
Even though the time required for training the model will not exactly be reduced by the factor of the number of GPUs used, as the distributed
communication between the GPUs introduces overhead, it is still significantly faster than training on a single GPU. From a cost perspective,
multiple GPUs are obviously more expensive than a single GPU, but the time saved by using DDP outweighs the increased cost to some extent.
This is especially important considering that we can traing more models in a shorter amount of time, which allows us to iterate faster.

While it is tempting, also from a technical perspective, to use as many GPUs as possible, we refrain from doing so. This is because
we do not want to make the success of our approach too dependent on the hardware we use. After all, the goal is to develop an approach
that is feasible even for researchers with limited resources. We therefore limit the number of GPUs to 2, effectively increasing the batch size
to 512. The architecture, loss, and training procedure remain the same as described in the previous sections.

As shown in @ddp_result_comparison, introducing DDP with a second GPU lead to an absolute gain of more than 4
percentage points on the COCO and Flickr30K retrieval tasks,
while almost reaching perfect performance on the average median rank task of SHRe @shre.
This underlines the importance of a large batch size for contrastive learning.
However, we observe no improvement on the (zero-shot) ImageNet classification task.
We hypothesize that the classification task is more dependent on the quality of the text-based class prototypes,
which are generated from predefined textual descriptions of the class labels (see @clip_zero_shot_section).
The quality of these prototypes, which are fixed representations of each class, is not directly influenced by the batch size used during training.
In contrast, retrieval tasks rely on contrastive learning, where having a higher diversity of negative samples during training
helps the model better distinguish between relevant and irrelevant image-text pairs.
As a result, an increase in batch size translates more directly into better retrieval performance,
while its effect on zero-shot classification is less pronounced.

Further, the average median rank (AMR) for image-to-text and text-to-image retrieval on COCO
decreases to almost perfect scores, seen in @ddp_result_amr_comparison.

The training time, displayed in @ddp_result_time_comparison behaves as expected: The wall clock time per batch (the time
needed for one full training step, including weight updates) is higher when using DDP,
but the total training duration is significantly lower. Again, the latter is due to the fact that the
number of steps per epoch is reduced by a factor equal to the number of GPUs used, which reduces the total training time.

#show table: set text(8pt)
#figure(
  table(
  columns: 4,
  stroke: none,
  table.hline(),
  table.header(
    table.cell(rowspan: 2, colspan: 1, align:horizon, [DDP]),
    table.cell(rowspan: 2, colspan: 1, align:horizon, [ImageNet-1K]),
    table.cell(colspan: 2, align:horizon, [Retrieval]),
    [MSCOCO],
    [Flickr30K],
  ),
  table.hline(stroke: .6pt),
  [$times$], [*40.26*], [55.45], [66.44],
  [$checkmark$], [40.0], [*59.59*], [*70.78*],
  table.hline(),
),
    caption: [
    DDP improves the performance of the model on the COCO and Flickr30K retrieval tasks, but not on (zero-shot) ImageNet classification.]
) <ddp_result_comparison>

#figure(
  table(
  columns: 5,
    stroke: none,
    table.hline(),
    table.header(
      table.cell(rowspan: 3, colspan: 1, align:horizon, [*Model*]),
      table.cell(colspan: 4, [*MSCOCO*]),
      table.cell(colspan: 2, [Image $arrow.r$ Text]),
      table.cell(colspan: 2, [Text $arrow.r$ Image]),
    ),
    table.hline(stroke: .4pt),
    [SHRe @shre], table.cell(colspan: 2, [5.8]), table.cell(colspan: 2, [6.0]),
    [$"SHRe"_T$ (ours)], table.cell(colspan: 2, [1.5]), table.cell(colspan: 2, [2.0]),
    [$"SHRe"_T$ + DDP (ours)], table.cell(colspan: 2, [*1.0*]), table.cell(colspan: 2, [*1.04*]),
    table.hline(),
  ),
  caption: [
    Increasing the negative samples by using DDP improves the average median rank on the COCO test set.
  ],
)<ddp_result_amr_comparison>

#show table: set text(8pt)
#figure(
  table(
    columns: 3,
    stroke: none,
    table.hline(),
    table.header(
      [*Approach*],
      [*Wall Clock Time per Batch (ms)*],
      [*Training duration (h)*],
      table.hline(stroke: .6pt),
    ),
    [Default], [*331*], [8.2],
    [DDP], [387], [*4.8*],
    table.hline(),
  ),
  caption: [
    Comparison of the wall clock time per batch and the total training duration for the default and DDP approach.
    The training duration is calculated for 7 epochs on the 3.3M image-text pairs we collected. While the wall clock time per batch
    is higher for the DDP approach, the total training duration is significantly lower.
  ],
)<ddp_result_time_comparison>
#show table: set text(11pt)

==== Shared Transformer Encoder
We prepend the name of our approach with "Transformer" to indicate that we use a Transformer architecture for both the image and text encoder.
However, this does not hold for the shared encoder, which is, like the original approach, a 3-layer MLP @shre. We now experiment with also
replacing the shared encoder with a Transformer, leading to a fully Transformer-based model. We motivate this decision with the architecture
of VLMo and BEiT-3, which both use Transformer layers towards the end of the model @vlmo @beit3. 

While our shared encoder would then mimic the architecture of VLMo and BEiT-3 in the sense that is uses Transformer layers that
process both image and text representations, there is still a significant difference to those models: The upper two Transformer layers,
which are the shared ones,
of VLMo and BEiT-3 do not process image and text representations separately, but receive a shared representation of an image-text pair.
This is created by simply concatenating the image and text representations, and passing them through the Transformer layers @vlmo @beit3. The input to
such a layer $l$ is then given by:

$
bold(H)_("vw", l-1) = [bold(H)_(w, l-1); bold(H)_(v, l-1)]
$

This allows the Self-Attention mechanism to attend between individual image and text tokens, leading to a more fine-grained alignment.
In contrast, our shared encoder will, just like the shared 3-layer MLP before, process image and text representations separately.
The input will therefore either be a single
image representation $bold(H)_(v, l-1)$, returned by the image encoder, or a single text representation $bold(H)_(w, l-1)$,
returned by the text encoder.

We hypothesize that our approach might be more challenging to learn, compared to VLMo and BEiT-3's concatenated strategy,
due to the following reasons:

- In VLMo and BEiT-3, the input is always a combined image-text representation, ensuring consistency for
  the Self-Attention mechanism, which can then concentrate solely on fine-grained alignment.
- In our case, the Self-Attention mechanism receives either an image sequence or a text sequence independently,
  requiring it to not only perform attention but also infer the modality of the input. Since image and text inputs
  are inherently different, this adds a layer of complexity.

Nevertheless, VLMo demonstrates that this modality-specific complexity may not be an issue.
In VLMo, masked language modeling is performed using a Transformer whose Self-Attention weights are
frozen and from a pretrained image model, yet the model still effectively handles text-only pretraining @vlmo.

Given that VLMo can successfully leverage frozen image Self-Attention weights for text tasks,
we believe that our approach, where the Self-Attention weights are explicitly trained to work with both
image and text inputs, should also be effective, if not more so.

The change in the architecture does not imply a change in the training procedure, loss, or hyperparameters.
In fact, the only thing we actually add is one Multi-Head Self-Attention layer to the shared encoder (plus the two residual connections).
Recall that we implemented the 3-layer MLP as a 2-layer MLP network
as used in Transformer layers, and added an additional LayerNorm and linear layer on top of it
(see @transformer_shre_architecture).
Replacing this with a single Transformer layer means we still have the 2-layer MLP network, but now also
add a Multi-Head Self-Attention layer before it. Since we are still predicting a probability distribution over the ImageNet classes
(the one returned by the teacher),
which can be seen as a type of classification task,
we still need a classification head on top of the Transformer layer to output logits for the 1000 classes. Fittingly, this is
exactly the task of the third MLP layer we had in the shared 3-layer MLP. The number of neurons for each linear layer
also remains the same: The first linear layer expands the hidden dimension of the Transformer from 768 to 3072, and the second
linear layer reduces it back to 768. The third linear layer then expands it to 1000, the number of classes in ImageNet.
In @comparison_shared_mlp_transformer, where we illustrate our previous explanation,
we indicate the difference in dimensionality between the linear layers by a different breadth.

#figure(
  image(
  width: 75%,
  "../figures/comparison_shared_mlp_transformer.png"),
  caption: [
    Switching from our implementation of the shared encoder as a 3-layer MLP to a Transformer layer only corresponds to adding a
    Multi-Head Self-Attention layer before the linear layers. The dimensionality of the linear layers remains the same.
    On the right, layer normalization, activations, and residual connections are omitted for simplicity.
],
) <comparison_shared_mlp_transformer>

Since the shared encoder is now can actual Transformer layer plus the classification head, we redefine our previous forward pass
of the shared encoder from @transformer_shre_shared_encoder_ffn_computation and @transformer_shre_shared_encoder_head_computation to the
following (also on the example of the text modality):

$
bold(H)^("mha")_(w,K) &= op("MHA")(op("LN")(bold(H)_(w,L_s))) + bold(H)_(w,L_s)
$ <shre_shared_transformer_layer_mha_eq>

$
bold(H)'_(w,K) &= op("LN")(bold(H)^("mha")_(w,K))bold(W)_1 + bold(b)_1 \
bold(H)''_(w,K) &= op("LN")(op("GELU")(bold(H)'_(w,K)))bold(W)_2 + bold(b)_2 \
bold(H)^("ln")_(w,K) &= op("LN")(bold(H)''_(w,K) + bold(H)^("mha")_(w,K))
$ <shre_shared_transformer_layer_ffn_eq>

$
bold(h)'''_(w, K, mono(["T_CLS"])) &= bold(h)^("ln")_(w,K, mono(["T_CLS"]))bold(W)_3 + bold(b)_3 \
$ <shre_shared_transformer_layer_head_eq>

It holds that $K=L_s+1=7$, since the shared encoder is one additional Transformer layer.
The subscripts for the weights $bold(W)$ and biases $bold(b)$ denote to which linear layer from @comparison_shared_mlp_transformer (right)
they belong. Correspondingly, the output of the equation where a pair of ($bold(W)_i$, $bold(b)_i$) is used, is
also the output of the linear layer $i$.
@shre_shared_transformer_layer_mha_eq is the usual operation of the Multi-Head Self-Attention layer known from the ViT architecture,
and is therefore also shown in @vit_full_forward_eq of the section on Vision Transformers (@vision_transformer).

@shre_shared_transformer_layer_ffn_eq shows the operation of the Feed-Forward Network (FFN) layer of the Transformer.
It is the same operation as defined in @transformer_shre_shared_encoder_ffn_computation, but applied pointwise on a sequence.
We divide the operations of the FFN into three formulas, as we want to explicitly show that we extract
the intermediate representation $bold(H)'_(w,K)$ and $bold(H)'_(v,K)$,
and final representation $bold(H)''_(w,K)$ and $bold(H)''_(v,K)$ from the FFN.
Both the intermediate and final representations we extract are the raw outputs from linear layer \#1 and linear layer \#2,
without any activation function or normalization applied.
From those representations, which are still sequences,
we take the representation of the $mono(["T_CLS"])$ and $mono(["I_CLS"])$ token for text and image, respectively.
Concretely, they are: $bold(h)'_(w, K, mono(["T_CLS"]))$ and $bold(h)'_(v, K, mono(["I_CLS"]))$, and
$bold(h)''_(w, K, mono(["T_CLS"]))$ and $bold(h)''_(v, K, mono(["I_CLS"]))$. Those are then used for the contrastive loss.

The operation in @shre_shared_transformer_layer_head_eq now resembles that of an actual
classification head from the vision Transformer architecture, and is only applied on the $mono(["I_CLS"])$ or $mono(["T_CLS"])$ token.
An important distinction we make here, compared to the original shared 3-layer MLP, is that we do not apply the contrastive loss on the output
of the classification head $bold(h)'''_(w, K, mono(["T_CLS"]))$
and $bold(h)'''_(v, K, mono(["I_CLS"]))$, and we explain the reasoning for this in the following.

The task of the classification head is to leverage the knowledge learned from the teacher to provide the student with guidance.
This guidance can be described as the fact that objects, or rather the ImageNet-1K classes that describe them, can be found in both
the image and its caption (see @shre_coco_prob_dist). The student will therefore learn that an image and its caption
both describe the same real-world object/concepts.
Based on this intuition, the classification head is not meant to output representations of the image or text, and should therefore not be used
for retrieval tasks.

For all retrieval tasks, including CLIP-like zero-shot classification, we consequently use the representations
$bold(h)''_(v, K, mono(["I_CLS"]))$ and $bold(h)''_(w, K, mono(["T_CLS"]))$ as final representations for image and text, respectively.
The classification head can therefore be discared after training, as it is not needed for retrieval
or any other downstream task. It is only used during training to provide the student with guidance from the teacher.
An overview which tokens are used in which part of the training objective is shown in @transformer_shre_loss_tokens.

The full contrastive loss is now:
$
cal(L)_("CL") &= \
1/2 * (cal(L)_("CL"') &+ cal(L)_("CL"'')) = \
1/4cal(L)_("CL"')^("i2t") &+ 1/4cal(L)_("CL"')^("t2i") + \
1/4cal(L)_("CL"'')^("i2t") &+ 1/4cal(L)_("CL"'')^("t2i")
$ <contrastive_loss_sx3hre>

#figure(
  table(
    columns: 4,
    stroke: none,
    gutter: (0pt, 0pt, 2pt, 0pt, 2pt, 0pt, 2pt, 0pt, 0pt),
    table.hline(),
    table.header(
      [*Loss*],
      [*Modality*],
      [*SHRe*],
      [*Transformer SHRe*],
      table.hline(stroke: .6pt),
    ),
    table.cell(rowspan: 2, align:horizon, [$cal(L)_("KD")$]), [Image], [$bold(h)'''_(v, K)$], [$bold(h)'''_(v, K, mono(["I_CLS"]))$],
      [Text], [$bold(h)'''_(w, K)$], [$bold(h)'''_(w, K, mono(["T_CLS"]))$],
    table.hline(stroke: .5pt),
    table.cell(rowspan: 2, align:horizon, [$cal(L)_("CL"')$]), [Image], [$bold(h)'_(v, K)$], [$bold(h)'_(v, K, mono(["I_CLS"]))$],
      [Text], [$bold(h)'_(w, K)$], [$bold(h)'_(w, K, mono(["T_CLS"]))$],
    table.hline(stroke: .1pt),
    table.cell(rowspan: 2, align:horizon, [$cal(L)_("CL"'')$]), [Image], [$bold(h)''_(v, K)$], [$bold(h)''_(v, K, mono(["I_CLS"]))$],
      [Text], [$bold(h)''_(w, K)$], [$bold(h)''_(w, K, mono(["T_CLS"]))$],
    table.hline(stroke: .1pt),
    table.cell(rowspan: 2, align:horizon, [$cal(L)_("CL"''')$]), [Image], [$bold(h)'''_(v, K)$], [-],
      [Text], [$bold(h)'''_(w, K)$], [-],
    table.hline(),
  ),
  caption: [
    A comparison between the tokens used in the loss functions for the approach of SHRe @shre with a shared 3-layer MLP
    and out Transformer SHRe.
    For Transformer SHRe we do not use the contrastive loss $cal(L)_("CL"''')$.
  ],
) <transformer_shre_loss_tokens>

==== Results

The influence on retrieval, when adding a shared Transformer layer and changing the representations used for retrieval,
can be seen in @image_text_retrieval_shre_overview. This change not only outperforms
our first two experiments in all metrics, but also FLAVA @flava in R@1 text retrieval on COCO. For other metrics on COCO, we are also
surprisingly close to FLAVA. On Flickr30K, we still lack behind FLAVA, but the gap is significantly smaller than before.

We are also able to increase the performance on CLIP-like ImageNet-1K classification from 40.26% to 44.57%, and reach a perfect average median rank
for both text and image retrieval on the COCO test set, which is 1.0. As a reference: The previous scores were 1.0 for text retrieval,
and 1.04 for image retrieval (see @ddp_result_comparison). While this is indeed the perfect score for the average median rank,
it is important to note that we use the _median_ rank, and a value of 1.0 therefore means that in at least half of all retrievals
(so $gt.eq 500$) the correct pair
is the first retrieved item. This does not account for outliers, and is only applied on a small subset of 1k image-caption pairs.
Nevertheless, this is the benchmark provided by SHRe @shre, so a perfect score is still worth mentioning. Unless there will be a decrease in performance
on the average median rank, we will refrain from reporting it in the future, as it is not a very informative metric and we have already reached the best
possible score.
A visualization of the retrieved samples for captions (image retrieval) and images (text retrieval) is shown in @shre_transformer_only_ir_1k
and @shre_transformer_only_tr_1k, respectively.


#show table: set text(8pt)
#figure(
  table(
  columns: (25%, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: none,
    table.hline(),
    table.header(
      table.cell(rowspan: 3, colspan: 1, align:horizon, [*Model*]),
      table.cell(colspan: 6, [*MSCOCO (5K test set)*]),
      table.cell(colspan: 6, [*Flickr30K (1K test set)*]),
      table.cell(colspan: 3, [Image $arrow.r$ Text]),
      table.cell(colspan: 3, [Text $arrow.r$ Image]),
      table.vline(stroke: .4pt),
      table.cell(colspan: 3, [Image $arrow.r$ Text]),
      table.cell(colspan: 3, [Text $arrow.r$ Image]),
      table.hline(start: 1, end: 4, stroke: .2pt),
      table.hline(start: 4, end: 7, stroke: .2pt),
      table.hline(start: 7, end: 10, stroke: .2pt),
      table.hline(start: 10, end: 13, stroke: .2pt),
      [R@1], [R@5], [R@10], [R@1], [R@5], [R@10], [R@1], [R@5], [R@10], [R@1], [R@5], [R@10]
    ),
    table.hline(stroke: .4pt),
    [FLAVA @flava], [42.74], [*76.76*], [-], [*38.38*], [*67.47*], [-], [*67.7*], [*94.0*], [-], [*65.22*], [*89.38*], [-],
    table.hline(stroke: .3pt),
    [Baseline], [37.06], [67.74], [79.7], [25.3], [54.73], [68.19], [49.9], [78.5], [88.5], [37.04], [67.34], [77.38],
    [  \+ DDP ($arrow.t$ 4.24)], [41.88], [71.86], [83.06], [29.4], [59.3], [72.06], [56.1], [83.5], [90.7], [41.80], [71.36], [81.14],
    [  \+ Shared $T$ ($arrow.t$ 5.68)], [*44.6*], [75.3], [85.64], [31.69], [62.1], [74.48], [58.8], [85.6], [92.4], [43.92], [74.06], [82.8],
    table.hline(),
  ),
  caption: [
    The overview of retrieval results on the COCO and Flickr30K test sets show significant improvements when using DDP and a shared Transformer encoder.
    We achieve an absolute gain of almost 7% on the COCO test set, and more than 6% on the Flickr30K test set, compared to our
    first approach, which we denote as "Baseline". We add FLAVA @flava as a reference.
  ],
)<image_text_retrieval_shre_overview>
#show table: set text(12pt)