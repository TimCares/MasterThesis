== Multimodal Knowledge Distillation <multimodal_knowledge_distillation>
=== Transformer SHRe <transformer_shre>
Before we develop an end-to-end self-supervised approach to multimodal knowledge distillation, we first follow the approach
of SHRe @shre and develop a multimodal knowledge distillation with a probability distribution over the classes as the prediction
target. This allows us to closely observe the impact of switching from a supervised to a self-supervised teacher model on the
student model's performance. Moreover, it allows us to gradually increase the complexity of our approach and build on our
previous advancements.

==== Method
===== Architecture <transformer_shre_architecture>
What makes our approach different from SHRe is that we use a langauge Transformer as the text encoder, and
a vision Transformer as the image encoder, bringing us closer to a unified architecture.
In contrast, SHRe uses 2D convolutions and 1D convolutions for the image and text, respectively. For now, the shared encoder remains
a 3-layer MLP, as in SHRe @shre. Since the output of the image and text encoder is a sequence of features, but we use a normal MLP as the shared encoder,
we have to reduce the sequence of features to a single feature vector. This is done by taking the representation of the $mono(["I_CLS"])$
and $mono(["T_CLS"])$
token from the image and text encoder, respectively, and aligns with our first implementation of a multimodal model shown in @multimodal_model.
This represenation, namely $bold(h)_(v, L_s, mono(["I_CLS"]))$
and $bold(h)_(w, L_s, mono(["T_CLS"]))$, is then passed to the shared encoder, which produces the final multimodal representations
$bold(h)_(v, K)$ and $bold(h)_(w, K)$. Here, $L_s$ denotes the number of Transformer layers in the image and text encoder, respectively, which is
defined as $L_s=6$. $K$ denotes the number of the output layer, which is defined as $K=9$. We set $K=9$ because we have a 3-layer MLP and we
count each MLP layer as a distinct layer. Since the MLP is stacked on top of the image and text encoder the last MLP layer, being the output layer,
is the 9th layer. Further, we remove the token indicator $mono(["I_CLS"])$ and $mono(["T_CLS"])$ from the notation, as the MLP only receives
those representations as input. There are no more time steps to distinguish between.

The keep the model size manageable, we will resort the the same approach as in our unimodal experiment, and use 6 Transformer layers
for the image and text encoder, respectively. This also gives us the opportunity
to directly compare the performance of the image and text encoder from our multimodal model to that of the unimodal models (@unimodal_knowledge_distillation). This can be done by evaluating e.g. the image encoder of the multimodal model on image classification tasks, and
then comparing its performance to the results observed for the unimodal image KD model of @unimodal_kd_vision
on the same tasks.
A performance similar to that of the strictly unimodal models would indicate that multimodal pretraining
yields strong unimodal encoders as a byproduct. As argued by the authors of FLAVA @flava, the aforementioned is in fact even a
requirement for multimodal models. This can be attributed to the fact that an alignment in
the shared encoder is only possible if the unimodal encoders generate features from which the shared encoder
can extract modality-invariant information, like the semantic content of an image or text. If the unimodal encoders are not able to extract
high-level features, then neither will the shared encoder be able to extract modality-invariant information,
nor will the features be useful in the corresponding unimodal downstream tasks, like for example image classification
using the image encoder of the multimodal model.

As done in our unimodal experiments, we initialize the image and text encoder using the embeddings, positional encodings,
and the first 6 Transformer layers from Data2Vec2 @data2vec2 and BERT @bert, respectively. The shared encoder will be initialized
randomly. For an illustration of the architecture
we refer to the same depiction used for SHRe, shown in @shre_fig. The only difference is that we only use image and text encoders,
which are now Transformers instead of CNNs.

The shared encoder, which is a 3-layer MLP, is implemented by using the 2-layer MLP module as implemented in a Transformer layer,
and adding an additional LayerNorm and MLP layer on top of it. Given the output from the text encoder, the forward pass of the shared encoder
is then given in the following, and changes for the image encoder accordingly:
$
bold(h)_(w, K-2) &= bold(h)_(w, L_s, mono(["T_CLS"]))bold(W)_1 + bold(b)_1 \
bold(h)_(w, K-1) &= op("LN")(op("GELU")(bold(h)_(w, K-2)))bold(W)_2 + bold(b)_2 \
bold(h)_(w, K) &= op("LN")(bold(h)_(w, K-1))bold(W)_3 + bold(b)_3 \
$

The computation of $bold(h)_(v, K-2)$ and $bold(h)_(v, K-1)$ is analogous to the definition of the MLP layers in a Transformer layer, defined in @transformer_ffn_eq. We choose a different notation here to be more precise when defining the loss in the next section.

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
The rest of parameters the text encoder has more is related to BERT's specific implementation of positional encodings.

While 115M parameters can be considered as quite large, considering we strive to build smaller models, 
it is still significantly smaller than the vision-language models we compare to.
For example, VLMo @vlmo has 175M
#footnote[#link("https://github.com/microsoft/unilm/tree/master/vlmo")], CLIP @clip has more than 400M
#footnote[#link("https://huggingface.co/openai/clip-vit-large-patch14")], and BEiT-3 @beit3 has 1.9B parameters @beit3.  

===== Training Objective

Since we start with using a supervised teacher, the loss for knowledge distillation will remain KL-Divergence.
As the application of the KL-Divergence is two-fold, once for the prediction based on the image and once for the prediction based on the caption,
we provide a refined version of the loss function. As a preliminary step, we define the softmax normalization of a vector $bold(u)$ as follows:

$
pi(bold(u)) = bold(z) &= [z_1, z_2, dots, z_n] in RR^n \
z_i &= exp(u_i) / (sum_(j=1)^n exp(u_j))
$

This allows us to generate a probability distribution over the classes for the logits generated by the teacher for the image, and for the
logits generated by the student for image and caption. The loss for knowledge distillation is then given by:

$
cal(L)_("KD") &= \ 
1/2 * cal(L)_("KD")^v + 1/2 * cal(L)_("KD")^w &= \
1/2 * D_("KL")(pi(bold(p)), pi(bold(h)_(v, K))) &+ 1/2 * D_("KL")(pi(bold(p)), pi(bold(h)_(w, K)))
$

$bold(p)$ denotes the logits generated by the teacher for the image, $bold(h)_(v, K)$ the logits generated by the student for the image, and
$bold(h)_(w, K)$ the logits generated by the student for the caption. All are in $RR^1000$ for the 1000 classes of ImageNet.

We decide to replace
the ranking loss of SHRe with the contrastive loss introduced by CLIP @clip, and
explained in @vision_language_contrast. We justify this decision with the fact that vision-language contrast has become the
de-facto standard for multimodal self-supervised learning, and has lead models like CLIP @clip, VLMo @vlmo, and CoCa @coca to reach
state-of-the-art results in image-text retrieval.

We apply this loss on the outputs of all three MLP layers of the shared encoder, as we want to enforce the shared encoder to generate
aligned representations in all layers. The refined contrastive loss is then given by:

$
cal(L)_("CL") &= \
1/3 * (cal(L)_("CL")^(K-2) &+ cal(L)_("CL")^(K-1) + cal(L)_("CL")^K) = \
1/6cal(L)_"CL"^(K-2, "i2t") &+ 1/6cal(L)_"CL"^(K-2, "t2i") + \
\ 1/6cal(L)_"CL"^(K-1, "i2t") &+ 1/6cal(L)_"CL"^(K-1, "t2i") + \
1/6cal(L)_"CL"^(K, "i2t") &+ 1/6cal(L)_"CL"^(K, "t2i")
$

Let's break this down: The superscript $K-2$, $K-1$, and $K$ denote on which layer the contrastive loss is applied. Since we have three layers
MLP layers in our shared encoder, and we want to enforce alignment in all layers, we apply the contrastive loss on all three layers. As introduced
in the previous section (@transformer_shre_architecture), $K$ denotes the number of layers that are stacked on top of each other. Since we have
6 layers for each modality-specific encoder, and 3 layers for the shared encoder, $K=9$. Therefore, we apply the contrastive loss on layers 7-9, which
conviniently are the the three MLP layers forming the shared encoder. The second superscripts $"i2t"$ and $"t2i"$ denote if we apply the contrastive loss
from image to text or from text to image, and should already be known from @vision_language_contrast. To sum up, we apply the contrastive loss on all
three MLP layers of the shared encoder, and we weight the loss equally for each layer. The contrastive loss also 
itself weights image-to-text and text-to-image equally, which is why each component of the contrastive loss is weighted with $1/6$.
For each $cal(L)_("CL")^Q$, with $Q in [K-2, K-1, K]$, we generate the matrix $bold(L)$ from @vision_language_contrast once, but
since we to not directly make use of the $mono(["I_CLS"])$ and $mono(["T_CLS"])$ token, we have to redefine the batched representations
$bold(I)$ and $bold(T)$ as follows:

$
bold(I)_Q = [bold(h)'_((v, Q),1), bold(h)'_((v, Q),2), ..., bold(h)'_((v, Q),B)] in RR^(B times D_Q)
$

$
bold(T)_Q = [bold(h)'_((w, Q),1), bold(h)'_((w, Q),2), ..., bold(h)'_((w, Q),B)] in RR^(B times D_Q)
$

Here, $D_Q$ denotes the dimensionality of the representations $bold(h)_(v, Q)$ and $bold(h)_(w, Q)$ from MLP layer $Q$, which is essentially
the number of neuros in the MLP layer. $B$ denotes the batch size, and we refer to @vision_language_contrast for the definition of $bold(h)'$.
It follows for the matrix $bold(L)$ of image-to-text and text-to-image similarities:

$
bold(L)_Q = exp(t_Q) * bold(I)_Q bold(T)_Q^T, bold(L)_Q in RR^(B times B)
$

Since we use the contrastive loss from CLIP, we also have to define a temperature for the contrastive loss of each layer. In total, we have
three temperature parameters, one for each MLP layer of the shared encoder. As done in CLIP, we optimize them in log space and initialize
them to 0.07 @clip.
The training objective is then given by:

$
min cal(L)_("KD") + cal(L)_("CL")
$

===== Training
For the teacher model we select an improved variant of the ResNet-50 @resnet, called ResNet-50-A1 @resnet_50_a1, which has
25.6M parameters but runs in inference mode, so no gradients are computed. The model was trained on ImageNet-1K @imagenet and is available
on HuggingFace#footnote[#link("https://huggingface.co/timm/resnet50.a1_in1k")].

We train the student model on all 3.3M image-text pairs we collected (@vl_dataset_summary) for 7 epochs, using a batch size of 256.
We do not train for longer, as (1) we want to keep the training time manageable, and (2) we use a lot of pretrained components, which
need less training time to converge. As done in prior experiments, we use the AdamW optimizer @adamW, and use a learning rate of $1e^-4$.
After every epoch, we validate the model on CLIP's zero-shot image classification approach,
introduced in @clip_zero_shot_section, and select the best model based on the achieved accuracy. The representations for the zero-shot
classification are generated by the shared encoder of the student model, which we define as $bold(h)_(v, K)$ and $bold(h)_(w, K)$.
The classification is performed on the validation set of ImageNet-1K @imagenet. At this point it is important to note that
the accuracy we report with CLIP zero-shot classification is actually not zero-shot. This is because the teacher model is trained
supervised on ImageNet-1K, and the student model is trained using the teacher's probability distribution over the ImageNet-1K classes.
Our student model therefore learns the ImageNet-1K classes directly, even though we do not train on ImageNet-1K directly.
However, the accuracy we achieve still gives us a good indication of the quality of the student model's representations.

As mentioned before, we tokenize the text using the uncased BERT tokenizer. Again, uncased means that all text is lowercased before tokenization.
Inspired by BEiT-3, weset the maximum text length, the length of the captions, to 64 tokens @beit3, truncate longer captions and pad shorter captions.
This reduces the time required for a forward pass of the text encoder, and the captions of the data we collect are on average not larger than
15 tokens anyway (see @vl_dataset_summary).

For image augmentation, we use the same techniques as in the unimodal image KD experiment (@unimodal_kd_data2vec2_distillation).
However, we make one important distinction in the size of the random crop: As seen in @distil_data2vec2_hyperparameters, the range of the random crop
size is between 0.08 and 1.0 of the original image size. The lower bound is quite small, but because student and teacher receive the same
crop, even if it is very small, the student can still learn the teacher's representation for a small part of an image, generated by the crop.
However, this gets problematic with image text pairs. If the crop is very small, then important semantic information of the image might be lost,
which is still present in the text. Therefore, the resulting probability distribution of the teacher for that small crop might not be representative
of the image's high-level content, which is described by the text. This could result in the student predicting a probability distribution that is
correct with respect to the whole image, but not with respect to the small crop. To avoid this, we set the lower bound of the random crop size
to 0.9, which is also the value used by VLMo @vlmo. This ensures that the crop is large enough to capture the high-level content of the image.
A visualization of too small crops is shown in @vl_crop_size_bad, and a visualization of a minimum crop size of 0.9 is shown in @vl_crop_size_good.

All hyperparameters as summarized in @transformer_shre_hyperparams, pytorch pseudocode for the forward pass can be found in @transformer_shre_forward_pseudocode.

===== Results
We report the results of CLIP zero-shot classification on the validation set of ImageNet-1K @imagenet, as well as the results of
image-text retrieval on the test sets of COCO @coco and Flickr30k @flickr30k. The latter is show in @image_text_retrieval_shre_transformer_first.
Note that we do not report results on unimodal downstream tasks like image classification
or text classification. This is because finetuning is expensive, which is why will refrain from doing so until we reach our final improved
approach.

While the results on image-text retrieval are significantly worse than the state-of-the-art, we can still observe that we are not far off
from the performance of FLAVA @flava. Considering that FLAVA was developed by a team of researchers from Meta AI, and that this is our first
iteration of a multimodal model, the results are promising. From CLIP, VLMo, and BEiT-3 we are still far off, but this can at least partly
be attributed to the fact that those model are significantly larger than ours, and that they have been trained on much more data, which we 
show in @models_data_size_comparison.
The increased performance on Flickr30K compared to COCO can be attributed to the fact that the Flickr30K dataset is smaller. For a given
image, there are 5 correct captions in only 5k possible captions (25k for COCO),
and for a given caption there are only 1k possible images (5k for COCO).


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
    [FLAVA @flava], [42.74], [76.76], [-], [38.38], [67.47], [-], [67.7], [94.0], [-], [65.22], [89.38], [-],
    [CLIP @clip], [58.4], [81.5], [88.1], [37.8], [62.4], [72.2], [88.0],[98.7], [99.4], [68.7], [90.6], [95.2],
    [VLMo @vlmo], [74.8], [93.1], [96.9], [57.2], [82.6], [89.8], [92.3], [99.4], [99.9], [79.3], [95.7], [97.8],
    [BEiT-3 @beit3], [*84.8*], [*96.5*],[*98.3*], [*67.2*], [*87.7*], [*92.8*], [*98*], [*100*], [*100*], [*90.3*], [*98.7*], [*99.5*],
    [$"SHRe"_T$ (ours)], [37.06], [67.74], [79.7], [25.3], [54.73], [68.19], [49.9], [78.5], [88.5], [37.04], [67.34], [77.38],
    table.hline(),
  ),
  caption: [],
)<image_text_retrieval_shre_transformer_first>
#show table: set text(12pt)

Since we are using the same approach as SHRe, we also provide a comparison of the average median rank on the COCO test set.
How SHRe reports the average median rank is described in @shre_section. Unfortunately, the authors 
do not provide which pairs they exacly use to calculate the average median rank. We therefore opt for an approach that should closely
resemble that of SHRe: We select all 5k images from the COCO test set, and split them into 5 distinct sets of 1k images each.
For each set we do the following: For each image one caption, out of the 5 captions available, is selected.
We then have 5 splits of 1k image-caption pairs each. In each split, for one candidate there is only one correct target and 999 incorrect targets.
We the perform retrieval on each split, and calculate the median rank of the correct target. The average median rank is then the average
of the median ranks over all 5 splits. This procedure itself is repeated 5 times, so that each image is paired with each of its
5 correct captions exactly once. The result is 5 average median ranks, which are then averaged to
get the metric reported in @transformer_shre_amr_results.
Our approach significantly improves the baseline of SHRe, and the results indicate that the correct pair for a query is
mostly either the first or second retrieved item, though the results do not account for outliers, since the median rank is used.
The reason for the almost perfect results, considering the minimum possible value is 1, can be attributed to the advances in
Deep Learning research since SHRe was published, which was in 2017 @shre. Some of the advances include the Transformer architecture,
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