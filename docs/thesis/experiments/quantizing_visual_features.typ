=== Quantizing Visual Features <quantizing_visual_features>
==== Motivation
Even though we were able to reduce the impact of the image-specific information in the teacher's $mono(["I_CLS"])$
token by introducing the contrastive target loss, the problem remains the same.
A repeated glance on the comparison between the training loss for the image and text component of the $cal(L)_"KD"$ loss
(@ctl_component_comp), which is now based
on a contrastive loss with memory bank, shows that the loss for the image component $cal(L)^"i2i"_"KD"$ is still significantly lower
than that for the text component $cal(L)^"t2i"_"KD"$. While our approach is able to achieve impressive results even with this imbalance,
we push the boundaries by introducing an additional component that aims to further reduce the impact of the image-specific information.

#figure(
  image(
  width: 75%,
  "../figures/ctl_component_comp.png"),
  caption: [
    Training loss for the image and text component of the $cal(L)_"KD"$ loss. Even though the contrastive target loss
    is able to increase the performance of the model, the loss for the image component $cal(L)^"i2i"_"KD"$ is still significantly lower
    than that for the text component $cal(L)^"t2i"_"KD"$.
  ],
) <ctl_component_comp>

The paper "Neural Discrete Representation Learning" by van den Oord et al. @neural_discrete_representation_learning first introduced
the concept of quantizing images to discrete values in order to learn representations. The idea is to learn a so-called codebook
that contains a finite set of prototype representations/embeddings. An autoencoder architecture is then used to first
encode and compress the image to a continuous representation, which could be a sequence of representations for vision Transformers,
or a set of 2D feature maps for convolutional neural networks.

Staying at the Transformer architecture, each token of the sequence, which represents an image patch, is then compared to all prototype
representations in the codebook using e.g. consine similarity. The closest prototype to each token is then selected, and each
token is replaced by the prototype representation it is closest to. Based on this sequence of prototype representations, the decoder
is then trained to reconstruct the original image. During training, each prototype is updated
using for example the mean over all patch representations that were closest to it. This is similar to the k-means
algorithm, and is called vector-quantization @quantizing_visual_features @beitv2.

The consequence is that it is possible to discretize patch-level representations, which are usually continuous,
to a finite set of prototype representations.
This technique is using in papers like BEiTv2 @beitv2 to perform e.g. masked image modeling to learn visual representations @beit @beitv2.

We aim to apply vector-quantization not on the patch-level representations, but on the image-level representation $mono(["I_CLS"])$.
The goal is discretize the representation space of a self-supervised image model, such as the teacher we use, to obtain a set of prototype
embeddings we can map the representations produced by the model to. This way, we 
could find something similar to classes produced by a supervised model, but without the need for labels, which should
remove large amounts of image-specific information.
Intuitively, this can be seen as some form of clustering the representations produced by a self-supervised model, and closely resembles
applying k-means clustering on representations produced by a model from a set of images.

==== Method
To achieve this, we first take our pretrained teacher model BEiTv2 @beitv2, pass an image through it, and extract the 
representation $bold(h)^t_(v, L_t, mono(["I_CLS"])) in RR^768$ of the $mono(["I_CLS"])$ token. This functions as the representation of the image,
and the weights of the teacher remain frozen throughout the whole process. The representation is then passed through a learnable linear projection
$bold(W)_(q arrow.b) in RR^(D times S)$ that projects the representation into a lower-dimensional space with $s$ dimensions, where $D$ is the dimensionality
of the original representation.

The projected representation $bold(h)^t_(v, L_t, mono(["I_CLS"]))bold(W)_(q arrow.b)$ is then compared to all prototype embeddings ${bold(q)_j}_(j=1)^J$
in the codebook $bold(Q) in RR^(J times S)$ using cosine similarity, and the index $m$ of the closest prototype is extracted.

$
m = op("argmin", limits: #true)_(1 lt.eq s lt.eq S) delta(bold(h)^t_(v, L_t, mono(["I_CLS"]))bold(W)_(q arrow.b))delta(bold(q)_s)^T
$

We use the projection $bold(W)_(q arrow.b)$, because this already reduces the dimensionality of the representation, forcing only the most important
information to be kept. At best, pixel-level information is completely removed. Furthermore, a projection to a lower-dimensional space
has shown to alleviate the problem of codebook collapse, where all representations are mapped to only a few, or even just one, prototype @beitv2.
If codebook collapse occurs, the model is not able to learn distinct representations, and
there would not be a set of prototypes that represent different semantic concepts, like classes in a supervised model.

The index $m$ of the closest prototype is then used to select the corresponding prototype embedding $bold(q)_m$ from the codebook $bold(Q)$, which is then
projected back to the original embeddings dimension $D$ using yet another learnable linear projection $bold(W)_(b arrow.t) in RR^(S times D)$.
The projected prototype embedding $bold(q)_m bold(W)_(b arrow.t)$ is then concatenated with the original image representation
and input to the Transformer layers of the BEiTv2 model, and serves as the $mono(["I_CLS"])$ token.

Then, each patch/token representation of the image ${bold(h)^t_(v, 0, n)}_(n=1)^N$ is replaced by a special learnable mask token
$bold(h)_mono(["MASK"])$ with a certain probability $p$. We sample the probability from a uniform distribution $alpha tilde U(0,1)$:

$
ohm(bold(x)) &:= cases(
  bold(h)_mono(["MASK"]) &"if" p > alpha,
  bold(x) &"else",
) \
bold(H)'_(v, L_t) &= [bold(q)_m bold(W)_(b arrow.t), ohm(bold(h)^t_(v, 0, 1)), ..., ohm(bold(h)^t_(v, 0, N))]
$

Let's break down why we do this: The idea is that $bold(H)'_(v, K)$ becomes input to a set of Transformer layers, which form the decoder.
The task of the decoder is to reconstruct the original representation $bold(h)^t_(v, L_t, mono(["I_CLS"]))$ of the image, produced
by the frozen teacher. Because of self-attention in the Transformer layers, the decoder is able to use both the content and the
spacial information of the unmasked tokens to reconstruct the teacher's $mono(["I_CLS"])$ token. To make the whole process rely on
quanized global image representation $bold(q)_m$, we select $p$ to a values close to 1, which is $p=0.9$ in our case, so 90% of the tokens
are replaced by the mask token. This way, the decoder will not be able to adequately reconstruct the image representation
$bold(h)^t_(v, L_t, mono(["I_CLS"]))$ without fully utilizing the quantized global image representation $bold(q)_m$. The only way through
which high level information about the image content can reach the decoder is through $bold(q)_m$, which should force the model to
learn a set of prototypes that represent different semantic concepts in the image. The few unmasked tokens should help the model to
have access to (1) spacial information, and (2) low-level information about the image content, which should not be captured by $bold(q)_m$,
and is generally not needed in $bold(q)_m$, because the unmasked tokens provide this information.

Say an image contains a picture of a sheep on a meadow with a blue sky in the background. Since $bold(q)_m$ can only represent relatively
little information (due to its low dimensionality and discrete nature), the model may only compress the most important information, like
the sheep, the meadow, and the sky, into $bold(q)_m$. The few image patches that are not masked out might contain information about the
spacial arrangement, or any other information originally captured by $bold(h)^t_(v, L_t, mono(["I_CLS"]))$. Those information should
be image-specific. That way, the model should be able to learn a set of prototypes that represent different semantic concepts in the image.
Examples of masking random image patches are illustrated in @rand_mask_ex.

The training objective is then:

$
cal(L)_(I-V Q) =& \
&1 - delta(bold(h)^t_(v, F, mono(["I_CLS"])))delta(bold(h)^t_(v, L_t, mono(["I_CLS"])))^T \
&+ ||delta(bold(h)^t_(v, L_t, mono(["I_CLS"]))bold(W)_(q arrow.b)) - op(s g)[delta(bold(q)_m)]||_2^2
$ <i_vq_loss>

Here, $bold(h)^t_(v, F, mono(["I_CLS"]))$ denotes the decoder output for the $mono(["I_CLS"])$ token, which is the one we want to reconstruct.
$F$ denotes the number of all Transformer layers of the encoder, the frozen BEiTv2 model, and the decoder. It holds that $F = L_t + U$, where
$U$ is the number of Transformer layers in the decoder. Since for our teacher it holds that $L_t = 12$, we have $F = 12 + U$.

The first term, $1 - delta(bold(h)^t_(v, F, mono(["I_CLS"])))delta(bold(q)_m)^T$, aims to maximize the cosine similarity between
the reconstructed $mono(["I_CLS"])$ token and the original representation $bold(h)^t_(v, L_t, mono(["I_CLS"]))$ for the image, produced by the teacher.
If the cosine similarity is maximized, so 1.0, then this part becomes 0. This is the reconstruction loss.
The other component aims to minimize the distance between the projected representation $bold(h)^t_(v, L_t, mono(["I_CLS"]))bold(W)_(q arrow.b)$
and the closest, and therefore selected, prototype $bold(q)_m$. It is the mean squared error.

The form $op(s g)[dot]$ denotes a stop-gradient operation. This operation is required, because the codebook lookup, i.e. replacing
the representation $bold(h)^t_(v, L_t, mono(["I_CLS"]))bold(W)_(q arrow.b)$ with the closest prototype $bold(q)_m$, is not differentiable.
It is not possible to compute gradients that express how $bold(W)_(q arrow.b)$, $bold(W)_(q arrow.t)$ and $bold(Q)$ should be updated
to minimize the loss, since the lookup cannot be expressed as a continuous function.

The solution is to compute the loss, and therefore the gradients, only with respect to the encoder and decoder 
weights. Since the encoder weights are frozen, gradients only need to be computed for $bold(W)_(q arrow.b)$, $bold(W)_(q arrow.t)$, and
the decoder weights. We treat the selected codebook embedding $bold(q)_m$ as constant, meaning that we do not compute gradients for it.
That is why in the loss, $op(s g)[dot]$ turns its input into a constant, and therefore avoids gradient computation for $bold(q)_m$. 

The problem this creates is that the gradients flow through the decoder and $bold(W)_(q arrow.t)$, but are they "cut off", leading to
no update of the projection $bold(W)_(q arrow.b)$, and the codebook $bold(Q)$. To solve this, the gradients with respect to
$bold(q)_m$ are copied to $bold(h)^t_(v, L_t, mono(["I_CLS"]))bold(W)_(q arrow.b)$, bypassing how $bold(q)_m$ was selected. For the weight
update of $bold(W)_(q arrow.b)$ this essentially means $bold(h)^t_(v, L_t, mono(["I_CLS"]))bold(W)_(q arrow.b)$ and $bold(q)_m$
its "replacement" are treated as the same.

This loss however does not lead to an optimization of the codebook $bold(Q)$, as, again, $bold(q)_m$ is treated as constant whose
origin is basically unknown to the model. This is done on purpose, as each codebook embedding is optimized/updated as an exponential moving average
(EMA) of the representations in the current batch (size $B$) it replaced. For a single codebook embedding $bold(q)_j$,
this set can be defined as:

$
bold(A)_j = {(bold(h)^t_(v, L_t, mono(["I_CLS"]))bold(W)_(q arrow.b))_i | i in [1, B] and m_i = j}
$

Here, $m_i$ is the index of the closest prototype for the $i$-th image in the batch. The update of the codebook embedding $bold(q)_j$
is then:

$
bold(q)_j = d * bold(q)_j + (1 - d) * 1/Z sum_(z=1)^(Z) (bold(A)_j)_z
$

The codebook embedding $j$ is updated with the mean over all representations in the batch it was closest to.
$d$ denotes the decay factor of the EMA update, normally specified as $m$. However, we use $d$ to avoid confusion with the index $m$ of the closest
prototype.
We show can illustration of the concept in @image_vq_fig. The loss formulation in @i_vq_loss, as well as the EMA update, is heavily
inspired by the vector quantization process in BEiTv2 @beitv2.

#figure(
  image(
  width: 100%,
  "../figures/image_vq.png"),
  caption: [
    Illustration of the vector quantization of a self-supervised image model's representation space. Models with dotted borders are frozen,
    and red lines indicate gradient flow. Opaque outputs, in this case patch representations, are ignored in the loss computation.
    The figure is inspired by papers "Neural Discrete Representation Learning" @neural_discrete_representation_learning, and
    BEiT @beit, heavily oriented on the depiction of the vector quantization process in BEiTv2 @beitv2, and the image stems
    from the COCO test set @coco.
  ],
) <image_vq_fig>


==== Training Image VQ
For training, we experiment with embedding dimensions of $S=16$ and $S=8$ for codebook.
For comparison, BEiTv2 uses $S=32$ for its patch-level
codebook. As might have come apparent from the previous section, we heavily orient on the vector quantization process in BEiTv2 @beitv2,
which is why we originally used $S=32$ as well. However, we found that this led to codebook collapse in preliminary experiments, which
is why we opt for a lower dimensionality. For the codebook size, we experiment with $J=1024$. This is motivated
by the fact that we aim to learn a set of prototypes that represent different semantic concepts in the image, like classes in a supervised model.
Since we use BEiTv2, which is our encoder, was pretrained on ImageNet-1K @beitv2,
we orient the number of possible concepts on the number of classes
in ImageNet-1K, which is 1000 @imagenet. The latter is used to investigate whether a larger codebook size can still capture semantic concepts
without image-specific information.

For the decoder, we merely use a single Transformer layer initialized from scratch. Few decoder layer means lower capacity to reconstruct
the image representation, which should force the model to rely even more on the quantized global image representation $bold(q)_m$.
This is also done in BEiTv2, where the authors experiment with a single, and three decoder layers. Consequently, for us it holds that $U=1$,
and $F=L_s + U = 12+1 = 13$. For the decay factor, we again orient on BEiTv2, and use $d=0.99$. As mentioned in the previous section,
we use a masking probability of $p=0.9$. All hyperparameters, including the ones described here, are shown in @image_vq_cls_hparams.

We train the model for 10 epochs on ImageNet-1K, and use a relatively large learning rate of $1e-3$. We do this, because we use two
GPUs with a combined batch size of 512, which is relatively large. Further, we do not have have a large number of trainable parameters,
only the projection matrices $bold(W)_(q arrow.b)$, $bold(W)_(q arrow.t)$, and the weights of one Transformer layer (decoder).
Simultaneously, we use a frozen teacher model, which acts as a feature extractor and provides high quality representations.

After each epoch, we validate the loss $cal(L)_(I-V Q)$ on the validation set of ImageNet-1K, and, most importantly,
calculate the codebook usage over the whole validation set to check for codebook collapse.

==== Insights

While the goal of the quantization process can only be measured by using the codebook embeddings when training S-SMKE,
we can still gain insights by visualizing the generated codebook embeddings and the codebook usage. We show the codebook usage
for different codebook configurations in @vq_codebook_usage. The codebook usage is calculated by collecting
all indices of the closest prototype for each image in the validation set, and then counting how many prototypes did not get
selected at all, from which we calculate the percentage.

#figure(
  image(
  width: 75%,
  "../figures/vq_codebook_usage.png"),
  caption: [
    Comparison of codebook utilization on the ImageNet-1K validation set for different codebook configurations. Variants are
    denoted in the form "VQ-$J$-$S$", with $J$ being the number of codebook embeddings, and $S$ the dimensionality of the embeddings.
  ],
) <vq_codebook_usage>

We observe that the codebook usage is generally higher for a lower dimension, where the configuration
with $J=1024$ and $S=8$ achieves a utilization of 100% early in training and maintains it throughout the whole training process.
In contrast, our configuration with a codebook size of 16 needs until the end of the training process to reach the same utilization.

We generally aim for a pattern that shows a codebook utilization of close to 100% early in training, and then maintains it
throughout the whole training process. This is because the model is forced to use all prototypes to best reconstruct the image
representation, but since there is no more capacity in the codebook (after all are used), the model is forced to make the prototypes
themselves as distinct from each other as possible. This will retain as much information as possible from the original representation,
but since at the same time
the codebook dimension is also low, the model is forced to compress the most important information into the codebook embeddings.
Consequently, the lower the codebook size, the more distinct the prototypes have to be, and the lower the codebook dimensionality,
the more information has to be compressed into the codebook embeddings, which leads to more high-level semantic content being captured.

The content that is captured by the codebook embeddings can be visualized by showing the set of images that are closest to a certain
prototype, which yields clusters of images that should be semantically similar. Examples of clusters for VQ-1024-8 are
shown in @image_vq_8_examples, and for VQ-1024-16 in @image_vq_examples. While both approaches show clusters of semantically
similar images, the clusters for VQ-1024-8 appear to be more distinct. Some clusters can even be considered as classes.
For example, codebook embedding 784 appears to capture the concept of a "tucan", and codebook embedding 1022 the concept of
a "turtle". However, in both cases the images of some codebook embeddings are not always as similar as one would expect
from a class level semantics. While
the images usually have common color or show similar objects with the same form,
they are not always semantically related. One examples of this can be
seen in codebook embedding 533 of VQ-1024-16 in @image_vq_examples.

Is is important to note that 
we do not select the quantizer configuration to use for training S-SMKE based on the reconstruction loss
$1 - delta(bold(h)^t_(v, F, mono(["I_CLS"])))delta(bold(h)^t_(v, L_t, mono(["I_CLS"])))^T$, because a larger codebook size and
dimensionality will always lead to a lower reconstruction loss. This is because more information from the original representation
can be captured by the codebook embeddings, thereby making the reconstruction easier.

Based on the examples provided in both @image_vq_8_examples and @image_vq_examples, we can conclude that the quantization process
is able to capture high-level semantic content in the image representation, although the members of the clusters are not always
consistent. Whether the quantization actually helps to reduce the gap between the image and text component of the knowledge distillation
is yet to be seen, and will be evaluated in the next section.

==== Training S-SMKE with Image VQ

We train S-SMKE using a classification task, which is now our knowledge distillation loss. The goal is to predict
the class, or rather index $m$, of the closest prototype $bold(q)_m$ to the representation $bold(h)^t_(v, L_t, mono(["I_CLS"]))$
of a candidate image, produced by BEiTv2. BEiTv2 is part of the image quantizer.
Because the index $m$ indirectly encodes what prototype $bold(q)_m$ semantically encodes,
it can be considered as a class label for the image. This class label is predicted by the student model when receiving the same
image, and again for the caption of the image.
The representations used as the predictions are $bold(h)'''_(v, K, mono(["I_CLS"])) in RR^J$ and $bold(h)'''_(w, K, mono(["T_CLS"])) in RR^J$
for the image and caption, respectively. As they now represent logits for the class/codebook prediction they have
as many dimensions as there are codebook embeddings, which is $J=1024$. Let $m$ denote the index of the closest prototype to
$bold(h)^t_(v, L_t, mono(["I_CLS"]))$, then $m$ is the correct class that needs to be predicted by the student. The loss, which is
the cross-entropy loss, for a single image-caption pair is then:

$
bold(hat(y))^w &= bold(h)'''_(w, K, mono(["T_CLS"])) \
cal(L)^"t2i"_"KD" &= -log exp(hat(y)^w_m) / (sum_(j=1)^(J) exp(hat(y)^w_j)) \

bold(hat(y))^v &= bold(h)'''_(v, K, mono(["I_CLS"])) \
cal(L)^"i2i"_"KD" &= -log exp(hat(y)^v_m) / (sum_(j=1)^(J) exp(hat(y)^v_j)) \
$

We introduce $bold(hat(y))^w$ and $bold(hat(y))^v$ for better readability, and $hat(y)^w_m$ denotes the logit for the correct class $m$
based on the caption, and $hat(y)^v_m$ the logit for the correct class $m$ based on the image. The loss is again the mean of both losses:

$
cal(L)_"KD" = 1/2 * (cal(L)^"t2i"_"KD" + cal(L)^"i2i"_"KD")
$

All other components, including hyperparameters, of the training process remain the same. Pytorch pseudocode for the training process
is shown in @s_smke_vq_forward_pseudocode.

==== Insights

Unfortunately, the training of S-SMKE with the image quantizer does not work as expected. The performance on all metrics decreases
significantly compared to memory bank contrastive target loss, displayed in @contrastive_target_vs_vq. Further, the quantization
is not able to reduce the gap between the image component $cal(L)^"i2i"_"KD"$ and text component $cal(L)^"t2i"_"KD"$
of the knowledge distillation loss. The accuracy, shown in @kd_acc_components_vq, is significantly higher for the image component, reflecting 
the lower loss for the image component seen in previous experiments.

#figure(
  image(
  width: 50%,
  "../figures/kd_acc_components_vq.png"),
  caption: [
    Comparison of accuracy between the image and text component of the knowledge distillation.
    Predicting to which codebook embedding index $m$ a candidate image belongs to is easier when the image itself is used
    as an input. In contrast, the model struggles to predict the correct index $m$ when it receives the caption of the image.
  ],
) <kd_acc_components_vq>

#figure(
  table(
  columns: 4,
  stroke: none,
  table.hline(),
  table.header(
    table.cell(rowspan: 2, colspan: 1, align:horizon, [KD Loss]),
    table.cell(rowspan: 2, colspan: 1, align:horizon, [ImageNet-1K]),
    table.cell(colspan: 2, align:horizon, [Retrieval]),
    [MSCOCO],
    [Flickr30K],
  ),
  table.hline(stroke: .6pt),
  [S-SMKE#sub[CTL_MB]], [*33.0*], [*67.15*], [*78.3*],
  [S-SMKE#sub[VQ]], [25.1], [61.2], [75.3],
  table.hline(),
),
    caption: [
      A classifcation of prototype membership of images from image-text pairs decreases the performance on all benchmarks.
    ]
) <contrastive_target_vs_vq>

To find the reason for the decrease in performance, it is helpful to visualize to which prototypes different image-text pairs
from our training set are mapped to. In @caption_vq_clusters, we show a selection
of the training examples belonging to the same prototype an image from an image-text pair is mapped to.
The examples show that applying the quantization process to the image representation shows only partial similarities
between the members of a prototype $bold(q)_m$ and the image that was mapped to it, which was already noted before.
Especially the similarity between the caption of the mapped image should show a high similarity to the members of the prototype.
In the best case scenario, the caption should also be able to describe the members of the prototype, meaning it could also be
used as a caption for all images that are mapped to the prototype. However, this is not the case, and in most cases
the caption is completely unrelated to the members of the prototype.

We suspect that while precise captions of images in the training set are important for alignment, they are harmful when
the quantization is used. The last examples of @caption_vq_clusters show that while the image can be considered as semantically
similar to the members of a prototype, only the token "airplane" of the caption may be considered as a correct text/caption
for the semantics (represented by the members) the prototype encodes.
Additional context, like "parked in a stationary position", is accurately describing its paired image, but this does not match
other members of the prototype. The same holds for examples 2, where there are multiple members that contain a "plate", but none
of them are placed on a mat. In other cases, both image and text are completely unrelated to the members of the prototype.

Even though the quantization process can be considered a failure,
the approach still provides valuable insights with respect to clustering image representations.
If images contain only a single concept, like a "tucan" or a "turtle" (see @image_vq_8_examples),
then the quantization process is able to capture this high-level semantic content in a prototype embeddings.

It can be considered as an alternative to clustering image representations with e.g. k-means, as the process not only focuses
on moving the prototypes to the center of their cluster, which is done by k-means, but also on compressing the most important
information of the image into the prototypes. In contrast, k-means only operates on the raw representations. While it is possible
to compress raw representations using algorithms like PCA, those methods are more generic and do not explicitly
focus on capturing high-level semantic content. In contrast, the quantization process effectively learns which information
in the raw representation is important, and retrains them in the prototypes.

#figure(
  image(
  width: 100%,
  "../figures/caption_vq_clusters.png"),
  caption: [
    We visualize members of a prototype $bold(q)_m$ an image, from an image-text pair, is mapped to.
    In most cases, both caption and image are unrelated to the members of their matching prototype.
  ],
) <caption_vq_clusters>
