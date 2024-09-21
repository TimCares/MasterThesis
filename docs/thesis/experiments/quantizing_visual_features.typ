=== Quantizing Visual Features <quantizing_visual_features>
==== Motivation
Even though we were able to reduce the impact of the image-specific information in the teacher's $mono(["I_CLS"])$
token by introducing the contrastive target loss, the problem remains the same.
A repeated glance on the comparison between the training loss for the image and text component of the $cal(L)_"KD"$ loss, which is now based
on a contrastive loss with memory bank, shows that the loss for the image component $cal(L)^"i2i"_"KD"$ is still significantly lower
than that for the text component $cal(L)^"t2i"_"KD"$. While our approach is able to achieve impressive results even with this imbalance,
we push the boundaries by introducing an additional component that aims to further reduce the impact of the image-specific information.

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
remove large amounts image-specific information.
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


==== Training
For training, we use an embedding dimension of the codebook of $S=16$. For comparison, BEiTv2 uses $S=32$ for its patch-level
codebook. As might have come apparent from the previous section, we heavily orient on the vector quantization process in BEiTv2 @beitv2,
which is why we originally used $S=32$ as well. However, we found that this led to codebook collapse in preliminary experiments, which
is why we opt for a lower dimensionality. We experiment with two different codebook sizes, $J=1024$ and $J=8192$. The former is motivated
by the fact that we aim to learn a set of prototypes that represent different semantic concepts in the image, like classes in a supervised model.
Since we BEiTv2, which is our encoder, was pretrained on ImageNet-1K @beitv2, we orient the number of possible concepts on the number of classes
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