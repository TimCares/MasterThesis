== Scaling up the final Approach <scaling_up>
Even though our approach is specifically build for efficiency, and therefore smaller than traditional
vision-language models like BEiT-3 @beit3, VLMo @vlmo, or CLIP @clip, we still want to investigate
how our approach scales with more parameters. This will help us understand whether the approach
may be used in contexts where efficiency is not the main concern, but rather performance.

*Parameters*\
Oriented on the empirical studies on vision-language models by the authors of METER @meter, we double
the size of the image and text encoder from 6 to 12 layers. Both are still initialized with
weights from Data2Vec2 @data2vec2 and BERT @bert for the image and text encoder, respectively.
However, since both Data2Vec2 and BERT consist of 12 layers, we will use the full model of both
approaches, so our image encoder is now Data2Vec2, and our text encoder is BERT.
For encoders of this size, VLMo and BEiT-3 introduce 2 shared layers, and we do the same. Consequently,
or shared encoder now consists of 2 Transformer layers instead of one.

*Dataset*\
Based on the neural scaling laws formulated by OpenAI (described in @introduction_efficiency) it would be
logical to also increase the size of the dataset when scaling up the model. However, we are limited by 1TB
of storage, and therefore cannot increase the size of the dataset. Furthermore, collecting more data is not
as easy as increasing the model size, as datasets of the magnitude used in this work and by other vision-language
models have to be scraped with great expense from the internet, which is time-consuming and costly.

*Compute*\
We therefore opt to increase the compute as the third factor of the neural scaling laws. Currently, we train
S-SMKE for only 7 epochs, which is enough to reach convergence, as each epoch includes more than 3M image-text pairs.
With the increased model size, we will train for 15 epochs, and use 2 NVIDIA A100 80GB GPUs instead of 2 NVIDIA RTX 4090 24GB GPUs.
We do this not only to increase the compute, but also to ensure that our upscaled model fits into the memory of the GPUs.
Beforehand, S-SMKE already almost maxed out the 24GB of the RTX 4090, and with the doubled model size, we would not be able to
train the larger model.

*Loss*\
The contrastive loss $cal(L)_"CL"$ is still applied to the last shared Transformer layer. Since we only
used one shared layer in the previous experiments, the contrastive loss was basically applied to "all" shared
layers. Now, we do not apply the contrastive loss to the first shared layer, but only to the *final output* of the *second* one.
This is because BEiT-3 @beit3, VLMo @vlmo, and FLAVA @flava also only apply the contrastive loss to the
last shared layer, which has empirically shown to lead to the best performance. A reason for this could be
that constraining intermediate shared layers to align image and text representations through the contrastive
loss will lead to less flexibility in how the shared encoder processes the input data. Further, the
only point in the model where the image and text representations actually have to be aligned is at the layer
whose output is used for tasks like image-text retrieval. Since the alignment at intermediate layers has no
practical relevance, it is not necessary to enforce it. This would only unnecessarily reduce the freedom of
the model and increase the computational cost, as the contrastive loss has to be computed for multiple layers.

The contrastive loss now changes from:

$
cal(L)_("CL") &= 1/2 * (cal(L)_("CL"') + cal(L)_("CL"''))\
&= 1/4cal(L)_("CL"')^("i2t") + 1/4cal(L)_("CL"')^("t2i") + \
& quad thin 1/4cal(L)_("CL"'')^("i2t") + 1/4cal(L)_("CL"'')^("t2i")
$ <contrastive_loss_small>

To the following:

$
cal(L)_("CL") &= 1/2 * cal(L)_("CL"'')\
&= 1/2cal(L)_("CL"'')^("i2t") + 1/2cal(L)_("CL"'')^("t2i")
$ <contrastive_loss_big>

For the loss $cal(L)_("CL"'')$ we keep using the representation $bold(h)''_(v, K, mono(["I_CLS"]))$ for the image and
$bold(h)''_(w, K, mono(["T_CLS"]))$ for the text, which are the final outputs of the shared encoder for tokens
$mono(["I_CLS"])$ and $mono(["T_CLS"])$, respectively. As before, these are also the representations that are later
used for the image-text retrieval. Note that since we use the full Data2Vec2 and BERT models, it now holds that
$L_s=12$, and because we use 2 shared layers, $K=U+L_s$, where $U$ denotes the number of shared layers. Therefore,
$U=2$ and $K=2+12=14$.

*Training*\
As previously mentioned, we now train for 15 epochs, and use 2 NVIDIA A100 80GB GPUs. The batch size is increased
to the maximum that fits into the memory of the GPUs, which is 512. We keep communicating the representations used
for the contrastive loss between both GPUs, so that the batch size on which the contrastive loss is computed is 1024.
We keep using BEiTv2 @beitv2 as the teacher, and the contrastive target loss with a memory bank for the
knowledge distillation loss stays the same as well.

The authors of METER @meter suggest to use a lower learning rate for the pretrained unimodal (image and text) encoders,
which ensures that the unimodal encoders largely retain their pretrained knowledge. Since the shared encoder is
initialized randomly, we use a higher learning rate for it.