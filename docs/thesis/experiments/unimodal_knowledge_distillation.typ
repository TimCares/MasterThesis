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
Additionally, the authors apply instance normalization to the activations of each layer before averaging, and then perform parameter-less layer normalization,
which we perform likewise.
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
cal(L)_("KD")(bold(Y), bold(hat(Y))) = ||bold(Y) - bold(hat(Y))||_2^2 = 1/(N+1) ( sum_(n=1)^M cal(L)_("MSE")(bold(y)_n, bold(hat(y))_n)
+ cal(L)_("MSE")(bold(y)_mono(["I_CLS"]), bold(hat(y))_mono(["I_CLS"])))
$

We denote $bold(y)_i$ and $bold(hat(y))_i$ as the average representation for image patch $i$ over all layers from the teacher and student model, respectively.
This includes instance norm before averaging, and layer norm afterwards.
$op("InstanceNorm")(dot)$ and $op("LayerNorm")(dot)$ are defined as specified in (TODO: cite notation),
and $cal(L)_("MSE")(dot, dot)$ is the mean squared error between two d-dimensional vectors, defined in (TODO: cite equation) in (TODO: cite notation).



==== Results
=== Language
==== Method
==== Results

#bibliography("../references.bib")