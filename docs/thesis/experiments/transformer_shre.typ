== Multimodal Knowledge Distillation <multimodal_knowledge_distillation>
=== Transformer SHRe <transformer_shre>
Before we develop an end-to-end self-supervised approach to multimodal knowledge distillation, we first follow the approach
of SHRe @shre and develop a multimodal knowledge distillation with a probability distribution over the classes as the prediction
target. This allows us to closely observe the impact of switching from a supervised to a self-supervised teacher model on the
student model's performance. Moreover, it allows us to gradually increase the complexity of our approach and build on our
previous advancements.

==== Method
What makes our approach different from SHRe is that we use a langauge Transformer as the text encoder, and
a vision Transformer as the image encoder, bringing us closer to a unified architecture.
In contrast, SHRe uses 2D convolutions and 1D convolutions for the image and text, respectively.
The second change lies in the ranking loss of SHRe, which we replace by the contrastive loss introduced by CLIP @clip, and
explained in @vision_language_contrast. We justify this decision with the fact that vision-language contrast has become the
de-facto standard for multimodal self-supervised learning, and has lead models like CLIP @clip, VLMo @vlmo, and CoCa @coca to reach
state-of-the-art results in image-text retrieval.

Pretrained unimodal parts, because we can reuse feature learned from unimodal pretraining -> BeiTv3, VLMo and FLAVA show peroformance gains
with this approach.