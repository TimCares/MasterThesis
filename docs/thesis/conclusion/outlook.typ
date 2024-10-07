== Future Work
// *Positional Encoding*\
// We initialized our multimodal model with components from pretrained unimodal models, which were Data2Vec2 @data2vec2
// for the image encoder and BERT @bert for the text encoder. The representations of these models were they passed seperately
// through a shared Transformer encoder. However, we only briefly considered what implications this has for the shared
// Transformer encoder: We introduced a learnable token-type embedding, which is used to distinguish between both modalities
// (see @token_type_embeddings), which is especially important for the self-attention mechanism of the Transformer as the sequence
// of representations has an inherently different meaning for each modality: 1D for text and 2D for images.

// A part of these sequences are the positional encodings of the image and text encoder, respectively. These positional encodings
// are used to provide the Transformer with information about the position of each token in the sequence. However, the positional
// encodings of the image and text encoder are of a different type. While the positional encodings of the text encoder (BERT)
// are learnable representations for each position, so a learnable absolute positional encoding, the positional encoding of the
// image encoder (Data2Vec2) is a fixed sinusoidal positional encoding. Therefore, the shared Transformer encoder not only has
// to account for the difference between the modalities, but also for the difference in positional encodings.
// In contrast, other vision-langaue models like BEiT-3 @beit3 or VLMo @vlmo use the same positional encoding for both modalities,
// so it would also be worth investigating the impact of using the same positional encoding for both modalities.

// A caveat of using the same positional encoding type for both modalities is that it greatly restricts the flexibility of the
// choice of the pretrained unimodal models we use to initialize the modality-specific encoders. One would have to find
// pretrained unimodal models that all use the same positional encoding type, which becomes increasingly difficult with
// more modalities.

*Strengthening Unimodal Encoders*\
Finetuning S-SMKE on unimodal downstream tasks showed that our performance is generally worse on those taks compared to the performance
of unimodal models, including our unimodal distilled models. Since this is a limitation of our approach, it is worth investigating
how the performance of our model can be improved on unimodal tasks. 
BEiT-3 @beit3 solves this problem by including modality-specific pretraining tasks during the training of the multimodal model,
which we find as worth investigating for our approach as well. It would be relatively easy to include unimodal tasks
such as masked language modeling for the text encoder and masked patch prediction for the image encoder.

*Fine-Grained Alignment*\
As already mentioned in the limitations of S-SMKE (see @mm_kd_limitations), the model processes image and text seperately.
Even though this makes alignment on the level of global representations possible, it does not allow for fine-grained alignment
of image and text on the level of individual image patches and text tokens. This is a limitation that is shared with approaches
like CLIP @clip. This *limits* the actual application of S-SMKE to image-text retrieval, as other vision language tasks
like image captioning, visual question answering, or visual reasoning require individual image patches
to attend to individual text tokens, and vice versa. This is only possible through cross-modal attention, which is not
part of our approach. To ensure a wider applicability of our approach, it would be necessary to include such a cross-modal
attention mechanism by e.g. concatenating the image and text representations and passing them through the shared Transformer
layer(s). For details on cross-modal attention through concatenation we refer to BEiT-3 @beit3.

== Outlook

*Towards a General Framework*\
While this work introduces efficient multimodal
learning on the example of vision-language models, the question arises whether this approach
can be extended to other modalities, such as audio or video. While the methodology presented
is not restricted to vision and language, as it can be adapted by using a teacher and
pretrained models (for initialization) from other modalities, the success of this approach
still needs to be practically demonstrated. Even though SHRe @shre has shown that the approach allows the
alignment of vision, language, and audio, the question remains whether this can also be achieved
through an end-to-end self-supervised learning approach.

To really ensure the general applicability of our method it is necessary to demonstrate
the presented learning paradigm with a variety of modality combinations.

*Application on Many-to-Many Alignment*\
As of September 2024, there exists a variety of vision-language models, each of them presenting different
approaches to aligning vision and language. However, with the exception of SHRe @shre and AudioCLIP @audio_clip,
to our knowledge, there are no multimodal models that align more than two modalities.

Based on our intuition that the same concept, expressed in different modalities, should be aligned, this would be
a promising direction for future research and the next logical step in multimodal learning. While
there has been extensive research on vision-language models since 2020, with over 20 papers published,
there is still a lack of research on models that align more than two modalities.

One possible reason could be that with each additional modality, the complexity of the model increases,
as the model needs to learn to align more modalities, and each modality requires additional parameters
that are responsible for extracting low-level features. Furthermore, while vision-language models
only require image-text pairs, and potentially unimodal image and text data (see e.g. VLMo @vlmo and BEiT-3 @beit3),
for training, vision-language-audio models would either require image-text-audio triplets, or at least
two datasets that align two of the three modalities and form a transitive alignment between the three modalities.
One example would be the strategy of SHRe @shre, which aligns image, text, and audio by aligning image-text and image-audio pairs.
The authors show that this approach naturally leads to an alignment of audio and text, as they are both aligned with the image modality @shre.
However, even though the approach is successful, the alignment of audio and text does not work as well as the alignment of image and text
(see @shre_retrieval in the introduction of SHRe of @shre_section).

Consequently, to actually ensure a many-to-many alignment, it would be necessary to
either collect a single dataset where all modalities form a pair (e.g. image-text-audio triplets), or to collect
multiple datasets with each aligning two of the modalities. The latter approach has the disadvantage that for
the alignment of $n$ modalities $(n(n-1))/2$ multimodal datasets, and potentially $n$ unimodal datasets, would be required.
For the former approach, with an increased number of modalities, finding or
constructing such datasets becomes increasingly difficult.

This is where the presented approach could be beneficial, as there are usually pretrained models available for
each modality, and the approach generally requires less data. Furthermore, SHRe @shre has demonstrated that the
approach is able to align multiple modalities with a transitive method, although only with a supervised teacher.

We therefore deem it as critical to explore the alignment of more than two modalities with an end-to-end self-supervised
learning approach, so that our philosophy of general modality-invariant representations can be fully realized.
