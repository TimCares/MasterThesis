== Outlook
This work has demonstrated the feasibility of generating a vision-language model from unimodal
components through an end-to-end self-supervised learning approach. The resulting model,
while not reaching the performance of state-of-the-art vision-language models, shows
promising results across various benchmarks.

*Towards a General Framework*\
While this work introduces efficient multimodal
learning on the example of vision-language models, the question arises whether this approach
can be extended to other modalities, such as audio or video. While the methodology presented
is not restricted to vision and language, as it can be adapted by using a teacher and
pretrained models (for initialization) from other modalities, the success of this approach
still needs to be practically demonstrated. Even though SHRe @shre has shown that the approach allows the
alignment of vision, language, and audio, the question remains whether this can also be achieved
through an end-to-end self-supervised learning approach.

To really ensure the general applicability of this approach, it would be necessary to demonstrate
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

Consequently to actually ensure a many-to-many alignment, it would be necessary to
either collect a single dataset where all modalities form a pair (e.g. image-text-audio triplets), or to collect
multiple datasets with each aligning two of the modalities. The latter approach has the disadvantage that for
the alignment of $n$ modalities $2^n$ multimodal datasets, and potentially $n$ unimodal datasets, would be required.
For the former approach, with an increased number of modalities, finding or
constructing such, especially large, datasets becomes increasingly difficult.

This is where the presented approach could be beneficial, as there are usually pretrained models available for
each modality, and the approach generally requires less data. Furthermore, as previously mentioned, the approach,
as least with a supervised teacher, has shown to be able to align multiple modalities with a transitive method (even though
this leads to suboptimal performance).

We therefore deem it as critical to explore the alignment of more than two modalities with an end-to-end self-supervised
learning approach so that our philosophy of general modality-invariant representations can be fully realized.

*Positional Encoding*\
We initialized our multimodal model with components from pretrained unimodal models, which were Data2Vec2 @data2vec2
for the image encoder and BERT @bert for the text encoder. The representations of these models were they passed seperately
through a shared Transformer encoder. However, we only briefly considered what implications this has for the shared
Transformer encoder: We introduced a learnable token-type embedding, which is used to distinguish between both modalities
(see @token_type_embeddings), which is especially important for the self-attention mechanism of the Transformer as the sequence
of representations has an inherently different meaning for each modality: 1D for text and 2D for images.

A part of these sequences are the positional encodings of the image and text encoder, respectively. These positional encodings
are used to provide the Transformer with information about the position of each token in the sequence. However, the positional
encodings of the image and text encoder are of a different type. While the positional encodings of the text encoder (BERT)
are learnable representations for each position, so a learnable absolute positional encoding, the positional encoding of the
image encoder (Data2Vec2) is a fixed sinusoidal positional encoding. Therefore, the shared Transformer encoder not only has
to account for the difference between the modalities, but also for the difference in positional encodings.
In contrast, other vision-langaue models like BEiT-3 @beit3 or VLMo @vlmo use the same positional encoding for both modalities,
so it would also be worth investigating the impact of using the same positional encoding for both modalities.

A caveat of using the same positional encoding type for both modalities is that it greatly restricts the flexibility of the
choice of the pretrained unimodal models we use to initialize the modality-specific encoders. One would have to find
pretrained unimodal models that all use the same positional encoding type, which becomes increasingly difficult with
more modalities.

*Modality-Specific Bias*\

//*Fine-Grained Alignment*\