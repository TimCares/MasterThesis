#set math.equation(numbering: "(1)")
=== Token-Type Embeddings (TTE)
In order for the shared Transformer architecture to work, the shared Transformer block needs to be able to somewhat differentiate between both
modalities (image and text). Even though we desire an aligned representation to form in the shared block, especially the Self-Attention mechanism
still needs to be able to differentiate between the two modalities. This can be explained by the fact that for images the model needs to
find 2-dimensional spatial relationships, while for text only 1-dimensional relationships are required. Even though we saw in previous experiments
that the shared Transformer block is able to learn a good representation without any modality-specific information, we still want to investigate
how the performance of the model changes when we explicitly provide the shared Transformer block with the information which modality it
is processing. This nessessitates a special embedding for both image and text, which is added to each token of the respective modality, 
even the special tokens.
Our motivation in this change lies in the fact that VLMo @vlmo also follows this approach.

The intuitive approach to implement this, would be to directly follow VLM @vlmo and add the token type embeddings after the positional
encoding, before the input is fed into the Transformer blocks @vlmo. Our definition of the Transformer input changes as follows for text:

$
bold(H)^w_(0)&=[bold(w)_0^mono(["T_CLS"]), bold(w)_0^1, ..., bold(w)_0^M, bold(w)_0^mono(["T_SEP"])] + bold(T)^w_"pos" + bold(T)^w_"type", "with"\
bold(T)^w_"pos" &= [bold(t)^w_"pos"_mono(["T_CLS"]), bold(t)^w_"pos"_1, ..., bold(t)^w_"pos"_M, bold(t)^w_"pos"_mono(["T_SEP"])], "and" \
bold(T)^w_"type" &= [bold(t)^w_"type"_mono(["T_CLS"]), bold(t)^w_"type"_1, ..., bold(t)^w_"type"_M, bold(t)^w_"type"_mono(["T_SEP"])]
$

Similar holds for images:

$
bold(H)^v_(0)&=[bold(v)_0^mono(["I_CLS"]), bold(v)_0^1, ..., bold(v)_0^N] + bold(T)^v_"pos" + bold(T)^v_"type", "with"\
bold(T)^v_"pos" &= [0, bold(t)^v_"pos"_1, ..., bold(t)^v_"pos"_N], "and" \
bold(T)^v_"type" &= [bold(t)^v_"type"_mono(["I_CLS"]), bold(t)^v_"type"_1, ..., bold(t)^v_"type"_N]
$

The token type embedding is the same for every token and patch in the sequence, respectively. 

$
bold(t)^w_"type"_i = bold(t)^w_"type"_j, forall i,j in {mono(["T_CLS"]), 1, ..., M, mono(["T_SEP"])} \
bold(t)^v_"type"_i = bold(t)^v_"type"_j, forall i,j in {mono(["I_CLS"]), 1, ..., N} \
$

This follows VLMo @vlmo and 
is the case because each token/patch of a text/image input sequence is of the same modality.
The parameters added to the model are negligible, as they
only include two additional embeddings of size $D$, with $D=768$ this accounts for just $768*2=1536$ parameters.

// #figure(
//   table(
//   columns: (25%, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
//     stroke: none,
//     table.hline(),
//     table.header(
//       table.cell(rowspan: 3, colspan: 1, align:horizon, [*Model*]),
//       table.cell(colspan: 6, [*MSCOCO (5K test set)*]),
//       table.cell(colspan: 6, [*Flickr30K (1K test set)*]),
//       table.cell(colspan: 3, [Image $arrow.r$ Text]),
//       table.cell(colspan: 3, [Text $arrow.r$ Image]),
//       table.vline(stroke: .4pt),
//       table.cell(colspan: 3, [Image $arrow.r$ Text]),
//       table.cell(colspan: 3, [Text $arrow.r$ Image]),
//       table.hline(start: 1, end: 4, stroke: .2pt),
//       table.hline(start: 4, end: 7, stroke: .2pt),
//       table.hline(start: 7, end: 10, stroke: .2pt),
//       table.hline(start: 10, end: 13, stroke: .2pt),
//       [R@1], [R@5], [R@10], [R@1], [R@5], [R@10], [R@1], [R@5], [R@10], [R@1], [R@5], [R@10]
//     ),
//     table.hline(stroke: .4pt),
//     [*Sx3HRe#sub[TTE]*], [50.64], [79.28], [88.02], [35.34], [65.47], [77.07], [67.10], [89.10], [94.40], [50.70], [77.52], [85.32],
//     table.hline(),
//   ),
//   caption: [],
// )<image_text_retrieval_tte_first>
We present the results in @image_text_retrieval_tte_first, and show that the variant with TTE does not improve the performance of the model.

#figure(
  table(
    columns: 5,
    stroke: none,
    table.hline(),
    table.header(
      [Model],
      [ImageNet-1K Top-1 Accuracy],
      [ImageNet-1K Top-5 Accuracy],
      [Retrieval MSCOCO],
      [Retrieval Flickr30K],
    ),
    table.hline(stroke: .6pt),
    [EMKUM], [26.1], [50.4], [66.3], [77.83], 
    [EMKUM#sub[TTE]], [25.6], [49.5], [66.0], [77.36],
    table.hline(),
  ),
  caption: [Introducing token-type embeddings (TTE) @vlmo after the positional encoding degrades performance slightly. We compare to the model
  variant introduced in (TODO: cite "Projections for Task-Seperation"), which does not use TTE.],
)<image_text_retrieval_tte_first>

We suspect that this is due to two reasons. First, the the TTE is added before the modality-specific encoders, so image and text encoder. Their
task is to extract features from the input, independent of the other respective modality. So TTE is of no use to them, as the same embedding
is added to every token/patch, and one does not need to differentiate between image and text in the modality-specific encoders. Second, even worse,
we use pretrained models for the encoders (Data2Vec2), which already extract rich features from the input. Adding a TTE to their input, which
is randomly initialized, will destroy the information the modality encoders receive as input, and thus the feature extraction will be less effective.
Even though we are also training the pretrained image and text encoder, it will take time until the encoders learn to adapt to the TTE, and until
the TTE has been trained in general.

We therefore opt for two changes to the TTE approach. First, we add the TTE after the modality-specific encoders, meaning the token type embedding
is added to the output of the image and text encoder. This way, the TTE will not disturb the feature extraction of the image and text encoder.
However, what will inevitably happen is that addind the TTE after the modality-specific encoders will destroy the features extracted by the aforementioned encoders, which is exactly the second problem mentioned before.

A possible solution can be found in the Transformer block of
extremely deep and large models, such as BEiT-3 @beit3. They multiply the output of the Self-Attention and Feed-Forward in a Transformer block
with a learnable scalar parameter per embedding dimension. What makes this approach special is that the scalar parameters are all initialized
with a values close to zero. As this multiplication is done before the residual connection, i.e. the addition of the input the the Self-Attention
and Feed-Forward respectively, the contribution of the Self-Attention and Feed-Forward to the output of the Transformer block is very small
at the beginning of training. What follows is that the initial input to the model is carried very far through the model, and the actual
contribution of the parameters is added as the model lears to extract meaningful features from the input @layer_scale.

#figure(
  image(
  width: 50%,
  "../figures/layer_scale_formula.png"),
  caption: [LayerScale adds a per-channel learnable weight to the output of Self-Attention and FFN. This can be modeled as a matrix
  with the diagonal being the learnable weights. With weights close to zero, the second term, i.e. the result of the feature extraction
  will have close to no contribution to the output of the Transformer block, leading the initial input to dominate during the initial
  stages of training. The input will therefore be carried very deep into the network @layer_scale.],
) <layer_scale_formula>

We will use LayerScale not for the purpose it was intended for, that is to allow training of extremely deep models @layer_scale, but to
allow the TTE to be added after the modality-specific encoders, without destroying the features extracted by the encoders.
We initialize the weights of the LayerScale to 1e-5, meaning the contribution of the TTE to the extracted features will be almost zero at the beginning. The shared Transformer block will receive the almost unaltered features, and as the scale is learned, the TTE will be added
as the models sees fit - as much as it helps the model to learn.

$
bold(H')^w_(L)&=bold(H)^w_(L) + op("diag")(lambda_(L, 1), ..., lambda_(L, D)) * bold(T)^w_"type", "or simply" \
bold(H')^w_(L)&=bold(H)^w_(L) + op("LayerScale")(bold(T)^w_"type")
$

$
bold(H')^v_(L)&=bold(H)^v_(L) + op("diag")(lambda_(L, 1), ..., lambda_(L, D)) * bold(T)^v_"type", "or simply" \
bold(H')^v_(L)&=bold(H)^v_(L) + op("LayerScale")(bold(T)^v_"type")
$

We use the same LayerScale for both image and text type embeddings, as the contribution of the type embeddings should be the same for both modalities. We do not want to bias the model towards one modality. @image_text_retrieval_tte_second shows that this approach is able to slightly improve the performance of the model, and we achieve a gain of at least 0.3% in all tasks. While this is not a significant improvement, it shows that the TTE @vlmo, together with LayerScale @layer_scale, can be beneficial, which is why we keep this approach.

#figure(
  table(
    columns: 5,
    stroke: none,
    table.hline(),
    table.header(
      [Model],
      [ImageNet-1K Top-1 Accuracy],
      [ImageNet-1K Top-5 Accuracy],
      [Retrieval MSCOCO],
      [Retrieval Flickr30K],
    ),
    table.hline(stroke: .6pt),
    [EMKUM], [26.1], [50.4], [66.3], [77.83], 
    [EMKUM#sub[TTE]], [25.6], [49.5], [66.0], [77.36],
    [EMKUM#sub[TTE']], [26.7], [50.9], [66.6], [78.09],
    table.hline(),
  ),
  caption: [Moving the TTE after the modality-specific encoders and using LayerScale with a low contribution weight of 1e-5, we denote this
  model variant as EMKUM#sub[TTE']. EMKUM#sub[TTE'] is able to achieve a slightly better performance than the model without TTE, and the version
  with TTE added before the modality-specific encoders.],
)<image_text_retrieval_tte_second>

After training, to validate whether the TTE is actually utilized, we check the average value of the LayerScale weights. If they show
a value lower or equal to the initial value of 1e-5, then we can conclude that even though the model performance improves, the relative
importance of TTE is low. We measure a mean of 7.2e-4, with a standard deviation of 1.6e-2, and we observe the maximum weight for a channel
to be 0.3. This shows that the actual features extracted by the image and text encoder are significantly more important than the TTE, which
is not surprising, but also that the model utilizes the TTE to some extent.


#bibliography("../references.bib")