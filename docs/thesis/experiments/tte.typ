=== Token-Type Embeddings <token_type_embeddings>
In order for the shared Transformer approach to work, the linear layers and self-attention mechanism in the
shared Transformer layer need to be able to somewhat differentiate between both
modalities (image and text). Even though we desire an aligned representation to form in the shared block, especially the self-attention mechanism
still needs to be able to differentiate between the two modalities. This can be explained by the fact that for images the model needs to
find 2-dimensional spatial relationships, while for text only 1-dimensional relationships are required. Even though we learned from
the previous experiments
that the shared Transformer block is able to learn a good representation without any modality-specific information, we still want to investigate
how the performance of the model changes when we explicitly provide the shared Transformer layer with the information which modality it
is currently processing. This necessitates a special embedding for both image and text, which is added to each token of the respective modality, 
even the special tokens. This is called a token-type embedding (TTE) @vlmo, and is also used in multimodal models such as VLMo @vlmo.

The intuitive approach to implement this would be to directly follow VLMo and add the token type embeddings after the positional
encoding, before the input is fed into the Transformer blocks @vlmo. Our definition of the Transformer input changes as follows for text:

$
bold(H)_(w, 0) &= [bold(h)_(w, 0, mono(["T_CLS"])), bold(h)_(w, 0, 1), ..., bold(h)_(w, 0, M), bold(h)_(w, 0, mono(["T_SEP"]))]
= bold(E)_w + bold(T)^"pos"_w + bold(T)^("type")_w \
bold(T)^("type")_w &= [bold(t)^w_"type", bold(t)^w_"type", ..., bold(t)^w_"type", bold(t)^w_"type"]
$ <text_representation_tte>

Similar holds for images:

$
bold(H)_(v, 0) &= [bold(h)_(v, 0, mono(["I_CLS"])), bold(h)_(v, 0, 1), ..., bold(h)_(v, 0, N)]
= bold(E)_v + bold(T)^"pos"_v + bold(T)^("type")_v \
bold(T)^("type")_v &= [bold(t)^v_"type", bold(t)^v_"type", ..., bold(t)^v_"type"]
$ <image_representation_tte>

The token type embedding is the same for every token in the sequence.
This follows VLMo @vlmo and 
is because each token/patch of a text/image input sequence is of the same modality.
The parameters added to the model are negligible, as they
only include two additional embeddings of size $D$, with $D=768$ this accounts for just $768*2=1536$ parameters.

We present the results in @image_text_retrieval_tte_first, and show that the variant with TTE does not improve the performance of the model.

#figure(
  table(
    columns: 4,
    stroke: none,
    table.hline(),
    table.header(
      table.cell(rowspan: 2, colspan: 1, align:horizon, [TTE]),
      table.cell(rowspan: 2, colspan: 1, align:horizon, [ImageNet-1K]),
      table.cell(colspan: 2, align:horizon, [Retrieval]),
      [MSCOCO],
      [Flickr30K],
    ),
    table.hline(stroke: .6pt),
    [$times$], [*30.4*], [*66.87*], [*77.1*],
    [$checkmark$], [29.6], [66.0], [76.5],
    table.hline(),
  ),
  caption: [Introducing token-type embeddings (TTE) after the positional encoding degrades performance slightly. We compare to
  the previous model, which does not use TTE.],
)<image_text_retrieval_tte_first>

We suspect that this is due to two reasons. First, the TTE is added before the modality-specific encoders, so the
image and text encoder. Their
task is to extract features from the input, independent of the *other* respective modality. Consequently,
the TTE is of no use to them, as the same embedding
is added to every token/patch, and both encoders do not need to differentiate between image and text
in their input: The image encoder will always receive an image, and the text encoder always a text. Second, even worse,
we use pretrained layers for the image and text encoder, which already extract rich features from the input. Adding the token-type
embedding, which is initialized randomly, to their input
will destroy the patch (token) embeddings the image (text) encoder receives as its input,
and thus the feature extraction will be less effective.
Even though we are also training the image and text encoder it will take time until the encoders learn to adapt to the TTE, and until
the TTE has been trained in itself.

We therefore opt for the following change: We add the TTE after the modality-specific encoders, meaning the token type embedding
is added to the output of the image and text encoder. This way, the TTE will not disturb the feature extraction of the image and text encoder.
However, what will inevitably happen is that adding the TTE after the modality-specific encoders will destroy the features extracted by the
encoders, which is exactly the second problem mentioned before, just at a different stage in the model.

A possible solution can be found in the Transformer block of
extremely deep and large models, such as BEiT-3 @beit3. Here, each embedding dimension of the output
generated by the self-attention and MLP in a Transformer layer
is multiplied with a separate, learnable, scalar.
What makes this approach special is that the scalars are all initialized
with values close to zero. As this multiplication is done before the residual connection, i.e. the
addition of the input to the output of self-attention
and MLP respectively, the contribution of the self-attention and MLP to the output of the Transformer block is very small
at the beginning of training. What follows is that the initial input to the model is carried very far through the model, and the actual
contribution of the self-attention and other parameters is added gradually as the model learns
to extract meaningful features from the input @layer_scale.

We will use LayerScale not for the purpose it was intended for, that is, to allow training of extremely deep models @layer_scale, but to
allow the TTE to be added gradually after the modality-specific encoders, without destroying the features extracted by the encoders.
Before the TTE is added to the output sequence of the image and text encoder, we multiply each dimension of the TTE with its own learnable
scalar, as given by LayerScale.
We initialize the weights of the LayerScale to 1e-5, so that each dimension of the TTE is multiplied with a very small scalar, making the
contribution of the TTE, when adding it to the output of the image and text encoder, very small initially. This way,
the shared Transformer block will receive the almost unaltered features from the encoders, and since the scale is learned, the TTE will be added
as the models sees fit - as much as it helps the model to learn. Implementing LayerScale is straightforward, the representation
of each TTE dimension is multiplied element-wise ($dot.circle$) with the LayerScale embedding
$bold(t)_"scale"$ before the TTE is added to the output of the image and text encoder:

$
bold(H)'_(w, L_s) = bold(H)_(w, L_s) + bold(t)_"scale" * bold(T)^w_"type" = \
bold(H)_(w, L_s) + [bold(t)_"scale" dot.circle bold(t)^w_"type",
bold(t)_"scale" dot.circle bold(t)^w_"type", ..., bold(t)_"scale" dot.circle bold(t)^w_"type",
bold(t)_"scale" dot.circle bold(t)^w_"type"]
$

$
bold(H)'_(v, L_s) = bold(H)_(v, L_s) + bold(t)_"scale" * bold(T)^v_"type" = \
bold(H)_(v, L_s) + [bold(t)_"scale" dot.circle bold(t)^v_"type",
bold(t)_"scale" dot.circle bold(t)^v_"type", ..., bold(t)_"scale" dot.circle bold(t)^v_"type"]
$

We use the same LayerScale $bold(t)_"scale" in RR^768$ for both image and text type embeddings,
as the contribution of the type embeddings should be the same for both modalities.
We do not want to bias the model towards one modality.

@image_text_retrieval_tte_second shows that
this approach is able to slightly improve the performance of the model, and we achieve a gain of at
least 0.3% in all tasks. While this is not a significant improvement, it shows that the TTE @vlmo,
together with LayerScale @layer_scale, can be beneficial, which is why we keep this approach.

#figure(
  table(
    columns: 5,
    stroke: none,
    table.hline(),
    table.header(
      table.cell(rowspan: 2, colspan: 1, align:horizon, [TTE]),
      table.cell(rowspan: 2, colspan: 1, align:horizon, [LayerScale]),
      table.cell(rowspan: 2, colspan: 1, align:horizon, [ImageNet-1K]),
      table.cell(colspan: 2, align:horizon, [Retrieval]),
      [MSCOCO],
      [Flickr30K],
    ),
    table.hline(stroke: .6pt),
    [$times$], [$times$], [*30.4*], [66.87], [77.1],
    [$checkmark$],[$times$], [30.3], [*67.2*], [78.08],
    [$checkmark$],[$checkmark$], [30.3], [*67.2*], [*78.29*],
    table.hline(),
  ),
  caption: [
    Comparison of S-SMKE with different TTE and LayerScale settings *after* the modality-specific encoders. We observe that
    adding TTE and LayerScale especially improves performance on the retrieval tasks, with Flick30K showing an absolute gain of 1.19%.
  ],
)<image_text_retrieval_tte_second>

After training, to validate whether the TTE is actually utilized, we check the average value of the LayerScale weights. If they show
a value lower or equal to the initial value of 1e-5, then we can conclude that even though the model performance improves, the relative
importance of TTE is low. We measure a mean of 2e-4 and observe the maximum weight for a channel
to be 0.37. This shows that the model utilizes the TTE to some extend,
and we keep the TTE and LayerScale in the model, as they increase the performance slightly.
Only few parameters are added: 1,536 for both token type embeddings, and 768 for the learnable scaling weights of LayerNorm.

