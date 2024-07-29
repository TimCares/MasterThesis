#set math.equation(numbering: "(1)")
$
bold(H)^w_(L-F)&=op("Encoder")_w (bold(H)^w_0) \
bold(H)^v_(L-F)&=op("Encoder")_v (bold(H)^v_0) \
bold(H)^w_(L)&=op("Encoder")_s (bold(H)^w_(L-F)) \
bold(H)^v_(L)&=op("Encoder")_s (bold(H)^v_(L-F)) \
bold(H)^w_(l)&=[bold(w)_l^mono(["T_CLS"]), bold(w)_l^1, ..., bold(w)_l^M, bold(w)_l^mono(["T_SEP"])]\
bold(H)^v_(l)&=[bold(v)_l^mono(["I_CLS"]), bold(v)_l^1, ..., bold(v)_l^N]\
$
- with $l in {1, ..., L-F, ..., L}$

- we define $bold(H)^w_(L)$ as the final output of the student model for the caption, and $bold(H)^v_(L)$ as the final output of the student model for the image, with $bold(H)^w_(L) in RR^((M+2) times D)$ and $bold(H)^v_(L) in RR^((N+1) times D)$

==== Image-Text Matching with Feature Fusion <image_text_matching_with_feature_fusion>

Up to this point, we only adapted the Image-Text Contrast (ITC) from VLMo to our model. Notably, we did not employ Masked Language Modeling (MLM) because we leverage Knowledge Distillation to learn the features of the teacher model, which are the representations the model returns. We also avoided using Image-Text Matching (ITM) due to the nature of the CLS token in the shared Transformer block of our adaptation, which represents image and text independently (as in TODO: cite SHRe), rather than an image-text pair (image and text together). This configuration prevents us from directly passing our representation to a classification head for ITM, since a combination of image and text is necessary to predict a match.

In VLMo, as described in @vlmo_out, the final output demonstrates that text tokens and image patches are concatenated, allowing text tokens to attend to image patches through Self-Attention (TODO: Cite transformer paper), and vice versa. Therefore, the global image-text representation, $bold(w)_L^mono(["T_CLS"])$, contains information from both the image and text. Consequently, a classification head can utilize this representation to infer if an image-text pair matches.


$
bold(H)^(w v)_(L)&=[bold(w)_L^mono(["T_CLS"]), bold(w)_L^1, ..., bold(w)_L^M, bold(w)_L^mono(["T_SEP"]), bold(v)_L^mono(["I_CLS"]), bold(v)_L^1, ..., bold(v)_L^N]
$ <vlmo_out>

In our approach however, $bold(w)_L^mono(["T_CLS"])$ and $bold(v)_L^mono(["I_CLS"])$ contain only the information of the image and text, respectively (see @sx3hre_out).

$
bold(H)^w_(L)&=[bold(w)_L^mono(["T_CLS"]), bold(w)_L^1, ..., bold(w)_L^M, bold(w)_L^mono(["T_SEP"])], bold(H)^v_(L)=[bold(v)_L^mono(["I_CLS"]), bold(v)_L^1, ..., bold(v)_L^N]\
$ <sx3hre_out>

Although it is possible to compute the cosine similarity between the global text representation $bold(w)_L^mono(["T_CLS"])$ 
and the image representation $bold(v)_L^mono(["I_CLS"])$, this is already performed by the contrastive loss (explained in TODO: cite cotrastive loss section). 
Instead, we propose an approach we call feature fusion. Inspired by the method of concatenating image and text timesteps (as implemented in VLMo,
BEiT-3, and FLAVA) to generate a representation of an image-text pair, we concatenate the global representations of the image
and text to form an image-text representation. This combined representation is then passed to a classification head
to predict whether the image-text pair matches.

$
bold(u)&=bold(w)_L^mono(["I_CLS"]) || bold(v)_L^mono(["T_CLS"]) \
bold(p)&=[p_0, p_1]=op("Head")_op("ITM")(bold(u))
$ <itm_feature_fusion>

Here, $bold(w)_L^mono(["T_CLS"])$ and $bold(v)_L^mono(["I_CLS"])$ together form the input $bold(u)$ to the image-text matching head $op("Head")_op("ITM")$, where $bold(u) in RR^(2 times D)$. The output $bold(p) in RR^2$ contains the logits indicating whether the image and text match ($p_1$) or not ($p_0$), they are normalized using softmax to obtain probabilities before computing the loss. 

Because the contrastive loss is applied on the $mono(["I_CLS"])$/$mono(["T_CLS"])$ token (depending which modality was the input) of both the first and last MLP layer of the shared Transformer block, we also introduce ITM for both layers. This means that we have two classification heads, $op("Interm-Head")_op("ITM")$ and $op("Head")_op("ITM")$, one for the first and one for the last layer, respectively.

#figure(
  image("../figures/Sx3HRe_ITM.png"),
  caption: [Image-Text Matching (ITM) with Feature Fusion in Sx3HRe. The global representations of the image and text are concatenated to form a representation of the image-text pair, which is then passed to a classification head to predict whether the image-text pair matches. The figure shows
  one classification head to keep the figure concise. The head is either $op("Interm-Head")_op("ITM")$, for outputs of the first linear layer (Linear \#1), or $op("Head")_op("ITM")$, for outputs of the second linear layer (Linear \#2).
  The Knowledge Distillation loss (MSE), and the teacher, are omitted for simplicity. The image-text example is taken from the COCO train set (TODO: cite coco).],
) <Sx3HRe_ITM>


Before applying binary cross-entropy (@bin_ce) to calculate the ITM loss $cal(L)_"ITM"$ (@itm_loss), $bold(p)$ is softmax-normalized. The labels are defined such that $y_0=1$ and $y_1=0$ if the image-text pair does not match, and $y_0=0$ with $y_1=1$ if they do. Since ITM is applied to two layers simultaneously, the total ITM loss is the mean of the losses of both layers.

$
op("CE")(bold(hat(y)), bold(y))=-sum_(i=0)^1 y_i log(hat(y)_i), y_i in {0, 1}
$ <bin_ce>

$
cal(L)_"ITM"= 1/2 * (op("CE")(bold(p), bold(y)) + op("CE")(bold(p), bold(y)))
$ <itm_loss>

In this initial experiment, $op("Head")_op("ITM")$ and $op("Interm-Head")_op("ITM")$ are single fully connected layers. The resulting loss for the student model is similar to that of VLMo and ALBEF:

$
cal(L)_"VLMo" = cal(L)_"MLM"+cal(L)_"ITC"+cal(L)_"ITM"
$

The difference lies in the MLM loss, which is replaced by the Knowledge Distillation loss $cal(L)_"KD"$:

$
cal(L)_"Sx3HRe" = cal(L)_"KD"+cal(L)_"ITC"+cal(L)_"ITM"
$

Image-Text Matching requires both positive and negative examples. An intuitive approach to generating negative examples is to sample a random text for a given image from the current batch, and vice versa. However, this method might make the task too easy, as negative example will be completely unrelated to the image in most cases.

To address this issue, we follow VLMo and ALBEF and employ hard-negative mining, which involves sampling difficult negative examples that are hard to distinguish from positive examples. ALBEF and VLMo utilize the cosine similarities of the current batch from contrastive learning to achieve this. A negative image-text pair for a given image is generated by sampling a text/caption using a multinomial distribution, where the cosine similarities serve as probabilities, and the actual text/caption of the image has a probability of 0.
This means that captions with a high cosine similarity to the image are more likely to be sampled as negative examples. The same process is applied to generate negative examples for captions.

Given a batch size $N$, $N$ positive examples (the actual image-text pairs), and $2N$ negative examples are generated:

- Fuse representations of actual image-text pairs to obtain $N$ positive examples.
- Fuse representations of each image in the batch with its hard-negative caption to get $N$ negative examples.
- Fuse representations of each caption in the batch with its hard-negative image to produce another $N$ negative examples.

In all cases, the joined representation $bold(u)$ (as described in @itm_feature_fusion) begins with the image representation, followed by the text representation. 

The remainder of the training setup remains unchanged, the classification heads adds just 6,148 parameters (3,074 each) to the model, which is negligible compared to the total number of trainable parameters, which now stands at 132 million.

The results (@itm_train_acc_initial) show a continuous training accuracy of 66.67%, which remains constant throughout the first epoch with no deviation from this value. This suggests that our approach is not effective. We suspect this outcome arises because the classification head fails to learn anything meaningful from the fused representations, leading it to predict the most frequent class in the batch to somehow minimize the loss.

In this setting, the most frequent class corresponds to the scenario where the image-text pair does not match, as each batch contains $2N$ negative examples and only $N$ positive examples. Consequently, 66.67% of the fused representations in a batch are negative examples. If the classification head predicts that all examples are negative, it will achieve an accuracy of exactly 66.67%.

#figure(
  image(
  width: 50%,
  "../figures/itm_train_acc_initial.png"),
  caption: [
    Training accuracy of the student model for Image-Text Matching (ITM) in the first epoch. The accuracy does not change from the initial value of 66.67%, and always predicts that image-text pairs do not match.
  ],
) <itm_train_acc_initial>

As this result gives no reason to continue the training, we stop it after the first epoch.

We hypothesize that a single linear layer may be insufficient for learning the matching between image and text representations. Our setting differs from that in VLMo, where the CLS token's representation already encapsulates information from both the image and text. In our case, we fuse separate embeddings, and the resulting representation by itself lacks semantic meaning. Consequently, a simple classification head is unable to extract any meaningful information from this representation.

To address this, we replace the simple classification heads with feed-forward networks (FFN) of the same architecture as those used in Transformer blocks, i.e. two linear layers. The first layer expands the dimensionality of the fused representation from a 1536-dimensional vector to a 6144-dimensional vector. The second layer reduces the dimensionality to 2, corresponding to the number of classes. This modification adds 6 million parameters to the model, which we deem acceptable if the results show a significant improvement.

As shown in @itm_train_acc_ffn, the ITM heads now appear to learn from the fused representations. Unfortunately, this fact is neither reflected in the zero-shot performance on ImageNet-1K, nor in the retrieval performance on MSCOCO and Flickr30K, illustrated in @itm_results.

#figure(
  image(
  width: 50%,
  "../figures/itm_train_acc_initial.png"),
  caption: [
    Training accuracy of the student model for Image-Text Matching (ITM) in the first epoch. The accuracy does not change from the initial value of 66.67%, and always predicts that image-text pairs do not match.
  ],
) <itm_train_acc_ffn>

#figure(
  table(
    columns: 5,
    stroke: none,
    table.hline(),
    table.header(
      [Model],
      [ImageNet-1k Accuracy],
      [Retrieval MSCOCO],
      [Retrieval Flickr30K],
      [\#Params],
    ),
    table.hline(stroke: .6pt),
    [CLIP (TODO: cite)], [*72.6*], [66.73], [*90.1*], [428M],
    [*Sx3HRe#sub[MSE Head]*], [26.1], [*66.3*], [42.4], [*132M*],
    [*Sx3HRe#sub[MSE Head + ITM]*], [25.5], [65.7], [41.6], [138M],
    table.hline(),
  ),
  caption: [Comparison of Sx3HRe with other approaches when adding ITM. The results show that our fusion-based approach does not improve the performance of the model. ImageNet-1k accuracy is based on CLIPs zero-shot transfer, while retrieval performance is the mean of R@1, R@5, and R@10 on the respective datasets.],
)<itm_results>