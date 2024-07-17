#set math.equation(numbering: "(1)")
$
bold(H)^w_(L-F)&=op("Encoder")_w (bold(H)^w_0) \
bold(H)^v_(L-F)&=op("Encoder")_v (bold(H)^v_0) \
bold(H)^w_(L)&=op("Encoder")_s (bold(H)^w_(L-F)) \
bold(H)^v_(L)&=op("Encoder")_s (bold(H)^v_(L-F)) \
bold(H)^w_(l)&=[bold(w)_l^[T\_C L S], bold(w)_l^1, ..., bold(w)_l^M, bold(w)_l^[T\_S E P]]\
bold(H)^v_(l)&=[bold(v)_l^[I\_C L S], bold(v)_l^1, ..., bold(v)_l^N]\
$
- with $l in {1, ..., L-F, ..., L}$

- we define $bold(H)^w_(L)$ as the final output of the student model for the caption, and $bold(H)^v_(L)$ as the final output of the student model for the image, with $bold(H)^w_(L) in RR^((M+2) times D)$ and $bold(H)^v_(L) in RR^((N+1) times D)$

==== Image-Text Matching with Feature Fusion <image_text_matching_with_feature_fusion>

Up to this point, we only adapted the Image-Text Contrast (ITC) from VLMo to our model. Notably, we did not employ Masked Language Modeling (MLM) because we leverage Knowledge Distillation to learn the features of the teacher model, which are the representations the model returns. We also avoided using Image-Text Matching (ITM) due to the nature of the CLS token in the shared Transformer block of our adaptation, which represents image and text independently (as in TODO: cite SHRe), rather than an image-text pair (image and text together). This configuration prevents us from directly passing our representation to a classification head for ITM, since a combination of image and text is necessary to predict a match.

In VLMo, as described in @vlmo_out, the final output demonstrates that text tokens and image patches are concatenated, allowing text tokens to attend to image patches through Self-Attention (TODO: Cite transformer paper), and vice versa. Therefore, the global image-text representation, $bold(w)_L^[T_C L S]$, contains information from both the image and text. Consequently, a classification head can utilize this representation to infer if an image-text pair matches.


$
bold(H)^(w v)_(L)&=[bold(w)_L^[T\_C L S], bold(w)_L^1, ..., bold(w)_L^M, bold(w)_L^[T\_S E P], bold(v)_L^[I\_C L S], bold(v)_L^1, ..., bold(v)_L^N]
$ <vlmo_out>

In our approach however, $bold(w)_L^[T\_C L S]$ and $bold(v)_L^[I\_C L S]$ contain only the information of the image and text, respectively (see @sx3hre_out).

$
bold(H)^w_(L)&=[bold(w)_L^[T\_C L S], bold(w)_L^1, ..., bold(w)_L^M, bold(w)_L^[T\_S E P]], bold(H)^v_(L)=[bold(v)_L^[I\_C L S], bold(v)_L^1, ..., bold(v)_L^N]\
$ <sx3hre_out>

Although it is possible to compute the cosine similarity between the global text representation $bold(w)_L^[T\_C L S]$ 
and the image representation $bold(v)_L^[I\_C L S]$, this is already performed by the contrastive loss (explained in TODO: cite cotrastive loss section). 
Instead, we propose an approach we call feature fusion. Inspired by the method of concatenating image and text timesteps (as implemented in VLMo,
BEiT-3, and FLAVA) to generate a representation of an image-text pair, we concatenate the global representations of the image
and text to form an image-text representation. This combined representation is then passed to a classification head
to predict whether the image-text pair matches.

$
bold(u)&=[bold(w)_L^[I\_C L S];bold(v)_L^[T\_C L S]]\
bold(p)&=[p_0, p_1]=op("Head")_op("ITM")(bold(u))
$ <itm_feature_fusion>

Here, $bold(w)_L^[T\_C L S]$ and $bold(v)_L^[I\_C L S]$ together form the input $bold(u)$ to the image-text matching head $op("Head")_op("ITM")$, where $bold(u) in RR^(2 times D)$. The output, $bold(p) in RR^2$ contains the logits indicating whether the image and text match ($p_1$) or not ($p_0$). 

Because the contrastive loss is applied on the [I_CLS]/[T_CLS] (depending which modality was the input) token of both the first and last MLP layer of the shared Transformer block, we also introduce ITM for both layers. This means that we have two classification heads, $op("Interm-Head")_op("ITM")$ and $op("Head")_op("ITM")$, one for the first and one for the last layer, respectively. The final loss for ITM is the mean of the losses of both layers.

Before applying cross-entropy to calculate the ITM loss $cal(L)_"ITM"$ (described in @itm_loss), $bold(p)$ is softmax-normalized. The labels are defined such that $y_0=1$ and $y_1=0$ if the image-text pair does not match, and $y_0=0$ with $y_1=1$ if they do.

$
cal(L)_"ITM"=op("CE")(bold(p), bold(y))=-sum_(i=0)^1 y_i log(p_i), y_i in {0, 1}
$ <itm_loss>

In this initial experiment, $op("Head")_op("ITM")$ is a single fully connected layer. The resulting loss for the student model is similar to that of VLMo and ALBEF:

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

The remainder of the training setup remains unchanged, the classification head $op("Head")_op("ITM")$ adds just 3,074 parameters to the model, which is negligible compared to the total number of trainable parameters, which now stands at 132 million.

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

To address this, we replace the simple classification head with a feed-forward network (FFN) of the same architecture as those used in Transformer blocks, i.e. two linear layers. The first layer expands the dimensionality of the fused representation from a 1536-dimensional vector to a 6144-dimensional vector. The second layer reduces the dimensionality to 2, corresponding to the number of classes. Since this is no longer a simple classification head, we rename it from $op("Head")_op("ITM")$ to $op("FFN")_op("ITM")$. This modification adds 6 million parameters to the model, which we deemed acceptable if the results show a significant improvement.
