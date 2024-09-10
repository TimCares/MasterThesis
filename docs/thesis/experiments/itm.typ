==== Image-Text Matching with Feature Fusion

Based on the success we experienced so far with applying contrastive learning to learn image-text representations,
we now turn our attention to Image-Text Matching (ITM). Used as a training objective by VLMo @vlmo and METER @meter,
this tasks involves predicting whether an image and text pair match. 
In both papers, the implementation is straightforward: Since the Self-Attention mechanism in both works processes image and text jointly, 
i.e. the text and image tokens are concatenated (see @vl_representation_to_remember),
allowing the text tokens to attend to image patches and vice versa, the representation
of the $mono(["T_CLS"])$ can be used as the representation of the image-text pair. This representation is then passed into a linear
layer, which is the classification head, to predict whether the image-text pair matches. In practice, this rather simple approach
actually leads to the best results in both papers.

$
bold(H)_(v w, l)=[bold(h)_(w, l, mono(["T_CLS"])), bold(h)_(w, l, 1), ..., bold(h)_(w, l, M), bold(h)_(w, l, mono(["T_SEP"])), 
bold(h)_(v, l, mono(["I_CLS"])), bold(h)_(v, l, 1), ..., bold(h)_(v, l, N)]
$ <vl_representation_to_remember>

The approach of using the $mono(["T_CLS"])$ token as the image-text representation in ITM stands in contrast to our approach, where
the image and text representations are separate, and the $mono(["T_CLS"])$ token only contains information about the text, and the $mono(["I_CLS"])$
token only about the image. If we want to apply ITM to our model, we need to find a way to combine the image and text representations.

An intuitive way to combine, or rather compare, the image and text representations is to compute the cosine similarity between the global text
representation $bold(h)''_(w, K, mono(["T_CLS"]))$ and the image representation $bold(h)''_(v, K, mono(["I_CLS"]))$. However,
this is already performed by the contrastive loss (explained in @vision_language_contrast). 
Instead, we propose an approach we call feature fusion, where we perform a linear transformation of
the representation of the $mono(["T_CLS"])$ and $mono(["I_CLS"])$ token, and then simply add them together.
The linear transformation is modeled by a linear layer. We justify approach by the intuition that a linear projection of the representations
might transform them in a way that adding them together gives a meaningful representation of the image-text pair. Using the actual representations
$bold(h)''_(v, K, mono(["I_CLS"]))$ and $bold(h)''_(w, K, mono(["T_CLS"]))$ for the addition is not a good idea, as they are already
enforced to be similar under the contrastive loss. Adding them together and using the result for ITM could work agaist the alignment
of the representations, which would lead to something similar as the loss-equlibrium described in @loss_equilibrium_harms_alignment.
Adding a linear projection however, allows the model to keep the representations aligned, while learning a projection of the aligned representations
that is useful for ITM.


$
bold(h)_(v, mono(["I_ITM"])) &= bold(h)''_(v, K, mono(["I_CLS"]))bold(W)_"ITM" \
bold(h)_(w, mono(["T_ITM"])) &= bold(h)''_(w, K, mono(["T_CLS"]))bold(W)_"ITM" \
bold(hat(y))_"ITM" &= [hat(y)_0, hat(y)_1] = (bold(h)_(v, mono(["I_ITM"]))+bold(h)_(w, mono(["T_ITM"])))bold(W)_"ITM_CLS" + bold(b)_"ITM_CLS"
$ <itm_feature_fusion>

$bold(hat(y))_"ITM" in RR^2$ denotes the logits for ITM, where $hat(y)_0$ and $hat(y)_1$ denote the score that the image-text pair
is not matching and matching, respectively.

Because the contrastive loss is applied on the $mono(["I_CLS"])$/$mono(["T_CLS"])$ token (depending which modality was the input) of both the first and last MLP layer of the shared Transformer block, we also introduce ITM for both layers. This means that we have two classification heads, $op("Interm-Head")_op("ITM")$ and $op("Head")_op("ITM")$, one for the first and one for the last layer, respectively. Since at its core this is a binary classification task, we can use cross-entropy as the loss function:

$
cal(L)_"ITM" = cal(L)_("CE")(bold(hat(y))_"ITM", bold(y)_"ITM")=-sum_(i=0)^1 y_i log exp(hat(y)_i)/(sum_(j=0)^1 exp(hat(y)_j)), wide y_i in {0, 1}
$ <itm_loss>

The target $bold(y)_"ITM" in RR^2$ is defined such that $y_0=1$ and $y_1=0$ if the image-text pair does not match, and $y_0=0$ with $y_1=1$ if they do.
Since $hat(y)_0$ and $hat(y)_1$ are just logits, they have to be softmax-normalized to get probabilities, which is why we use exponentiation
and division in the loss function.

The loss is added as a third loss term to the total loss function, which now reads:

$
cal(L)_"S-SMKE" = cal(L)_"KD"+cal(L)_"ITC"+cal(L)_"ITM"
$

#figure(
  image("../figures/Sx3HRe_ITM.png"),
  caption: [Image-Text Matching (ITM) with Feature Fusion in Sx3HRe. The global representations of the image and text are concatenated to form a representation of the image-text pair, which is then passed to a classification head to predict whether the image-text pair matches. The figure shows
  one classification head to keep the figure concise. The head is either $op("Interm-Head")_op("ITM")$, for outputs of the first linear layer (Linear \#1), or $op("Head")_op("ITM")$, for outputs of the second linear layer (Linear \#2).
  The Knowledge Distillation loss (MSE), and the teacher, are omitted for simplicity. The image-text example is taken from the COCO train set (TODO: cite coco).],
) <Sx3HRe_ITM>

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
    columns: 4,
    stroke: none,
    table.hline(),
    table.header(
      table.cell(rowspan: 2, colspan: 1, align:horizon, [ITM]),
      table.cell(rowspan: 2, colspan: 1, align:horizon, [ImageNet-1K]),
      table.cell(colspan: 2, align:horizon, [Retrieval]),
      [MSCOCO],
      [Flickr30K],
    ),
    table.hline(stroke: .6pt),
    [$times$], [30.3], [*67.2*], [*78.29*],
    [$times$], [30.3], [*67.2*], [*78.29*],
    table.hline(),
  ),
  caption: [
    
  ],
)<itm_results>