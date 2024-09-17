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
this is already performed by the contrastive loss. 
Instead, we propose an approach we call feature fusion, where we perform a linear transformation of
the representation of the $mono(["T_CLS"])$ and $mono(["I_CLS"])$ token, and then simply add them together.
The linear transformation is modeled by a linear layer. We justify approach by the intuition that a linear projection of the representations
might transform them in a way that adding them together gives a meaningful representation of the image-text pair. Using the actual representations
$bold(h)''_(v, K, mono(["I_CLS"]))$ and $bold(h)''_(w, K, mono(["T_CLS"]))$ for the addition is not a good idea, as they are already
enforced to be similar under the contrastive loss. Simply adding them together does not express whether image and text match or not.
Adding a linear projection however, allows the model to keep the representations aligned, while learning a projection of the aligned representations
that is useful for ITM.

As with the contrastive loss, we apply ITM to the raw output of the intermediate and final output of the FFN from the shared Transformer block.
That also means that we have seperate projections and classification heads for both outputs:

$
bold(h)'_(v, mono(["I_ITM"])) &= bold(h)'_(v, K, mono(["I_CLS"]))bold(W)'_mono(["I_ITM"]) \
bold(h)'_(w, mono(["T_ITM"])) &= bold(h)'_(w, K, mono(["T_CLS"]))bold(W)'_mono(["T_ITM"]) \
bold(h)''_(v, mono(["I_ITM"])) &= bold(h)''_(v, K, mono(["I_CLS"]))bold(W)''_mono(["I_ITM"]) \
bold(h)''_(w, mono(["T_ITM"])) &= bold(h)''_(w, K, mono(["T_CLS"]))bold(W)''_mono(["T_ITM"]) \
$
$
bold(h)'_(mono(["IT_ITM"])) &= (bold(h)'_(v, mono(["I_ITM"]))+bold(h)'_(w, mono(["T_ITM"])))bold(W)'_"itm_cls" + bold(b)'_"itm_cls" \
bold(h)''_(mono(["IT_ITM"])) &= (bold(h)''_(v, mono(["I_ITM"]))+bold(h)''_(w, mono(["T_ITM"])))bold(W)''_"itm_cls" + bold(b)''_"itm_cls"
$ <itm_feature_fusion>

In general, $bold(h)_(mono(["IT_ITM"])) in RR^2$ denotes the logits for ITM, where $h_(mono(["IT_ITM"]), 0)$ and
$h_(mono(["IT_ITM"]), 1)$ denote the score that the image-text pair
is not matching and matching, respectively.
For clarity, we name the learnable parameters with the same primes ($'$) as the representations they are applied to.

Since at its core ITM is a binary classification task, we can use cross-entropy as the loss function:

$
cal(L)_"ITM" = 1/2*cal(L)_("CE")(bold(h)'_(mono(["IT_ITM"])), bold(y)_"ITM") + 1/2*cal(L)_("CE")(bold(h)''_(mono(["IT_ITM"])), bold(y)_"ITM") =
$ <itm_loss>

With the loss for a single component being (on the example of the output from linear \#1):

$
cal(L)_("CE")(bold(h)'_(mono(["IT_ITM"])), bold(y)_"ITM") = -sum_(i=0)^1 y_i log exp(h'_(mono(["IT_ITM"]), i))/(sum_(j=0)^1 exp(h'_(mono(["IT_ITM"]), j))), wide y_i in {0, 1}
$

The target $bold(y)_"ITM" in RR^2$ is defined such that $y_0=1$ and $y_1=0$ if the image-text pair does not match, and $y_0=0$ with $y_1=1$ if they do.
Since $h_(mono(["IT_ITM"]), 0)$ and $h_(mono(["IT_ITM"]), 1)$ are
just logits, they have to be softmax-normalized to get probabilities, hence the exponentiation
and division in the loss function. We do not use the tradiational binary cross-entropy loss, as applying softmax, followed by the logarithm,
is numerically more stable.

The loss is added as a third loss term to the total loss function, which now reads:

$
cal(L)_"S-SMKE" = cal(L)_"KD"+cal(L)_"ITC"+cal(L)_"ITM"
$

#figure(
  image("../figures/Sx3HRe_ITM.png"),
  caption: [In our approach to Image-Text Matching (ITM), the global representations of the image and text are first
  transformed through seperate learnable linear projections, and then added to form a joint representation of the image-text pair.
  This represenation is then passed to a classification head to predict whether the image-text pair matches. This is done for both the
  output of linear \#1 and linear \#2, and for both there are seperate classification and linear projection heads, leading to a total of
  six additional linear layers.
  The Knowledge Distillation loss (MSE), and the teacher, are omitted for simplicity. The image-text example is taken from the COCO train set 
  @coco.],
) <Sx3HRe_ITM>

Image-Text Matching requires both positive and negative examples, and there are two approaches to generating the latter.
The first, more intuitive approach, is to sample a random text from the current batch for a given image, and vice versa.
While straightforward, this method might make the task too easy, as negative example will usually be completely unrelated to the candidate.

To address this issue, the papers follow VLMo @vlmo and ALBEF @albef employ hard-negative mining, which involves sampling
difficult negative examples that are hard to distinguish from positive examples.
ALBEF and VLMo utilize the cosine similarities between all possible image-text pairs of the current batch, which are generated
as part of contrastive learning. They are stored in the matrix $bold(L)$, explained in @vision_language_contrast.
A negative text for a given image is generated by randomly sampling a text/caption based on the cosine similarities
between the image and all texts in the batch,
where the similarity to the actual text/caption of the image is set to 0 to avoid selecting it as a negative example.
This means that captions with a high cosine similarity to the image are more likely to be sampled as negative examples.
The same process is applied to generate negative examples for captions.

Given a batch size $B$, $B$ positive examples (the actual image-text pairs), and $2B$ negative examples are generated.
The $2B$ negative examples consists of the $B$ images in the batch paired with their hard-negative captions, and
the $B$ captions in the batch paired with their hard-negative images.

The remainder of the training setup remains unchanged, the classification heads and 
and the linear projections add around 1.1M parameters to the model, which is negligible compared
to the total number of trainable parameters, which now stands at 117 million.

We compare both ITM with random negatives as well as ITM with hard-negative mining with our previous baselines.

Plotting the training accuracy (@itm_train_acc) shows a continuous score of 66.67% for both approaches,
which remains constant throughout the first epoch with no deviation from this value. This suggests
that our approach is not effective. We suspect this outcome arises because the classification head
fails to learn anything meaningful from the fused representations, leading it to predict the most frequent
class in the batch to somehow minimize the loss.

In this setting, the most frequent class corresponds to the scenario where the image-text pair does not match, as each batch contains $2B$ negative examples and only $B$ positive examples. Consequently, 66.67% of the fused representations in a batch are negative examples. If the classification head predicts that all examples are negative, it will achieve an accuracy of exactly 66.67%.

#figure(
  image(
  width: 50%,
  "../figures/itm_train_acc.png"),
  caption: [
    Training accuracy of the student model for Image-Text Matching (ITM) in the first epoch. The accuracy does not change from the initial value of 66.67%, and always predicts that image-text pairs do not match.
  ],
) <itm_train_acc>

As this result gives no reason to continue the training, we stop it after the first epoch.
We suspect that ITM might not be a suitable task for our model, as the representations of the image and text are separate. Generating a meaningful
joint representation of an image-text pair is only possible if the tokens of the image and text can attend to each other using Self-Attention.
This so-called cross-modal attention is given in models like VLMo @vlmo, METER @meter, and ALBEF @albef, but not in our model.

Although, as mentioned before, a logical step would be to simply compute the cosine similarity between an image-text pair, and using the score as 
an indicator of matching or non-matching, as this is already performed in the contrastive loss. Contrastive learning directly optimizes
for similarity between image and text embeddings by maximizing the cosine similarity for positive pairs and minimizing it for negatives. Adding an
cosine similarity step for ITM would only duplicate this process,
without introducing any new information on which the model can improve.