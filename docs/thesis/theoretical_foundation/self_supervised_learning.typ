#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
== Self-Supervised Learning <self_supervised_learning>

Supervised models, while powerful, are not inherently scalable. Although their architecture can be
extended to create larger models that achieve better performance, these larger models require more data for training.
In the context of supervised learning, this data must be labeled, which presents a significant challenge.
Labeled data is scarce and expensive to obtain, as it requires human annotation, thereby limiting the scalability of supervised models.

The primary objective of self-supervised learning is to learn representations of data without relying on human-annotated labels.
However, self-supervised learning is not unsupervised learning.
Unsupervised learning operates without any form of supervision, meaning that no labels are required at all,
as seen in clustering methods like K-means. In contrast, self-supervised learning requires, as supervised learning,
labeled data, but in contrast to supervised learning, labels are generated directly from the data itself.

A prominent example of self-supervised learning is Masked Language Modeling (MLM) in Natural Language Processing (NLP),
which is used in the popular NLP model BERT, the latter being one of the first models trained using self-supervised
methods to achieve state-of-the-art performance in NLP @bert.
In BERT, certain tokens, or words, are masked, i.e., removed, from a sentence, and the model is tasked with predicting the masked tokens.
Since the labels are derived from the data itself — the words to predict are part of the original data — no human annotation is needed @bert.
This allows for the utilization of large amounts of unlabeled data, as any text data can be used.

What makes self-supervised learning particularly powerful is its applicability to any type of data with a hierarchical structure, such as text,
images, audio, or video. In these cases, part of the data can be masked, and the model must predict the masked part based on the context
provided by the remaining data. An intuitive example, presented by Yann LeCun and Ishan Misra of Meta, illustrates why this approach is effective.
Consider the sentence “The lions chase the wildebeests in the savanna.” If “lions” and “wildebeests” are masked, the input becomes
“The [MASK] chases the [MASK] in the savanna.” To successfully predict the masked words, the model must understand the real-world
concepts expressed by the sentence. While “The cat chases the mouse in the savanna” might be a valid prediction in the context of
“chase,” the word “savanna” provides additional context, as it is not a typical habitat for cats and mice, but rather for lions and wildebeests.
Thus, the model must understand that lions and wildebeests are animals that inhabit savannas, in order to make a correct prediction.
Through this process of predicting masked words, the model learns about the concepts of the world we live in @self_supervised_learning_dark_matter.

As mentioned before, this approach can be transferred to other modalities, such as images, where parts of the image can be masked,
and the model must predict the masked parts based on the visible parts of the image. This is known as Masked Image Modeling (MIM).



#figure(
  image(
  width: 50%,
  "../figures/self_supervised_learning_dark_matter.png"),
  caption: [In Self-Supervised Learning parts of the data are masked (grey), and the task of a model is to predict the
  masked parts using the visible data (green) @self_supervised_learning_dark_matter.],
) <self_supervised_learning_dark_matter_fig>

=== Masked Language Modeling (MLM) <masked_language_modeling>

As introduced in @self_supervised_learning, masked language modeling (MLM) operates by masking tokens in a sentence
and predicting the masked tokens based on the context provided by the remaining tokens.
For each masked token, the model predicts a probability distribution over the vocabulary, representing all possible tokens.
The token with the highest probability is considered the predicted token.
Fundamentally, MLM is a classification task where the model classifies each masked token into one of the tokens in the vocabulary.

Given a text model $f$ and a text representation input $bold(H)_(w, 0)$, where a fraction $p in (0, 1)$ of tokens $cal(M)$
(not to be confused with $M$, denoting the number of text tokens) are masked, typically between 15% and 50% of tokens as per @bert
and @beit3, the set of masked tokens is defined as:

$
cal(M) = {i | i tilde op("Uniform")({1, ..., M})} in NN^floor(M*p)
$

The process begins by passing the text representation through the model $f$:

$
bold(H)_(w, L) = f(bold(H)_(w, 0))
$

The representations for the masked tokens are then gathered:

$
bold(H')_(w, L) = {bold(h)_(w, l, j) | j in cal(M)}
$

These representations are passed through the classification head $op("g")$, which returns logits for each token in the vocabulary:

$
bold(hat(H))_(w, L) = op("g")(bold(H')_(w, L)) = {op("g")(bold(h')_(w, L, j)) | j in cal(M)}
$

The model is trained to minimize the cross-entropy loss of the correct token $i$ given the output representation of the corresponding
masked token $bold(hat(h))_(w, L, j)$. The MLM loss for an input text is the sum of the cross-entropy losses for the masked tokens:

$
cal(L)_"MLM" = sum_(j in cal(M)) -log op("p")(i | bold(hat(h))_(w, L, j))
$

Here, $op("p")(i | bold(hat(h))_(w, L, j))$ implies the use of softmax normalization over the embedding dimensions in $bold(hat(h))_(w, L, j)$. 

=== Masked Image Modeling (MIM) <masked_image_modeling>

#bibliography("../references.bib")