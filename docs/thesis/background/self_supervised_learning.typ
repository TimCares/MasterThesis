== Self-Supervised Learning <self_supervised_learning>
=== Motivation
While we previously identified large Deep Learning models as generally expensive to train,
we now focus on the problem of scalability in the context of supervised learning, the most common form of training AI models.

Supervised models, while powerful, are not inherently scalable. Although their architecture can be
extended to create larger models that achieve better performance, these larger models require more data for training.
In the context of supervised learning, this data must be labeled, which presents a significant challenge.
Labeled data is scarce and expensive to obtain, as it requires human annotation, thereby limiting the scalability of supervised models.

The primary objective of self-supervised learning is to learn representations of data without relying on human-annotated labels.
However, self-supervised learning is not unsupervised learning.
Unsupervised learning operates without any form of supervision, meaning that no labels are required at all,
as seen in clustering methods like K-means. In contrast, self-supervised learning requires, as supervised learning,
labeled data, but in contrast to supervised learning labels are generated directly from the data itself.

A prominent example of self-supervised learning is Masked Language Modeling (MLM) in Natural Language Processing (NLP),
which is used in the popular NLP model BERT, being one of the first models trained using self-supervised
methods to achieve state-of-the-art performance in NLP @bert.
In BERT, certain tokens, or words, are masked, i.e., removed, from a sentence, and the model is tasked with predicting the masked tokens.
Since the labels are derived from the data itself — the words to predict are part of the original data — no human annotation is needed @bert.
This allows for the utilization of large amounts of unlabeled data, as any text data can be used.

What makes self-supervised learning particularly powerful is its applicability to any type of data with a hierarchical structure, such as text,
images, audio, or video. In these cases, part of the data can be masked, and the model must predict the masked part based on the context
provided by the remaining data. An intuitive example, presented by Yann LeCun and Ishan Misra of Meta, illustrates why this approach is effective.
Consider the sentence “The lions chase the wildebeests in the savanna.” If “lions” and “wildebeests” are masked, the input becomes
“The [MASK] chases the [MASK] in the savanna.”. To successfully predict the masked words, the model must understand the real-world
concepts expressed by the sentence. While “The cat chases the mouse in the savanna” might be a valid prediction in the context of
“chase,” the word “savanna” provides additional context, as it is not a typical habitat for cats and mice, but rather for lions and wildebeests.
Thus, the model must understand that lions and wildebeests are animals that inhabit savannas, in order to make a correct prediction.
Through this process of predicting masked words, the model learns about the concepts of the world we live in @self_supervised_learning_dark_matter.

While this example is specific to text data, the same principle can be applied to other types of hierarchical data, e.g. images and audio.

Consequently, self-supervised learning allows makes models scalable, as they can be trained on large amounts of unlabeled data.

#figure(
  image(
  width: 50%,
  "../figures/self_supervised_learning_dark_matter.png"),
  caption: [In self-supervised learning parts of the data are masked (grey), and the task of a model is to predict the
  masked parts using the visible data (green) @self_supervised_learning_dark_matter.],
) <self_supervised_learning_dark_matter_fig>

=== Relationship to Multimodal Models