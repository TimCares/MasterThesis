#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
== Self-Supervised Learning <self_supervised_learning>
=== Motivation

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
which is used in the popular NLP model BERT, the latter being one of the first models trained using self-supervised
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

#figure(
  image(
  width: 50%,
  "../figures/self_supervised_learning_dark_matter.png"),
  caption: [In self-supervised learning parts of the data are masked (grey), and the task of a model is to predict the
  masked parts using the visible data (green) @self_supervised_learning_dark_matter.],
) <self_supervised_learning_dark_matter_fig>

// While self-supervised models are still predictive models, they are trained to e.g. predict the masked tokens in a sentence,
// they differ from supervised models in that they do not provide a probability distribution over a set of abstract classes,
// which describe real-world concepts and objects. So a self-supervised text model can only provide a probability distribution
// over the vocabulary for a masked token, which consists of all possible words in the language, are layer activations
// resulting from the input (text).

// This is important in the context of knowledge distillation. If a teacher model is trained using self-supervised learning,
// then response-based knowledge distillation.

// === Masked Data Modeling <masked_data_modeling>

// ==== Language <mdm_language>

// As introduced in @self_supervised_learning, the self-supervised training of language models is mostly done
// through masked language modeling (MLM). While there are also approaches with Self-Distillation, which will be introduced in
// (TODO: cite Self-Distillation), MLM is by far the most popular approach.

// MLM operates by masking tokens in a sentence and predicting the masked tokens based on the context provided by the remaining tokens.
// For each masked token, the model predicts a probability distribution over the vocabulary, representing all possible tokens.
// The token with the highest probability is considered the predicted token.
// Fundamentally, MLM is a classification task where the model classifies each masked token into one of the tokens in the vocabulary.

// Given a text model $f$ and a text representation input $bold(H)_(w, 0)$, where a fraction $p in (0, 1)$ of tokens $cal(M)$
// (not to be confused with $M$, denoting the number of text tokens) are masked, typically between 15% and 50% of tokens as per @bert
// and @beit3. The set of masked tokens is defined as:

// $
// cal(M) = {i | i tilde op("Uniform")({1, ..., M})} in NN^floor(M*p)
// $

// The process begins by passing the text representation through the model $f$:

// $
// bold(H)_(w, L) = f(bold(H)_(w, 0))
// $

// The representations for the masked tokens are then gathered:

// $
// bold(H')_(w, L) = {bold(h)_(w, l, j) | j in cal(M)}
// $

// These representations are passed through the classification head $op("g")$, which returns logits for each token in the vocabulary:

// $
// bold(hat(H))_(w, L) = op("g")(bold(H')_(w, L)) = {op("g")(bold(h')_(w, L, j)) | j in cal(M)}
// $

// The model is trained to minimize the cross-entropy loss of the correct token $i$ given the output representation of the corresponding
// masked token $bold(hat(h))_(w, L, j)$. The MLM loss for an input text is the sum of the cross-entropy losses for the masked tokens:

// $
// cal(L)_"MLM" = sum_(j in cal(M)) -log exp(bold(hat(h))_(w, L, j)_i)/(sum_(k=1)^V exp(bold(hat(h))_(w, L, j)_k))
// $

// Here, $i$ denotes the index of the correct token in the vocabulary, and $V$ the size of the vocabulary. $j$ loops over all tokens in
// the vocabulary. The fraction, of which the logarithm is taken, essentially represents the probability of the correct token $i$
// at time step $j$, given the model's output $bold(hat(h))_(w, L, j)$ for the masked token at time step $j$.

// === Vision <mdm_vision>

// - not as straightforward as in language
// - text is discrete, images are continuous
// - for text, there are fixed borders -> words or subwords, smallest possible level is single characters.
// - and there is a fixed number of possible tokens that can occur at a time step
// - image has fixed number of pixels, but the number of possible values for each pixel is infinite
// - also, single pixel has no meaning, way too low level, not like a subword or word
// - in vision Transformers, the smallest meaningful unit is a patch of pixels, usually 16x16 or 14x14 pixels in size @vit
// - even if a patch of pixels is used, the possible content of the patch is still infinite -> we can't do a probability distribution
// over all possible patches/pixels, as done with (sub-)words in text

// - solution is training a visual tokenizer as a preprocessing step, before the actual self-supervised training
// - converts patches of pixels into discrete tokens -> represents the content of the patch
// - patches can then be masked, and the model can predict the visual token of the masked patch
// -> number of possible visual tokens is finite, and the model can predict a probability distribution over these tokens
// - set of all possible visual tokens is called codebook, can be thought of as the same as the vocabulary in language models
// - as content of a patch is still infinite, the visual token of the patch is the visual token in the codebook with the highest
// cosine similarity to the content of the patch
// - most popular approach introduced by BEiT @beit and BEiT v2 @beitv2
// -> train a visual tokenizer with a codebook of 8,192 visual tokens -> each image patch is classified into one of 8,192 visual tokens,
// or rather semantic concepts
// - for details on how a visual tokenizer is trained and how it works, please refer to papers BEiT @beit and BEiT v2 @beitv2

// - training process is then the same as in language models

// #figure(
//   image("../figures/beitv2_codebook.png"),
//   caption: [@beitv2.],
// ) <beitv2_codebook>

#bibliography("../references.bib")