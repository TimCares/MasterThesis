#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

=== Contrastive Learning <contrastive_learning>
==== Vision
In settings where masking discrete tokens and predicting them based on a set of possible tokens, as in language models,
is not possible, contrastive learning can be used as a self-supervised method.
This is especially useful in vision models, as images are continuous, so there is no discrete set of possible tokens/words to predict.

Contrastive learning, or the contrastive loss, is a method
to learn representations of data without the need for labels, and used in computer vision models like MoCo @moco, SimCLR @simclr,
and CLIP @clip.

In computer vision, contrastive learning exploits the fact that the high-level semantics of an image are invariant to
small (or moderate) changes in pixel-level information. This is achieved by augmenting the input image, e.g., by cropping,
rotating, or flipping it. Provided the augmentation is not too drastic (e.g. the crop size too large),
the high-level semantics of the image will remain the same after augmentation, even though pixel-level information do not.
The goal of the image model is then to maximize the cosine similarity between the global representations of two 
augmented versions of the same image. In Transformers, the global representation is usually the $mono(["I_CLS"])$ token retuned
by the final layer of the model, which is a vector that can be compared with the $mono(["I_CLS"])$ token of another image
using the cosine similarity. 
The augmented versions are often referred to as different _views_ of the same image @simsiam, as shown in @image_views.

#figure(
  image(
  width: 25%,
  "../figures/image_views.png"),
  caption: [Adding small translations to an image, e.g. a random crop, as illustrated in the figure, will retrain
  high-level semantic features while changing pixel-level information. The content of the image stays the same, and the
  same should therefore hold for the representations produced by the model.
  Image in the figure has been taken from the COCO train set @coco.],
) <image_views>

However, this alone is not sufficient, as the model will collapse to a trivial solution by simply returning 
the same representation for all inputs, as demonstrated in the papers MoCo @moco and SimSiam @simsiam.
Producing the same representation for all inputs is the simplest way to maximize the cosine similarity between the original image
and its augmented versions, because the representation produced for an image would always be the same, therefore maximizing the cosine
similarity (a value of 1).
To prevent this, negative samples are introduced. Negative samples are other images that do not contain the same
content as the original image, and the cosine similarity between the original image and these
negative samples should therefore be minimized.
This prevents the model from collapsing to a constant representation, as it would not minimize the cosine similarity
between negative samples, and thus not minimize the loss. A simple yet expressive visualization can be found here
#footnote[#link("https://research.google/blog/advancing-self-supervised-and-semi-supervised-learning-with-simclr/")].
This makes self-supervised training of image models possible, and the learned representations represent the high-level semantics
of the images, learned without the need for labels.

An implementation and mathematical formulation of the contrastive loss will be introduced
on the example of vision-language models in the next section.

==== Vision-Language <vision_language_contrast>

Introduced as a method for self-supervised learning of image models, contrastive learning
can be extended from unimodal (image) to multimodal applications, such as image and text.
As mentioned in the previous section, we aim to maximize the cosine similarity between
an image and its corresponding text (i.e., caption), and vice versa.
Augmentation is not needed, as we always have pairs: one image and one text.
Negative samples for images are captions of other images, and vice versa.
In this setting, the model learns to produce similar representations for an image and its caption, describing the same real-world concept,
and dissimilar representations for an image and caption that are unrelated. A conceptual example for both vision and vision-language
contrastive learning can be seen in @contrastive_alignment.

#figure(
  image(
  width: 75%,
  "../figures/contrastive_alignment.png"),
  caption: [Contrastive learning aims to align the same (or similar) real-world concepts in representation space, while pushing different concepts apart. Multimodal contrastive learning (b) requires existing pairs, e.g. image-text, while for the unimodal case (a) pairs are synthetically created by augmenting the input. Images and text in the figure have been taken from the COCO train set @coco.],
) <contrastive_alignment>


Contrastive learning requires a (global) representation of the input, which is then used to compare it with other inputs.
Since the introduction of the vision Transformer in 2020 by Dosovitskiy et al. @vit, most vision-language models
are exclusively based on the Transformer architecture, which is why the cls token is used as the global representation
for both image ($mono(["I_CLS"])$) and text ($mono(["T_CLS"])$), respectively. There have been other approaches, such as
Cross-Modal Late Interaction introduced in FLILP @filip, but they usually require significantly more compute @filip and
do not outperform global contrastive learning @beit3, which is what we use here.

The representations are generated by passing the image sequence $bold(H)_(v, 0)$ and text sequence $bold(H)_(w, 0)$
through the vision-language model,
and extracting the representations for both tokens ($bold(h)_(v, L, mono(["I_CLS"]))$ and $bold(h)_(w, L, mono(["T_CLS"]))$)
from the output of the final layer $bold(H)_(v, L)$ and $bold(H)_(w, L)$, which is the output of the Transformer.
For the resulting batch of image and text representations
${bold(h)_((v, L, mono(["I_CLS"])), k), bold(h)_((w, L, mono(["T_CLS"])), k)}_(k=1)^B$, where $B$ is the batch size,
the cosine similarity between all possible image-text pairs is computed. The cosine similarity between two
vecotrs $bold(a)$ and $bold(b)$ is given by:

$
cos(bold(a), bold(b)) = (bold(a) bold(b)^T) / (||bold(a)||_2 * ||bold(b)||_2) = bold(a)/(||bold(a)||_2) bold(b)^T/(||bold(b)||_2)
$

$bold(a) bold(b)^T$ denotes the simple dot product between both representations. $||bold(a)||_2$ and $||bold(b)||_2$
denote the L2-norm of the representations.

The cosine similarity between all possible image-text pairs can be computed efficiently by organizing all image and text representations
in a matrix, which is already given in a batch-wise training, and normalizing every representation, which we
denote by the function $delta(dot)$.

$
delta(bold(h)) = bold(h) / (||bold(h)||_2)
$

$
bold(I) = [delta(bold(h)_((v, L, mono(["I_CLS"])),1)), delta(bold(h)_((v, L, mono(["I_CLS"])),2)),
..., delta(bold(h)_((v, L, mono(["I_CLS"])),B))] in RR^(B times D)
$

$
bold(T) = [delta(bold(h)_((w, L, mono(["T_CLS"])),1)), delta(bold(h)_((w, L, mono(["T_CLS"])),2)),
..., delta(bold(h)_((w, L, mono(["T_CLS"])),B))] in RR^(B times D)
$

$bold(I)$ denotes the batch/matrix of image representations, and $bold(T)$ contains the text representations.
$D$ is the dimensionality of the representations.

A matrix multiplication of both batches of representations then computes the dot product between every image with every text, and vice versa.
Since the representations are normalized, the result will be the cosine similarity between all possible image-text pairs in the batch.

$
bold(L) = bold(I) bold(T)^T, bold(L) in RR^(B times B)
$ <contrastive_logits>

$L_(i,j)$ then denotes the similarity between image $i$ and text $j$ in the batch. The diagonal of the matrix contains the similarity
between positive pairs, i.e. the correct image-text pairs $(i,i)$, with $L_(i, i)$ describing their similarity.
For an image, all other texts in the batch are considered as negative samples, and vice versa for text. The superscript $T$ denotes the transpose
of a matrix, and is not to be confused with the batch of text representations $bold(T)$.

For a batch size of 256 ($B=256$), each image has 255 negative samples (captions of other images)
and one positive sample (its own caption), the same holds vice versa. This can be seen as a classification problem with 256 classes,
where the model has to predict the correct class out of 256 classes, and each class representing one caption or image, respectively.
For an image, the logit for the correct class is the similarity (cosine) to its own caption, and the logits for the negative classes
are the similarities to the captions of other images. The same holds vice versa for text.

To calculate the loss, the cross-entropy loss is used. For a batch, the loss for selecting the correct caption for each image is given by:

$
cal(L)_"CL"^("i2t") = 1/B sum_(i=0)^(B-1) -log exp(L_(i, i))/(sum_(k=0)^(B-1) exp(L_(i, k)))
$ <contrastive_loss_i2t>

$exp(L_(i, i))/(sum_(k=0)^(B-1) exp(L_(i, k)))$ denotes the softmax-normalized similarity between an image and its correct caption,
which is the usual way to calculate the cross-entropy. The result of this normalization is a probability distribution for each image,
where each caption in the batch has a probability of being the correct caption for the image, and vice versa. The probability that the correct
caption belongs to the current image is then used to calculate the negative log-likelihood, which is the loss.

Accordingly, the loss for selecting the correct image for each caption is given by:

$
cal(L)_"CL"^("t2i") = 1/B sum_(i=0)^(B-1) -log exp(L_(i, i))/(sum_(k=0)^(B-1) exp(L_(k, i)))
$ <contrastive_loss_t2i>

Here, the softmax-normalization is with respect to the similarity of a text with all other images in the batch. The final loss is the mean
of the image-to-text and text-to-image loss:

$
cal(L)_"CL" = 1/2 * (cal(L)_"CL"^("i2t") + cal(L)_"CL"^("t2i"))
$

Returning to the concept of contrastive learning, this process ensures that the similarity between the representation of an image and its caption
is maximized, i.e. close to each other, while the similarity between an image and an unrelated caption is minimized, i.e. far apart.
Only this would appropriately minimize the loss, and thus the model learns to align the representations of the same concept across modalities.
An illustration of multimodal contrastive learning can be found in @contrastive_learning_fig. Intuitively, it can be
thought of as _finding the correct caption for an image among all captions in the current batch_, and vice versa.

#figure(
  image("../figures/itc.png"),
  caption: [Contrastive Learning is performed using matrix multiplication of normalized representations (1), and the result
  is matrix $bold(L)$ described in @contrastive_logits. The diagonal of the resulting matrix contains
  the cosine similarity between positive samples. The softmax operation along the rows yields a probabilty distribution for
  each image over all captions, and the softmax operation along the columns vice versa (2). The cross-entropy loss is
  then used to calculate the loss for the distributions. The final loss is the mean of both losses.
  Image-text pairs in the figure have been taken from the COCO train set @coco.],
) <contrastive_learning_fig>

The performance of contrastive learning is highly dependent on the number of negative samples available, which directly
translates to the batch size.
For instance, with a batch size of two, the model only needs to differentiate between one caption that belongs
to the image and one that does not (a negative sample), and vice versa. This task is significantly simpler than
with 255 negative samples or more, where there might be captions
that are semantically similar to the image, but do not belong to it. With increased negative samples,
the probability of encountering hard-negative examples increases, forcing the model to aggregate as much information as possible
in $mono(["I_CLS"])$ and $mono(["T_CLS"])$ to even differentiate between semantically similar concepts.

The results improve with an increased number of negative examples @moco @beit3, which 
we will also show later
More negative samples are usually achieved by using larger batch sizes @moco @clip @beit3,
however, this typically requires higher VRAM GPUs, or multiple GPUs, which is costly.

== Image-Text Retrieval <image_text_retrieval>

The goal of image-text retrieval (ITR) is to find the matching caption for a given image in a set of captions, and likewise,
finding the matching image for a given caption in a set of images.
This is exactly what is learned through contrastive learning, where we try to maximize the similarity between an image
and its paired caption among other captions, and vice versa.
The process begins with embedding and normalizing a set of samples, which become a set of keys.
For some normalized candidate representation, called the query, the most similar key among the set
of keys is the retrieved sample. In the context of vision-language models, the query could be an image, and the keys
a set of captions.
For the normalization and comparison the same batch-wise computation introduced for contrastive loss can be used.
The retrieved sample is the one with the highest cosine similarity to the query among all keys.

Image-text retrieval is the direct application of vision-language contrastive learning, and can be viewed as
a form of semantic search, which has significant practical relevance in areas like recommendation systems,
e.g. to find fitting images based on a given text query. This is precisely what is learned through multimodal contrastive learning.

Image-text retrieval is a simple and efficient way to benchmark the quality of the learned representations of a vision-language model,
as it does not require any finetuning, just the embeddings produced by the model.
The metric used for benchmarking is Rank\@K (R\@K), where K determines at which rank the paired/correct sample has to be in the
ranking of keys, in order for the retrieval to be considered correct.
We use R@1, R@5, and R@10, where R@1 is the normal accuracy, i.e., the paired sample has to be the most similar one.
R@5 means that the paired sample has to be in the top 5 most similar samples, and for R@10, it has to be in the top 10 most similar samples.
The higher the R\@K scores, the better the model has learned to align the representations of the same concept across modalities.

In this thesis, we use the 5K test set of MSCOCO @coco, and the 1K test set of Flickr30k @flickr30k for benchmarking,
which are the standard benchmarking dataset for multimodal models like FLAVA @flava, CLIP @clip, VLMo @vlmo, and BEiT-3 @beit3.
MSCOCO contains 5K images with 5 captions for each image @coco, and Flickr30k contains 1K images with 5 captions each @flickr30k.
For both datasets, all images and all texts are embedded and normalized, so that each image and each text is represented by
the respective cls token returned by the model. Then, matrix multiplication between all images and all captions of a dataset
is performed, resulting in a matrix of shape (F, W), where F is the number of images and W is the number of captions in the dataset.
So for MSCOCO, the matrix is of shape (5K, 25K), and for Flickr30k, the matrix is of shape (1K, 5K).

#show table: set text(8pt)
#figure(
  table(
  columns: (25%, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: none,
    table.hline(),
    table.header(
      table.cell(rowspan: 3, colspan: 1, align:horizon, [*Model*]),
      table.cell(colspan: 6, [*MSCOCO (5K test set)*]),
      table.cell(colspan: 6, [*Flickr30K (1K test set)*]),
      table.cell(colspan: 3, [Image $arrow.r$ Text]),
      table.cell(colspan: 3, [Text $arrow.r$ Image]),
      table.vline(stroke: .4pt),
      table.cell(colspan: 3, [Image $arrow.r$ Text]),
      table.cell(colspan: 3, [Text $arrow.r$ Image]),
      table.hline(start: 1, end: 4, stroke: .2pt),
      table.hline(start: 4, end: 7, stroke: .2pt),
      table.hline(start: 7, end: 10, stroke: .2pt),
      table.hline(start: 10, end: 13, stroke: .2pt),
      [R@1], [R@5], [R@10], [R@1], [R@5], [R@10], [R@1], [R@5], [R@10], [R@1], [R@5], [R@10]
    ),
    table.hline(stroke: .4pt),
    [FLAVA @flava], [42.74], [76.76], [-], [38.38], [67.47], [-], [67.7], [94.0], [-], [65.22], [89.38], [-],
    [CLIP @clip], [58.4], [81.5], [88.1], [37.8], [62.4], [72.2], [88.0],[98.7], [99.4], [68.7], [90.6], [95.2],
    [BEiT-3 @beit3], [84.8], [96.5], [98.3], [67.2], [87.7], [92.8], [98.0],[100.0], [100.0], [90.3], [98.7], [99.5],
    table.hline(),
  ),
  caption: [Benchmarks of different vision-language models on the MSCOCO and Flickr30K datasets for image-text retrieval.
  Metrics are in percent, so the maximum score is 100(%).],
)<image_text_retrieval_example>
#show table: set text(11pt)

For each image, R\@1, R\@5, and R\@10 are computed. The mean of R\@1, R\@5, and R\@10 over all images are then called
text-retrieval of the respective metrics (e.g. R@1-text-retrieval).
We call this text-retrieval, because we are trying to retrieve the correct caption for a given image.
The same is done for each caption, resulting in image-retrieval of the respective metrics (e.g. R@1-image-retrieval).
For each dataset, we have 6 metrics in total: R@1, R@5, and R@10 for text-retrieval and image-retrieval, respectively.
We will report the results of image-text retrieval in the format seen in @image_text_retrieval_example.
