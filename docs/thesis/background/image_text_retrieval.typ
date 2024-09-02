== Image-Text Retrieval <image_text_retrieval>

The goal of image-text retrieval (ITR) is to find the matching caption for a given image in a set of captions, and likewise,
finding the matching image for a given caption in a set of images.
The process begins with embedding and normalizing a set of samples, which become a set of keys.
For some normalized candidate representation, called the query, the most similar key is retrieved among the set of keys is the retrieved sample.
This is exactly what is learned through contrastive learning, where we try to maximize the similarity between an image or caption (query)
and its paired caption or image among other samples (keys), respectively.
For that, we can use the same batch-wise computation introduced in the previous section about the contrastive loss.
The similarity is computed by the cosine similarity, which is, again,
computed by matrix multiplication of the normalized embeddings.

Image-Text retrieval can be viewed as a form of semantic search, which has significant practical relevance in areas like recommendation systems,
e.g. to find fitting images based on a given text query. This is precisely what is learned through multimodal contrastive learning.

Image-Text retrieval is a simple and efficient way to benchmark the quality of the learned representations of a vision-language model,
as it does not require any finetuning, just the embeddings produced by the model.
The metric used for benchmarking is Rank\@K (R\@K), where K determines at which rank the paired/correct sample has to be in the
ranking of keys, in order for the retrieval to be considered correct.
We use R@1, R@5, and R@10, where R@1 is the normal accuracy, i.e., the paired sample has to be the most similar one.
R@5 means that the paired sample has to be in the top 5 most similar samples, and for R@10, it has to be in the top 10 most similar samples.

In this thesis, we use the 5K test set of MSCOCO @coco, and the 1K test set of Flickr30k @flickr30k for benchmarking,
which are the standard benchmarking dataset for multimodal models like FLAVA @flava, CLIP @clip, VLMo @vlmo, and BEiT-3 @beit3.
MSCOCO contains 5K images with 5 captions for each image @coco, and Flickr30k contains 1K images with 5 captions each @flickr30k.
For both datasets, all images and all texts are embedded and normalized, so that each image and each text is represented by
the respective $mono(["CLS"])$ token returned by the model. Then, matrix multiplication between all images and all captions of a dataset
is performed, resulting in a matrix of shape (N, M), where N is the number of images and M is the number of captions in the dataset.
So for MSCOCO, the matrix is of shape (5K, 25K), and for Flickr30k, the matrix is of shape (1K, 5K).

For each image, R\@1, R\@5, and R\@10 are computed. The mean of R\@1, R\@5, and R\@10 over all images are then called
text-retrieval of the respective metrics (e.g. R@1-text-retrieval).
We call this text-retrieval, because we are trying to retrieve the correct caption for a given image.
The same is done for each caption, resulting in image-retrieval of the respective metrics (e.g. R@1-image-retrieval).
For each dataset, we have 6 metrics in total: R@1, R@5, and R@10 for text-retrieval and image-retrieval, respectively.
We will report the results of image-text retrieval in the format seen in @image_text_retrieval_example.

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
  caption: [Benchmarks of different vision-language models on the MSCOCO and Flickr30K datasets for image-text retrieval.],
)<image_text_retrieval_example>
