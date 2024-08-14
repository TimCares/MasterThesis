=== Image-Text Retrieval <image_text_retrieval>

The goal of image-text retrieval (ITR) is to find the matching (most similar) caption for a given image, and vice versa.
The process begins with embedding and normalizing a set of samples, such as images or captions, which become a set of keys.
For some candidate image or text, called the query, the most similar key is retrieved, after the query is also embedded and normalized.
Similar to contrastive learning, cosine similarity is used to compute the similarity between the query and all keys, which is, again,
computed by matrix multiplication of the normalized embeddings.
The similarities between the query and keys are then ranked, and the key with the highest similarity to the query is the retrieved sample.
This method can be viewed as a form of semantic search, which has significant practical relevance in areas like recommendation systems,
e.g. to find images based on a given text query. This is precisely what is learned through multimodal contrastive learning.

Image-Text Retrieval is a cheap and efficient way to benchmark the quality of the learned representations of a vision-language model,
as it does not require any finetuning, just the embeddings produced by the model.
The metric used for benchmarking is Rank\@K (R\@K), where K determines at which rank the paired/correct sample has to be in the
ranking in order to be considered as a correct retrieval.
We use R@1, R@5, and R@10, where R@1 is the normal accuracy, i.e., the paired sample has to be the most similar one.
R@5 means that the paired sample has to be in the top 5 most similar samples, and for R@10, it has to be in the top 10 most similar samples,
in order for the retrieval to be considered correct.

In this thesis, we use the 5K test set of MSCOCO @coco, and the 1K test set of Flickr30k @flickr30k for benchmarking,
which is the standard benchmarking dataset for multimodal models like FLAVA @flava, CLIP @clip, VLMo @vlmo, and BEiT-3 @beit3.
MSCOCO contains 5K images with 5 captions each @coco, and Flickr30k contains 1K images with 5 captions each @flickr30k.
For both datasets, the all images and all texts are embedded and normalized, so that each image and each text is represented by
the cls token that was returned by the model. Then, matrix multiplication between the images and captions of a dataset
is performed, resulting in a matrix of shape (N, M), where N is the number of images and M is the number of captions in the dataset.
So for MSCOCO, the matrix is of shape (5K, 25K), and for Flickr30k, the matrix is of shape (1K, 5K).

#bibliography("../references.bib")
