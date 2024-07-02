=== Contrastive Learning <contrastive_learning_section>
- is used to compare samples (e.g. images) with each other in representation space, typically by some distance metric, of which cosine similarity is
  the usual choice
- used in self-supervised learning to learn representations without classical labels such as class targets
- goal is to learn (abstract) representation of the input modality, e.g. images
- originally used in computer vision
  - idea is, that a representation of one image should be similar, or very close, to augmented versions of the same image
  - after all, content of the image stays the same after augmentation (provided the augmentation is not too drastic, e.g. crop size too big)
- goal of the (image) model is to maximize the cosine similary between the original image and its augmented versions
- this alone not sufficient, as model will collapse to a trivial solution, by simply return the same representation for all inputs
  - will maximize the cosine similarity between between the original image and its augmented versions, as representation produced for an image
    will always be the same
- to prevent this, negative samples are introduced
  - negative samples are other images, so not the original image
  - (usually) does not contain the same content as the original image, so cosine similarity between the original image should be minimized
- that way, model can't collapse to a constant representation, as this would not minimize the cosine similarity, and thus not minimize the loss
- this can be extended from unimodal to multimodal applications, in our case: images and text
- here we would like to maximize the cosine similarity between an image and its corresponding text, i.e. caption, and vice versa
- we do not need any augmentation, as we always have pairs: one image and one text
- negative samples for images are captions of other images, and vice versa
- model learns to produce similar representations for an image and its caption, describing the same real-world concept

Implementation:
- staying at our multimodal case, contrastive learning/loss is usually done on the batch-level
- means the multimodal model creates representations for all images and captions in the batch
- then, the cosine similarity between, the representations, of all images and captions in the batch is computed
  - can be done efficiently by normalizing each embedding and then perform matrix multiplication
- for a batch size of e.g. 256, each image has 255 negative samples, i.e. captions of other images, and one positive sample, i.e. its own caption, and vice versa
- can be interpreted as a classification problem with 256 classes, where the model has to predict the correct class, i.e. the positive sample, out of 256 classes/representations
- result of matrix multiplication is a 256x256 matrix with logits, where the diagonal contains the cosine similarity between the positive samples, i.e. the correct class
- we can do softmax row-wise to get probabilities for each image, and column-wise to get probabilities for each caption
- cross-entropy can then be used as the loss function on the probability distributions, metric is accuracy

Problem:
- result highly dependend on the amount of negative samples that are available
  - as an example, if batch size would be two, then the model would have to differentiate between one caption that belongs to the image and one that does not (negative sample), and vice versa
  - a lot simpler than with 255 negative samples, or even more
- result will be better with more negative examples, as task more challenging
- more negative samples can be achieved by using larger batch sizes, but this usually require, depending on the model architecture, higher VRAM GPUs or even multiple GPUs
  - costly

#figure(
  image("../figures/itc.png"),
  caption: [Contrastive Learning is performed using Matrix-Multiplication of normalized representations (1). The diagonal of the resulting matrix contains the cosine similarity between positive samples. The softmax operation along the rows yields a probabilty distribution for each image over all captions, and the softmax operation along the columns vice versa. The cross-entropy loss is then used to calculate the loss for the image scores and caption scores, respectively. The final loss is the mean of both losses. Image-Text pairs in the figure have been taken from the COCO train set @coco.],
) <contrastive_learning_fig>

=== Retrieval
- useful for benchmarking multimodal models
- cheap way, as is does not involve finetuning, just the embeddings produced by the model are needed
- goal: find matching (most similar) caption for a given image, and vice versa
  - means other part of the pair
- we first have a set of samples, e.g. images/captions, which we embed and normalize the embedding-> become a set of keys
- then we have a query, e.g. an image/text, which we also embed and normalize -> becomes a query
- now we compute cosine similarity between the query and all keys
- rank them based on the similarity, and retrieved sample is the one with the highest similarity
- computation can be done the same as described for contrastive learning @contrastive_learning_section (contrastive learning and retrieval are basically the same)
  - only softmax operation is not needed (step 2 in @contrastive_learning_fig)
  - just take the maximum of all similarities -> most similar sample
- in application of this thesis, as contrastive learning, done for image-text pairs
- i.e. find caption for a given image in all captions of the dataset, that is used for benchmarking, and vice versa
- metric used is Rank\@K (R\@K), where K determines at which rank the paired sample has at least to be in the ranking in order to be considered as a correct retrieval
  - we use R@1, R@5, and R@10
  - R@1 is simply the normal accuracy, i.e. paired sample has to be the most similar one
  - R@5 means that the paired sample has to be in the top 5 most similar samples
  - R@10 means that the paired sample has to be in the top 10 most similar samples
- in this thesis we use the 5k test set of MSCOCO @coco, and the 1k test set of Flickr30k @flickr30k for benchmarking
- used by most multimodal models like FLAVA @flava, CLIP @clip, VLMo @vlmo, and BEiT-3 @beit3
-> we can easily and cheaply compare our model to those papers/models