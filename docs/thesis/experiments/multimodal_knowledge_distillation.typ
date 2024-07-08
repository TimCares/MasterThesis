== Multimodal Knowledge Distillation
=== Aligned Representations
==== Dual encoder
===== Unimodal Student
- we start as simple as possible
- we just want to create aligned cross-modal representations, i.e. have the same representation for the same concept -> same
  representation for an image and its corresponding text (image-text pair in the dataset)
- multiple architectures are possible, we start with a CLIP-like architecture
- means one encoder per modalitiy, so one for image and one for text
- usually trained for scratch -> but expensive, and we want to utilize already pretrained(!) (self-supervised, not trained on labeled data) 
  unimodal models
- we use BEiT-2 as the teacher model, which is an image model, and at the same time serves as our image encoder
- to now align representations, we just train the text encoder to regress the cls token of the image encoder
- hope is that the cls token of the teacher, which is regressed, has learned a representation that is abstract enough so that it can be
  applied independently of the modality, in this case images
  - although there has been no incentive for the teacher model to learn which is independent of the modality, as the model has only been pretrained
    on images -> if the representation (cls token) does still contain image specific information, the student model will not be able to learn a
    meaningful representation of the text, as image and text are inherently different
- in the first experiment we just train the text encoder to regress the cls token of the image encoder, nothing more
- the text model is smaller than the image model, it contains only 7 layers, while the image model contains 12 layers
- hope is that through Knowledge-Distillation we do not need a model as large as the teacher model, as we have seen that a smaller model can achive
  a performance quite similar to the larger teacher model through KD in the unimodal case
  - whether this translates to an multimodal setting can be derived from the results of this experiment
  - this could be seen in the retrieval application of our model -> if the performance on text-image retrieval is similar to the performance of the
    teacher model, then we can assume that the student model has learned a meaningful representation of the text(?)
- we still have the option to expand our student model to the same size as the teacher model, i.e. 12 layers
- still relatively cheap, as we only have to train the text encoder and, as we are doing in the first experiment with 7 layers, we can initialize
  the text encoder (the student) with the weights of the text D2V2 model, meaning we do not start from scratch

===== Reproducing: "See, Hear, and Read: Deep Aligned Representations" @shre
- we start as simple as possible
- we first want to reproduce the results of the paper "See, Hear, and Read: Deep Aligned Representations" @shre
- for the sake of simplicity, we will refer to the paper as SHRe (SEe, HEar, and REad), as the title is quite long and the authors do not name the architecture
- will give us a baseline on which to improve, compare our results to (especially because SHRe does not benchmark retrieval on karpathy COCO), and test new ideas
- we use the same architecture as in the paper, which is a dual encoder architecture
  - has one text and one image encoder/network, quite like CLIP does
- difference to CLIP is: we now have a shared network at the top
- for our image and text encoder we use the pretrained image D2V2, and the pretrained text D2V2, model respectively
  - this differs to SHRe, as they used conv nets for both image and text encoder
- unlike SHRe, we directly initialize the encoders with the weights of the respective D2V2 model, and do not train them from scratch
- further, because we would like to keep the model smaller, which is less expensive to train and we can train it, thanks to Deepspeed,
  on a single GPU, we only use the first 7 layers of the D2V2 models
- SHRe added two MLP layers as the shared network, we will do the same, but use the same MLP architecture as present in Transformer layers
  - means, two linear layers, with a GELU activation and layer norm in between
  - first linear layer expands the embedding dimension to 4 times the size, second linear layer reduces it to the number of classes in imagenet (1000), not
    the embedding dimension
- recall the SHRe uses the kl-divergence loss, common in labeled KD
  - means the output of our multimodal model, i.e. shared network, is a probablity distribution over the classes in imagenet
- we regress the probablity distribution of a resnet-b-50 model, which was trained on imagenet with labeled data, and is used as the teacher model
  - SHRe did not mention which architecture they used, only that teacher was trained on imagenet
- to pass the features from the modality specific encoders (image and text), which operate on time steps, to the shared network, which does not operate on time steps, we use the cls tokens of the output of the encoders (the first token)
- SHRe uses two training targets: minimization of the KL-divergence, and the maximization of the cosine similarity for the
  activations of the shared network between matching image-text pairs (positive) and the minimization of the cosine similarity for the activations of the shared network between non-matching image-text pairs (negative)
- this is basically a contrastive loss, similar to that of CLIP
- we do not exactly follow this approach, but instead the of VLMo
- we compute the cosine similarity between the normalized logits of the shared network for all possible image-text pairs of a batch
  - we take the output of our model, i.e. the shared network, without softmax, and normalize it
- and then take softmax over the cosine similarities, and then compute the cross-entropy loss
- for each image/text, the target is the matching text/image in the batch, of which there is only one, and the rest texts/images in the batch are non-matching
- following CLIP, we divide the cosine similarity by a, in log-space, learnable temperature parameter
- this way we align image-text pairs in predictions, i.e. class probabilities, and in represenation space

Training Setup:
- we train the model for 60k steps, with a batch size of 256
- we train using Deepspeed stage 2, to be able to train on a single GPU
- we use the AdamW optimizer, with a peak learning rate of 5e-4
  - we do not have the resources for extensive tuning of the learning rate, and other hyperparameters
  - we therfore use the same learning rate as in the unimodal experiments, which gave good results
- we set the AdamW epsilon to 1e-6, weight decay to 0.01, and the betas to (0.9,0.98)
- we use cosine scheduling, with a warmup of 10% of the total steps -> 6k steps

- as explained in the chapter about the datasets, we set the max text sequence length to 64 tokens
  - captions tend to be small and concise, as shown in section < TODO >
  - enables faster training and less memory usage
- we use COCO and Conceptual Captions in a round-robin fashion

- text encoder and image encoder are initialized with the weights of the respective D2V2 models
  - we only use the first 7 layers of the D2V2 models so that the model is smaller
  - therefore, each encoder has 7 transformer layers
- shared network is a two-layer MLP, and follows the same architecture as the MLP in a standard transformer layer
  - first layer has 4 times the size of the embedding dimension, second layer directly reduces it to the number of classes in imagenet (1000)
  - second layer is output layer, initialized with random weights

- we use the CLIP implementation of imagenet zero-shot classification during validation, which we run every 6k steps

#figure(
  table(
    table.vline(x:1, stroke: .3pt),
    table.vline(x:2, stroke: .3pt),
    columns: 3,
    stroke: none,
    table.hline(),
    table.header(
      [*Type*],
      [*Hyperparameter*],
      [*Value*],
    ),
    table.hline(stroke: .6pt),
    table.cell(rowspan: 5, align:horizon, [*_Image/Text Encoder_*]),
    [Layers], [7],
    [Hidden size], [768],
    [Attention heads], [12],
    [FFN inner hidden size], [3072],
    [Shared Encoder FFN output size], [1000],
    table.hline(stroke: .6pt),
    table.cell(rowspan: 9, align:horizon, [_*Training*_]), 
    [Training steps], [60,000],
    [Batch size], [256],
    [AdamW ε], [1e-6],
    [AdamW β], [(0.9, 0.98)],
    [Peak learning rate], [5e-4],
    [Learning rate schedule], [Cosine],
    [Warmup steps], [6k (10%)],
    [Weight decay], [0.01],
    [Teacher], [Imagenet ResNet-50]
  ),
  caption: [Hyperparameters of reproduced SHRe model.],
)<shre_first_hyperparams>

After the training, we evaluate the model on the same CLIP zero-shot imagenet classification task, and using image-text retrieval on the MSCOCO and Flickr30K datasets. In both case the model is relieant on the quality of the representations it produces, making it a good benchmark. Further, we can compare the results directly to other papers like FLAVA, VLMo, and CLIP, and do not need to do any seperate finetuning. Both benchmarks are a direct indicator of the success of the method.

#figure(
  table(
    columns: 2,
    stroke: none,
    table.hline(),
    table.header(
      [*Model*],
      [*Accuracy*],
    ),
    table.hline(stroke: .6pt),
    [Random], [0.001],
    [Visual N-Grams @visual_n_grams], [11.5],
    [CLIP @clip], [*72.6*],
    [*SHRe reproduction (ours)*], [21.8],
    table.hline(),
  ),
  caption: [First comparison of zero-shot imagenet classification accuracy with CLIP and Visual N-Grams.],
)<clip_imagenet_zero_shot_first>

As seen in table @clip_imagenet_zero_shot_first, the model achieves almost double the accuracy as Visual N-Grams, which was the initial approach on zero-shot transfer for classification. However, CLIP outperforms our model by a margin.
This is to be expected as:
1. The reported accuracy uses a model setup with 428 million parameters.
2. The model was trained on up to 400 million image-text pairs.
3. The model was trained on 256 V100 GPUs for 12 days.

Huge difference to our model, which has 144 million parameters, was trained on just short of 1.4 million image-text pairs, and was trained on a single RTX 4090 for 7 hours. The cost accounts to 5.25 USD, compared to more than 73,000 USD for the CLIP model
#footnote[Calculation was done based on the hourly GPU cost on #link("https://runpod.io")[runpod.io], which is the platform we used to rent GPUs. As of the time of our research,
a single RTX 4090 costs 0.75 USD, and a single V100 1 USD per hour.].

- table @image_text_retrieval_shre_first shows results of image-text retrieval on MSCOCO and Flickr30K
- for now, we only compare to CLIP and FLAVA, as this is the only dual encoder model, i.e. the image and text encoder are completely seperate
- we will compare to VLMo and BEiT-3 in the section on Mixture-of-Modality-Experts

- we observe that model performs quite well on MSCOCO, but not so well on Flickr30K
- reason might be that MSCOCO training and test datasets are very similar, and since we only use Conceptual Captions as additional data,
  the model might be very biased towards MSCOCO
- would explain poorer performance on Flickr30K, as the model has not seen any Flickr30k data during training
- we can even see that the model performs slightly better image retrieval for MSCOCO on R@10, and is, for the model size and training data,
  quite close to CLIP and FLAVA
- important to note that the teacher is still supervised, which is a clear advantage over CLIP and FLAVA, which are trained from scratch, without KD
- will be even more interesting to see how the model performs when the teacher is self-supervised, and does not provide us with a probability distribution to regress -> later sections

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
    table.cell([_Zero-Shot_], align: left), table.cell(colspan: 12, []),
    [FLAVA @flava], [42.74], [76.76], [-], [38.38], [67.47], [-], [67.7], [94.0], [-], [65.22], [89.38], [-],
    [CLIP @clip], [58.4], [81.5], [88.1], [37.8], [62.4], [72.2], [88.0],[98.7], [99.4], [68.7], [90.6], [95.2],
    [*SHRe \ reproduction (ours)*], [41.36], [71.16], [82.0], [30.2], [59.46], [72.54], [9.5], [35.68], [50.18], [8.38], [37.54], [49.88],
    table.hline(),
  ),
  caption: [],
)<image_text_retrieval_shre_first>

These results can already be considered as a success, as the aim of this work is not to reach state-of-the-art performance, but to create a poof-of-concept for multimodal knowledge distillation, although a high performance is desirable.

==== Multimodal Self-Attention
- VLMo showed that shared block(s) do not necessarily have to only be linear layers, as in SHRe @shre
- they use normal Transformer blocks, with self-attention @vlmo
  - required self-attention to capture modalitiy-independent information
- seemd to work well in practice, so we adapt this approach, so that our model consists only of Transformer blocks

==== Applying Contrastive Learning
==== On FLAVA's retrieval performance
- FLAVA authors claim their performance on image-text retrieval on MSCOCO and Flickr30K is zero-shot
- however, this is not true
- they train there model using contrastive loss, among other losses
- as mentioned in @retrieval, contrastive learning and retrieval are basically the same
  - contrastive learning uses cosine similarity to push embeddings of matching pairs closer together, and embeddings of non-matching pairs further apart
  - retrieval uses cosine similarity to find the matching pair for a given query (aims to e.g. find the caption for a given image)
- so with contrastive learning, the model learns to perform retrieval
- zero-shot is when a trained model is applied to a task it has not been trained on, e.g. when a multimodal model is trained only using masked-image-modeling (MIM) and masked-language-modeling (MLM), and then applied to image-text retrieval
  - or when an LLM is applied to a tabular classification task
- however, FLAVA uses contrastive learning, and therefore the model has been trained on the retrieval task, and is not zero-shot
- also the model has been trained using the MSCOCO train set, and the image-text retrieval is done on the MSCOCO test set
- if the samples from MSCOCO train and test set are similar, then it is just a normal application of a trained model to a task it has been trained on (among other tasks)
- application of itr on Flickr30K is still not zero-shot, as, again, the model has been trained using a contrastive loss
- one example for actual zero-shot retrieval would be BEiT-3, having only been trained using MIM and MLM @beit3
  - finetuning BEiT-3 on Flickr30K using contrastive less then show slight improvement

#figure(
  table(
  columns: (25%, auto, auto, auto, auto, auto, auto),
    stroke: none,
    table.hline(),
    table.header(
      table.cell(rowspan: 3, colspan: 1, align:horizon, [*Model*]),
      table.cell(colspan: 6, [*Flickr30K (1K test set)*]),
      table.cell(colspan: 3, [Image $arrow.r$ Text]),
      table.cell(colspan: 3, [Text $arrow.r$ Image]),
      table.hline(start: 1, end: 4, stroke: .2pt),
      table.hline(start: 4, end: 7, stroke: .2pt),
      [R@1], [R@5], [R@10], [R@1], [R@5], [R@10]
    ),
    table.hline(stroke: .4pt),
    [BEiT-3 _zero-shot_], [94.9], [99.9], [*100.0*], [81.5], [95.6], [97.8],
    [BEiT-3 _finetuning_], [*98.0*], [*100.0*], [*100.0*], [*90.3*], [*98.7*], [*99.5*],
    table.hline(),
  ),
  caption: [],
)<beit3_flickr30k>

==== Short Captions <short_captions>
- many papers use the Visual Genome dataset, consisting of images with region descriptions
  -> attractive source, as region descriptions are human annotated and highly curated -> focus on specfic regions of the image
- as mentioned in section @data_collection_and_preprocessing, we do not use Visual Genome because we encoutered problems when using it with Contrastive Learning
- @itc_vg shows accuracy on image-text contrast, which is image-text retrieval, when using data datasets in combination with Visual Genome and without

#figure(
  image("../figures/itc_vg.png"),
  caption: [Training accuracy of Image-Text Contrast with Visual Genome (left) vs. without Visual Genome (right). Removing Visual Genome from the training data leads to a more stable training and a higher accuracy in the first 6k steps.],
) <itc_vg>

- comparison is only for the first 6k steps, as we stop experiments that show errors or do not seem promising, due to the high computational cost
- contrary to expectations, the accuracy of the model without Visual Genome continously increases after the first 6k steps (@itc_vg_full), where is stagnates for a while

#figure(
  image("../figures/itc_vg_full.png", width: 50%),
  caption: [],
) <itc_vg_full>

- we assume reason is that region descriptions are too specific, i.e. focus on a specific part/region of the image, and do not capture the overall content of the image
- also, since the regions can be small, the caption will also be

==== Memory Bank for Larger Batch Sizes
- as mentioned in the section about contrastive learning @contrastive_learning_section, quality of representations learning using contrastive loss greatly improves with more negative samples
-> for example, CLIP, trained only using contrastive learning, uses a batch size of 32k @clip, the the large variant of VLMo between 16k and 32k @vlmo, and FLAVA 8k @flava
- not necessary though -> base model of VLMo uses "just" 1024 @vlmo, and SHRe just 200 @shre, both achieve, as described in the respective chapters, good results
- we are limited by GPU memory, currently using a batch size of 256, can't increase it without further optimizations or multiple GPUs
===== GPU Offloading for Larger Batch Sizes
===== VLMo Contrast vs. SHRe Contrast

===== Feature-based Knowledge Distillation
- if we want to keep the approach from SHRe, i.e. use KD and contrastive learning, for a self-supervised trained teacher that does not provide us with a
  probabilty distribution to regress, then we have to regress the activations/features of the teacher model
- this is similar to the unimodal distillation in the first experimental sections
- however, we can't just regress all time steps of the teacher model, i.e. all activations of the teacher, as we did in the unimodal case
  - this works for the image encoder of our multimodal model, because the teacher model is an image model, but not for the text encoder
- firstly, is is not possible because the text might not have as many tokens (time steps) as the image has patches (time steps)
- so regressing a timestep/patch of the image, say 180, with the same timestep of the text, also 180, is not possible if the text is not that long
  - in our case for example, the text has a max sequence length of only 64 tokens
- even if we were to set the max text sequence length to the same number of time steps as an image has (196+1), i.e. through padding,
  it will certainly never be the case that a specific timestep/patch of an image aligns semantically with the text token at the same timestep 
- the reason is that each time step for an image corresponds to a patch, which contains some information
- if we want to align the representation of that specific patch with a text token as the same timestep, then the text token has to contain the same
  information as the patch in text form
  - e.g. if a patch contains a piece of cheese, then the text token at the same timestep has to contain the word "cheese"
  - this would have to hold for all timesteps
  - so no fill words or padding allowed, as they do not contain any information
  - text would not make any sense and would not describe the overall content of the image
- therefore, we have to somehow regress the global information of the image, and can't regress any specific patch
- this is where the cls token comes in handy, as its goal is to aggregate a global representation of the image
- this is independent of any patch
- if our text encoder also has a cls token, then we can make the cls token of the text encoder regress the cls token of the (image) teacher
- in principle, for the image encoder we could regress all timesteps as in the unimodal case, as this is the same modalitiy, but we should
  keep it consistent and regress the cls token of the image encoder as well

- in principle nothing "bad", as probabilty distribution originates from a linear layer, which uses the cls token of the last layer
- also: makes the encoders push as many information as possible to their cls token
- to first validate if regressing just the cls token of the teacher model even works with a teacher that has been trained with labels, we first
  use the exact same approach as before, i.e. keep the supervised teacher model, but regress the cls token of the teacher model now
- we do not need any linear classifier on top of our model, and we also do not need it from the teacher model
- we just take the cls token output of the last transformer layer of the teacher model as the target

- questions is whether the information contained in the cls token of the teacher, which has now been trained without lables, are abstract enought...

===== Modality-invariant Representations
- we want to learn a representation that is invariant/independent to the modality
  - optimally representation of an image should not change if we pass a caption (text) of the image through the model
- can be learned e.g. by contrastive learning, here image-text contrastive learning
- we are not only using contrastive learning, but also KD to align the representations of the image and text
- we regress the output of a teacher model, in the above case the probability distribution over the 1000 classes of imagenet
- because this forces the model to learn a representation which is more focused on real-world concepts/objects, like e.g. a cat,
  the focus is shifted away from the modality specific features
- this is actually what we want -> we do not want to regress any representation of probabilty distribution that still contains modality specific features/information
  - imagenet classes are not modality specific, therefore the probabilty distribution we regress does not contain modality specific information
    - a cat is a cat, no matter if we see it or read about it
- this is why the whole process works -> we can still regress the probabilty distribution of imagenet, even though we are processing text
  - e.g. a text about a cat should have the imagenet class "cat" as the highest probability
- generally, there has been no incentive for our unimodal teacher model to learn a representation that is independent of the modality
- however, labels are modality independent, so the ouput of the model, i.e. the logits become modality independent

- goal of this paper is to create a multimodal model from an unimodal one, with the constraint that the unimodal (teacher) model has not been trained on labeled data -> we do not want to rely on labeled data -> self-supervised
- the question is, if the model is unimodal, meaning there has been no incentive to learn a representation that is independent of the modality, and there
  were no labels involved that could push the model to learn representations independent of the modality, then can we still learn a representation that is
  modalitiy-invariant?
- this is crucial, as if the teacher image model, e.g. BEiT-2, outputs a cls token that still encodes image specific information, then the student text model
  will not be able to learn a meaningful representation of the text
- so what it boils down to is: Do unimodal, self-supervised trained, models learn representations abstract enough so that they can be regressed by models
  of a different modality?

==== Contrastive Learning with Memory Bank
- currently we do contrastive learning based on all samples of the current batch
  - for an image, its corresponding text is the positive example, all other 255 (we use batch size of 256) captions are negative examples
- generally, it is advised to use bigger batch sizes with contrastive learning -> increases number of negative examples
  - VLMo uses batch size of up to 32k, FLAVA 8192, CLIP also 32k
  - comparison between VLMo models trained with batch size of 1024 and 32k showed that this improves the model greatly
  - FLAVA mentioned it made their training more stable
- not feasible for us -> max batch size, with deepspeed stage 2, is 256, gpu memory is full
- we would still like to have more negative examples, while keeping batch size the same
- solution is memory bank
- Initially popularized by @memory_bank
- initially developed for self-supervised learning on images only
- stores the representations/embeddings of all samples of the dataset, this is the memory bank
- for each sample passed through the model, i.e. each sample in the batch, embedding is computed and cosine similarity to all embeddings in the memory bank, i.e. all samples of the dataset, is computed
  - softmax is then applied over the cosine similarities -> each image in the dataset, i.e. each sample in the memory bank, is seen as one class
  - for n images in the dataset, we have an n-class classification problem
- each time an image is padded through the model, the embedding is updated in the memory bank
- helps to increase the number of negative examples, without increasing the batch size

#figure(
  image("../figures/memory_bank_images.png"),
  caption: [Memory Bank],
) <memory_bank_images>

- disadvantage:
  - memory bank is very large for large datasets, e.g. ImageNet -> they report 600MB for 128 dimensional embeddings, we use 768
  - too much memory for our GPU setup
  - classification problem -> as in our contrastive loss, we do softmax over all samples/classes
    - will take long on large datasets
  - embeddings are only updated when their sample is passed through the model
    - for large datasets, some embeddings might be old and not of high quality

- we use a memory bank, but not based on the size of the whole dataset -> we want to "simulate" the contrastive loss of large batch sizes
- means memory bank of size similar to the batch size of e.g. VLMo -> 16k-32k
- should fit on GPU
- softmax over less classes, so relatively fast -> should not be the bottleneck operation
- not all samples of the dataset will be in the memory bank
- for each batch, we do contrastive loss over the batch and all samples in the memory bank
- we treat the memory bank as a FIFO queue
  - after contrastive loss has been computed for current batch, we add it to the memory bank and discard the oldest batch
- depending on the size of the memory bank, samples in the memory bank are more "fresh"

- during start of the training, we progressively fill the memory bank with batches
- during the first iterations/batches, memory bank will not be full
- we therefore only do the contrastive loss based on the current amount of batches in the memory bank
- after some steps, depending how large the memory bank is, the memory bank will be full, and we do contrastive loss with the whole memory bank
- from this point on, with each batch we will replace the oldest batch in the memory bank with the current batch

- we need two memory banks, one for images and one for text



Init results:
- for 16k -> model bad -> task prob too difficult
- for 1024 -> model starts well, then collapses -> lr too high (5e-4, but much lower in other itc papers)

==== Scaling Memory Bank

==== Feature Whitening
- as noted by \@feature_whitening, masked image modeling (MIM) usually ahead of contrastive learning
  - especially for downstream/finetuning tasks
- authors report increased performance, when distilling contrastive models using a special feature distillation, on downstream tasks
- fits our use case, as we already do distillation and contrastive learning
- higher performance on downstream task, like imagenet (zero-shot) classification is desirable
  -> as mentioned by FLAVA authors, multimodal model should not only perform well on multimodel tasks, but also on unimodal tasks

...

- they use feature map as targets, not logits, as some models do not have logits, i.e. probability distributions, as output/target to regress
- exactly what we are aiming for, as self-supervised models do not have logits as outputs, but feature (maps)

===== Adding Image-Text Contrast
- until now we did not actually used the same philosophy as in CLIP, which relies on, next to a seperate image and text encoder, a contrastive loss
  to align the representations of the image and text, so does not do KD and trains both text and image encoder from scratch
- as mentioned in the chapter about CLIP, the architecture features two linear projections, one for each modality/encoder
- goal is to project the image/text represenation in a shared multimodal embedding/latent space, on which the contrastive loss is computed
- if we also manage to do this successfully, the performance on image-text retrieval should increase by a margin


===== Stagewise Unimodal Distillation
== Seperate Self-Attention

=== Baseline

- currently only 6 layers, 5 out of which are modality specific, 1 is shared
- we experiment with adding one additional moadality specific layer, and one additional shared layer in another experiment
-> more difficult to align mutliple modalities, than just training one -> add one layer
  -> motivation for modality specific: after 5 layers information might not be high level enough so that one layer can process the information
    -> add one additional modality specific
  -> motivation for shared: after 5 layers information might be high level enough, but capturing modality agnostic information might take more than one layer
    -> add one additional shared

- added shared layer improves performance slightly, but adds 7 million parameters and 41 minutes to training time
- looking at the improvement in zero-shot, which increases the average Recall from 29.93% to 30.8%, this is not much of an improvement, 
  considering the amount of parameters we add to the model

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
    [FLAVA], [42.74], [76.76], [-], [38.38], [67.47], [-], [67.7], [94.0], [-], [65.22], [89.38], [-],
    [Data2Vec2], [0.02], [0.08], [0.22], [0.01], [0.10], [0.19], [0.02], [0.12], [0.26], [0.02], [0.06], [0.12],
    [*MM-D2V2 (Ours)*], [4.24], [12.12], [17.96], [1.77], [6.54], [10.91], [1.2], [4.88], [8.18], [0.54], [2.52], [4.58],
    [*MM-D2V2 (Ours)†*], [31.72], [56.78], [67.9], [12.42], [31.05], [42.5], [7.7], [26.18], [37.6], [4.08], [17.01], [24.26],
    [*MM-D2V2 7_2(Ours)†*], [32.78], [58.34], [69.3], [12.83], [31.85], [43.4], [8.08], [27.92], [38.6], [4.14], [17.5], [24.82],
    [*MM-D2V2 7(Ours)†*], [30.24], [56.48], [67.46], [11.96], [30.48], [41.88], [7.36], [26.42], [36.6], [3.7], [16.58], [23.84],
    table.hline(),
  ),
  caption: [Comparison of Zero-shot Image-Text and Text-Image Retrieval of first results with FLAVA and Data2Vec2 papers. Because Data2Vec2 is a unimodal model, we embed each image with the D2V2-Image model and each text with the D2V2-Text model. This yields unusable results, as there has been no incentive for the models to learn a shared representation, as both are unimodal. This is why we had to use both the image and the text model to embed the data. \ *†*: This version has been trained with BEiT-2 as the teacher model, not the D2V2 Image model.],
)<image_text_retrieval_1>


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
    table.cell([_Zero-Shot_], align: left), table.cell(colspan: 12, []),
    [FLAVA], [42.74], [76.76], [-], [38.38], [67.47], [-], [67.7], [94.0], [-], [65.22], [89.38], [-],
    [CLIP], [58.4], [81.5], [88.1], [37.8], [62.4], [72.2], [88.0],[98.7], [99.4], [68.7], [90.6], [95.2],
    [*MM-D2V2 (Ours)*], [31.72], [56.78], [67.9], [12.42], [31.05], [42.5], [7.7], [26.18], [37.6], [4.08], [17.01], [24.26],
    table.hline(stroke: .4pt),
    table.cell([_Finetune_], align: left), table.cell(colspan: 12, []),
    [BEiT-3], [84.8], [96.5],[98.3], [67.2], [87.7], [92.8], [98], [100], [100], [90.3], [98.7], [99.5],
    [VLMo], [74.8], [93.1], [96.9], [57.2], [82.6], [89.8], [92.3], [99.4], [99.9], [79.3], [95.7], [97.8],
    table.hline(),
  ),
  caption: [],
)<image_text_retrieval_2>

- looking at the validation loss of image and text seperatly, on COCO val set, we observe that the loss on images is
  significantly lower than the loss on text, which might be due to the fact that the teacher model is a vision model and
  the target, the cls token, might be biased towards the image modality, as it is unimodal
- interestingly, this bias also seem to be directly translated to the performance on image-text retrieval, as the performance
  on image-text retrieval is significantly higher than on text-image retrieval
  -> we are learning the cls token representation, and using the leared cls token as an output for the student model,
    to encode a modality for retrieval and other downstream task
-> suggests that the cls token is biased towards the image modality, or rather that the model is better in encoding images than
  text
- we can see that the performance of e.g. BEiT-3 and VLMo is also lower on text-image retrieval than on image-text retrieval,
  but not the the extend that we observe with our model

#figure(
  table(
  columns: (25%, auto, auto, auto, auto),
    stroke: none,
    table.hline(),
    table.header(
      table.cell(rowspan: 2, colspan: 1, align:horizon, [*Model*]),
      table.cell(colspan: 2, [*MSCOCO (5K test set)*]),
      table.cell(colspan: 2, [*Flickr30K (1K test set)*]),
      table.cell(colspan: 1, [Image $arrow.r$ Text]),
      table.cell(colspan: 1, [Text $arrow.r$ Image]),
      table.vline(stroke: .4pt),
      table.cell(colspan: 1, [Image $arrow.r$ Text]),
      table.cell(colspan: 1, [Text $arrow.r$ Image]),
    ),
    table.hline(stroke: .4pt),
    [*MM-D2V2 7(Ours)†*], [51.39], [28.11], [23.46], [14.71],
    [BEiT-3], [93.2], [82.57], [99.33], [96.17],
    [VLMo], [88.27], [76.53], [97.2], [90.93],
    table.hline(),
  ),
  caption: [Average recall of image-text and text-image retrieval on MSCOCO and Flickr30K. All models continously perform better on
  image-text retrieval than on text-image retrieval, but the difference is more pronounced for our model.],
)<image_text_retrieval_mean>

- currently vl layer(s) (or rather mulimodal layer(s)) are randomly initialized, one option is to specifically initialize the multimodal
  layers with the weight of the final layers of the D2V text model
  -> initial state is closer closer to text modality
  -> did not work

- currently embedding layer of text and patch embed layer of image model frozen, includes cls token of text and image frozen
- fuzed layer takes the cls token of the image model and the cls token of the text model as input
- so maybe unfreezing the embedding layers will help
-> did not work
- however, when we disable weight decay for image cls token and for text embedding layer, contains text cls/bos token,
  then we observed an increase in performance
- for all other params, weight decay stays at 0.01

- test adding image-mm and text-mm projections between encoders and shared encoder (FLAVA)
  - check retrieval performance when using ourputs of encoders, are they already a little bit aligned?
  - check retrieval performance when using outputs of projections, are they more aligned?
  - check retrieval performance when using outputs of shared encoder, are they even more aligned?
  - avg. cosine similarity for positive pairs and negative pairs


- we do not compare with See, Hear, and Read: Deep Aligned Representations// @shre, as they did not use the karpathy splits @karpathy_split,
  use the average median rank instead of recall at a specific percent, and from their experimental setup it is not clear which samples
  they used from Visual Genome for their retrieval experiments.

- what is interesting however, is that with just model transfer, which is Knowledge-Distillation in our case, their model did not perform well
  on zero-shot retrieval
  -> halfed score of linear regression
  -> especially for image-sound retrieval just model transfer, i.e. labeled variant of KD, did not work well
- what made the important difference, which might also be the case for us, is Contrastive Learning, which, with the exception of BEiT-3,
  was used by both VLMo, FLAVA as one of the pretraining tasks, and for CLIP it was the only pretraining task 

=== Image-Text Contrastive Learning
- solution is to also incorporate contrastive learning into our training
- as we still do KD, we now have two losses, the KD loss and the contrastive loss
  - nothing unusual, done by VLMo, FLAVA (use masked modality modeling as second pretraining task)
  - only contrastive loss done by CLIP
- how is it done in the papers? -> generally always the same
  - take the cls token of the text encoder output, and the cls token of the image encoder output
  - project each of them into a shared embedding space
  - compute the cosine similarity between the image embeddings and the text embeddings of the current batch
  - projection is done by linear layer, popularized by CLIP -> done the same across VLMo, FLAVA, CLIP, BEiT-3
- VLMo additionally takes the output of cls token of the whole model, not of the text and image encoder,
  and projects it into the shared embedding space with different linear layers
  - each model usually has two projection layers, one for the image encoder and one for the text encoder
  - VLMo has four projection layers, two additional for the cls token of the VL-expert
    - one if the output of the VL-expert is for text, and one if it is for image
  - is surprising, as the the VL-expert forces to learn a shared representation, so projecting it into a shared space
    with seperate projection layers seems counterintuitive, more intuitive would be to use one projection layer for the output
    of the cls token of the VL-expert
  - authors did not provide a reason for this
- therefore, we will start with the following:
  - seperate projection layers for image encoder and text encoder -> used to project image/text into multimodal space (FLAVA)
  - one projection layer for the cls token of the shared layer(s) -> used for contrastive learning
  - for unimodal finetuning: use output of the encoder without projection
  - for multimodal finetuning: use output of the shared layer without projection
  - for retrieval: use output of the projection layer

- why even use shared/fuzed layers, why not directly use the same approach as in CLIP?
  -> test the following: just train a text model with beit-2
-> (CLIP like) just train a text model to regress cls token output of final beit-2 layer
-> use blocks of d2v2 text to init text model (generally one could take any pretrained text model)
  -> freeze embedding layer + pos embedding layer -> has the advantage that max possible context stays 512 tokens,
    so no need to interpolate for downstream tasks, even if we now use just 64 tokens


  - have seperate projection layer for cls token of text encoder and image encoder
  - contrastive loss is done based on the cosine similarity of the projected cls tokens
  - output of the cls token of the shared multimodal layer(s) is ignored for now
  - also means that we use the projected cls tokens for retrieval, which is now not zero-shot anymore,
    as we explicitly train the model to maximize the cosine similarity between the cls tokens of an matching image-text pair
    - in that regard what FLAVA claims is not true, as they name their results on cross-modal retrieval for COCO and Flickr30K zero-shot,
      but pretrain their model using a contrastive loss
  - we have 5 encoder layers for each modality, and two shared layers
  -> means we now use 5 layers + projection for contrastive learning, and therefore for retrieval
  -> performance result will be interesting
- the question is in which case we will then utilize the output of the shared layers (cls tokens)
- for multimodal task we would use the output of the projections of the encoder's cls tokens
- for unimodal tasks we would use the output of the encoders without the projections

- therefore better option would be to use the cls token of the last layer output, which is a shared one, and project this into the shared space
- even though the representation should already be shared at this point -> so no projection necessary
- for modality specific tasks -> use output of the corresponding encoder
- for multimodal tasks -> use output of the shared layer

following options:
- no shared layers, seperate image and text encoders and two linear projections to shared space (CLIP)
- shared layers, seperate image and text encoders and one linear projection to contrast space
  - use output of shared layer for multimodal tasks, output of encoder for modality specific tasks
  - output of projection for retrieval

CLIP: no shared layers, seperate image and text encoders and two linear projections to shared space
FLAVA: shared layers, seperate image and text encoders, two linear projections to shared space for image and text encoder,
  two lin projections for image/text to multimodal space for mm encoder, on unimodal downstream classification tasks: use output of respective encoder
  withouth projection
- we do not do it exactly the same as in FLAVA, as our shared layers are not (yet) for referencing multimodal tasks, but for aligning 
  the modalities on the "concept" level, so use single projection layer for contrastive learning on this output
- also test exactly as in flava




  - have one projection layer for the final output of the cls token, stemming from the shared layer(s)

==== Contrastive Learning with Memory Bank

==== Decaying Memory Bank

=== Importance of the Teacher Model
- BEiT-2 vs. D2V2 Image shows significant difference in performance
- Model distilled from BEiT-2 teacher outperforms the one from D2V2 Image teacher by a large margin
- teacher model size is around the same -> both use ViT-B/16, BEiT-2 around one percent better on Imagenet-1k after finetuning
- too small of a difference that this could be the reason for the large difference in performance
- most likely the handling of the CLS token, which is regressed by our students, is the reason
  - D2V2 Image introduces special CLS loss to aggregate as much (global) information as possible
    - cls token regresses mean activations of all patches
    - was inspired by BEiT-2
  - BEiT-2 introduces a bottleneck to force the model to push as much information as possible towards the cls token
  - latter seems to be more effective
- which teacher to use does make a difference!


