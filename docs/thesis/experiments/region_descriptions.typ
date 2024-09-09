=== Enhancing Alignment <enhancing_alignment>
==== Region Descriptions for Contrastive Learning <region_descriptions_for_contrastive_learning>
Many papers we compare to, namely VLMo @vlmo, FLAVA @flava, and BEiT-3 @beit3, use Visual Genome @vg as one of the
datasets to train their Vision-Language models. Developed for a variety of descriptive visual tasks, like visual question answering,
it also contains images with so-called region descriptions. These are human annotated and highly curated descriptions of small regions
in an image, often focusing on specific objects or parts of the image @vg. This makes them an attractive source for training vision-language
models, as they provide a more fine-grained description of the image content than the global image caption, while still being abstract
enough to accurately describe real-world objects and scenes.

Because the dataset is so highly curated, and the region descriptions often only cover a small part of the image, each image in the dataset
has up to 50 region descriptions. Given that the dataset contains 108,077 images, and each region description can be considered as a seperate
image-text pair, this results in a quite large dataset of 5.4M image-text pairs. Since the number of images is also relatively small,
it does not require as much disk space as the other datasets we use, like CC3M @cc3m or CC12M @cc12m. All of this makes Visual Genome an
attractive source for our experiments, however, as seen in our overview of the datasets we use in @vl_dataset_summary, we do not use it
throughout this work.

This is because we encountered an unstable training behavior when using Visual Genome in combination with contrastive
learning, as shown in @itc_vg. As a side note, even though it looks like that without Visual Genome the accuracy saturates at around 70%
after the first 6k steps, this is not the case: @itc_vg_full shows that the accuracy of the contrastive loss continues to increase
with more training steps. We did not however observe this behavior when using Visual Genome.

#figure(
  image("../figures/itc_vg.png"),
  caption: [Training accuracy of Image-Text Contrast with Visual Genome (left) vs. without Visual Genome (right).
  Removing Visual Genome from the training data leads to a more stable training. Even though we only show the
  first 6k steps, this behavior continues throughout training.],
) <itc_vg>

We assume that the reason for this behavior is that the region descriptions are too specific, i.e. they focus on parts too small to capture
the overall content of the image, and generally shorter than those of COCO, CC3M, and CC12M:
While the data we collect has a mean caption length of 11.1 tokens
(see @vl_dataset_summary), the region descriptions of Visual Genome only have a mean length of 4.7 tokens.
Since our alignment of image and text is based on the global information carried by the image and the caption,
too specific or too short descriptions might not be helpful, but rather confuse the model.

The reason why this works for other models, like VLMo, FLAVA, or BEiT-3, is because they use a concatenation of the image sequence and the
text sequence as input to the shared encoder @flava @vlmo @beit3. This approach itself allowes for a more fine-grained alignment of image and text, where
individual text tokens can attend to individual image patches, and vice versa. Therefore, even descriptions of small regions can be helpful
for these models.

Furthermore, larger models, like the VLMo's ViT-L/16 variant @vlmo, have empirically shown to be robust to noisy image-text pairs. Since
we "only" perform global image-text alignment, the fine-grained captions of Visual Genome can be considered as noisy, as they not always 
capture the global semantics of their image. This, and the fact that our model is relatively small compared
to models like VLMo or BEiT-3, might also be a reason why we encounter this unstable training behavior when using Visual Genome.



#figure(
  image("../figures/itc_vg_full.png", width: 50%),
  caption: [
    Visualizing the training accuracy of image-text contrast without Visual Genome shows that the accuracy continues to increase
    with longer training. This is in contrast to the training with Visual Genome, where the accuracy saturates after the first 6k steps.
  ],
) <itc_vg_full>

