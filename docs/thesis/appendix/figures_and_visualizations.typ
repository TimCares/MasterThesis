== Figures and Visualizations

#figure(
  image(
  width: 75%,
  "../figures/clip.png"),
  caption: [Illustration of CLIP training. A batch of image-text pairs is passed through the model and embedded into a shared latent space.
  The cosine similarity between all pairs is computed and softmax-normalized to calculate the image-to-text and text-to-image loss. The final loss
  is the mean of both losses @clip.
  The example is shown with a batch size of 6. The figure does not originate from the original paper, but is a custom visualization of the concept. Image-Text pair is taken from the MSCOCO train set @coco, 
  and do not refer to the contrastive loss of 6 pairs at the top of the figure. They are merely indicators of the intput to the model.],
) <clip_fig>


#bibliography("../references.bib")