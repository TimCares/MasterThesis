== Figures and Visualizations

#figure(
  image(
  width: 75%,
  "../figures/norm_comparison.png"),
  caption: [Comparison of different normalization operations on the example of images.
  Dimension "H, W" refers to the height and width of the input, "C" to the number of channels or
  embedding dimensions, and "N" to the number of samples, i.e. the batch dimension. Since we are working with sequences of embeddings, the height and width
  dimension correspond to the time steps ("H, W" -> "T"). The normalization operations work correspondingly on text sequences, where we also have time steps,
  so dimension "H, W" can also be replaced by "T" @group_norm. Please note that group norm, even though it is displayed in bold
  , is not used in this work. The figure merely was introduced in the paper of Group Norm @group_norm.
],
) <norm_comparison>

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