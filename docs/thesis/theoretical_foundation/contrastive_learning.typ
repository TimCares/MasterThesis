#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

=== Contrastive Learning <contrastive_learning_section>
In settings where masking discrete tokens and predicting them based on a set of possible tokens, as in language models,
is not possible, contrastive learning can be used as a self-supervised method.
This is especially useful in vision models, as images are continuous, so there is no discrete set of possible tokens to predict.

Contrastive learning, or the contrastive loss, is a method
to learn representations of data without the need for labels, and used in computer vision models like MoCo @moco, SimCLR @simclr,
and CLIP @clip.

In computer vision, contrastive learning exploits the fact that the high-level semantics of an image are invariant to
small (or moderate) changes in pixel-level information. This is achieved by augmenting the input image, e.g., by cropping,
rotating, or flipping it. Provided the augmentation is not too drastic (e.g., crop size too large),
the high-level semantics of the image will remain the same after augmentation, even though pixel-level information do not.
The goal of the image model is then to maximize the cosine similarity between the global representations of two 
augmented versions of the same image. In Transformers, the global representation is usually the $mono(["CLS"])$ token retuned
by the final layer of the model, which is a vector that can be compared with the $mono(["CLS"])$ token of another image
using the cosine similarity. 
The augmented versions are often referred to as a different _view_ of the same image @simsiam, as shown in @image_views.

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
content as the original image, and the cosine similarity between the original image and these negative samples should therefore be minimized
(a cosine similarity of 0 indicates no similarity between the input vectors).
This prevents the model from collapsing to a constant representation, as it would not minimize the cosine similarity
and thus not minimize the loss. A simple yet expressive visualization can be found in @simclr_vis.
This makes self-supervised training of image models possible, and the learned representations represent the high-level semantics
of the images, learned without the need for labels.

An implementation and mathematical formulation of the contrastive loss will be introduced in (TODO: cite vision language contrast)
on the example of vision-language models.

#bibliography("../references.bib")
