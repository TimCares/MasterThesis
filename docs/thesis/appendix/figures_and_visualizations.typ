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

#show table: set text(8pt)
#figure(
  table(
    columns: 3,
    stroke: none,
    table.hline(),
    table.header(
      [*Dataset*],
      [*Example*],
      [*Label*],
    ),
    table.hline(stroke: .4pt),
    [CoLA], [[CLS] Our friends won't buy this analysis, let alone the next one we propose. [SEP]], [1],
    [SST-2], [[CLS] hide new secretions from the parental units [SEP]], [0],
    [MRPC], [[CLS] Amrozi accused his brother, whom he called "the witness", of deliberately distorting his evidence. [SEP] Referring to him as only "the witness", Amrozi accused his brother of deliberately distorting his evidence. [SEP]], [1],
    [STS-B], [[CLS] A plane is taking off. [SEP] An air plane is taking off. [SEP]], [5.0],
    [QQP], [[CLS] How is the life of a math student? Could you describe your own experiences? [SEP] Which level of prepration is enough for the exam "jlpt5"? [SEP]], [0],
    [MNLI], [[CLS] Conceptually cream skimming has two basic dimensions - product and geography. [SEP] Product and geography are what make cream skimming work. [SEP]], [1],
    [QNLI], [[CLS] When did the third Digimon series begin? [SEP] Unlike the two seasons before it and most of the seasons that followed, Digimon Tamers takes a darker and more realistic approach to its story featuring Digimon who do not reincarnate after their deaths and more complex character development in the original Japanese. [SEP]], [1],
    [RTE], [[CLS] No Weapons of Mass Destruction Found in Iraq Yet. [SEP] Weapons of Mass Destruction Found in Iraq. [SEP]], [1],
    [WNLI], [[CLS] I stuck a pin through a carrot. When I pulled the pin out, it had a hole. [SEP] The carrot had a hole. [SEP]], [1],
  ),
  caption: [Training examples of the GLUE benchmark tasks (one example per task). Examples are taking from the GLUE dataset card on Hugging Face
  #footnote[#link("https://huggingface.co/datasets/nyu-mll/glue")].],
)<glue_example>
#show table: set text(11pt)

#figure(
  image(
  width: 75%,
  "../figures/vl_crop_size_bad.png"),
  caption: [A small lower bound (8%) for a random crop erases high-level semantic features which are important for aligning text and image.
  Image-text pairs have been taken from the COCO train set @coco.
],
) <vl_crop_size_bad>

#figure(
  image(
  width: 75%,
  "../figures/vl_crop_size_good.png"),
  caption: [A larger lower bound (90%) for a random crop retains high-level semantic features. Notice how this is not the case for very low
  values, as shown in @vl_crop_size_bad.
  Image-text pairs have been taken from the COCO train set @coco.
],
) <vl_crop_size_good>

#figure(
  image(
  width: 100%,
  "../figures/shre_coco_prob_dist.png"),
  caption: [
  Visualization of the predicted probabilities for the top-5 ImageNet-1K @imagenet classes on image-text pairs from the COCO train set @coco.
  While the predicted classes are not always correct, e.g. bottom right, they are able to capture to some extend the semantic content
  of the image, and even the text. The latter is crucial for the approach of SHRe @shre.
  Note: The figure does not stem from the SHRe paper @shre, but is a custom visualization of the concept. However, it is inspired by
  the CLIP paper @clip.
  The ResNet-50-A1 @resnet_50_a1 model is used for the predictions.
],
) <shre_coco_prob_dist>

#figure(
  table(
    columns: 2,
    stroke: none,
    table.hline(),
    table.header(
      [*Approach*],
      [*\# Image-Text pairs*],
      table.hline(stroke: .6pt),
    ),
    [FLAVA @flava], [70M],
    [CLIP @clip], [400M], 
    [VLMo @vlmo], [4M/1B],
    [CoCa @coca], [>3B],
    [BEiT-3 @beit3], [21M],
    table.hline(stroke: .6pt),
    [This work], [3.3M],
    table.hline(),
  ),
  caption: [A comparison of the number of image-text pairs used for pretraining in different approaches. We use significantly
  fewer pairs compared to other approaches.
  We compare the 1B variant of VLMo @vlmo to our models,
  and compare the cost of training our final model to the cost of
  VLMo with 4M pairs. This is because the authors of
  VLMo did not publish the compute used for their 1B variant. 
  ],
)<models_data_size_comparison>

#figure(
  image(
  width: 75%,
  "../figures/ddp_visualization.png"),
  caption: [Distributed Data Parallel allows training a model with larger batch sizes than possible on a single GPU. The actual batch
  size per GPU does not change, but gradient updates are synchronized across GPUs, leading to weight updates that are equivalent
  to a larger batch size @pytorch_ddp.
],
) <ddp_illustration>

#figure(
  image(
  width: 75%,
  "../figures/shre_transformer_only_ir_1k_crop.png"),
  caption: [
    Results of image retrieval on a 1k subset of the COCO test set @coco, as described in @shre.
],
) <shre_transformer_only_ir_1k>

#figure(
  image(
  width: 75%,
  "../figures/shre_transformer_only_tr_1k_crop.png"),
  caption: [
  Results of text/caption retrieval on a 1k subset of the COCO test set @coco, as described in @shre.
],
) <shre_transformer_only_tr_1k>

#figure(
  image(
  width: 100%,
  "../figures/filip_cmli_examples.png"),
  caption: [
  Cross-Model Late Interaction (CMLI) on FILIP @filip and CLIP @clip. Numbers in parentheses describe the index of the text token
  in the text sequence, and numbers in patches to which text token a patch is matched.
  While FILIP is able to achieve a localization of the correct image patches, CLIP fails to do so. The figure is directly taken
  from FILIP @filip.
],
) <filip_cmli_examples> 

#figure(
  image(
  width: 100%,
  "../figures/dino_cls_attn_examples.png"),
  caption: [
    Fine-grained DINO @dino self-attention map of the $mono(["I_CLS"])$ token on image-text pairs of the COCO test set @coco.
    Heatmap is average over all attention heads.
],
) <dino_cls_attn_examples>

// #figure(
//   image(
//   width: 75%,
//   "../figures/rand_mask_ex.png"),
//   caption: [
//     Examples of random masking applied to images from the COCO test set @coco. The masking probability is set to $p=0.9$.
// ],
// ) <rand_mask_ex>

// #figure(
//   image(
//   width: 100%,
//   "../figures/image_vq_8_examples.png"),
//   caption: [
//     Members of codebook embeddings generated by VQ-1024-8 on the ImageNet-1K validation set @imagenet. The index of the codebook
//     embeddings are shown left to the images.
//   ],
// ) <image_vq_8_examples>

// #figure(
//   image(
//   width: 100%,
//   "../figures/image_vq_examples.png"),
//   caption: [
//     Members of codebook embeddings generated by VQ-1024-16 on the ImageNet-1K validation set @imagenet. The index of the codebook
//     embeddings are shown left to the images.
//   ],
// ) <image_vq_examples>

#figure(
  image(
  width: 100%,
  "../figures/coco25k_retrieval_examples.png"),
  caption: [
    Example text retrievals on the full COCO test set @coco with 25k images.
    Green and red boxes indicate correct and incorrect retrievals, respectively. Since each image
    has multiple captions, there are multiple correct retrievals for each image.
],
) <coco25k_retrieval_examples> 

#figure(
  image(
  width: 100%,
  "../figures/nlvr2_examples.png"),
  caption: [
    Example images with corresponding statements from the NLVR2 dataset @nlvr2.
    Green and red boxes at the bottom right of each example indicate a correct and incorrect statement, respectively.
    The examples are taken from the
    official NLVR2 website#footnote[#link("https://lil.nlp.cornell.edu/nlvr/nlvr.html")].
],
) <nlvr2_examples> 

#figure(
  image(
  width: 100%,
  "../figures/ev_lp.png"),
  caption: [
    Visualization of the overall framework of S-SMKE. The teacher remains frozen during training, indicated by the snowflakes.
],
) <ev_lp> 
