== Discussion of Results <discussion_of_results>

In this section, we briefly discuss the performance on all the tasks that we use to evaluate the proposed method.
We compare the results on natural language processing, computer vision,
and vision-language tasks in @result_overview.

In most cases, the proposed method does not outperform the methods to which we compare. However, recall that our approach
was designed as a *proof-of-concept* to demonstrate the feasibility of creating cost-effective and efficient vision-language models.
Since we achieve reasonable performance across all tasks, and even outperforming well-known baselines on tasks like WNLI
and COCO image retrieval, we consider our approach to be successful.

#figure(
  image(
  width: 100%,
  "../figures/result_overview.png"),
  caption: [
    Overview of all benchmark results across natural language processing (top left), computer vision (top right),
    and vision-language tasks (bottom).
],
) <result_overview> 

Furthermore, it is not realistic to expect the proposed method to outperform the state-of-the-art methods, as they are
larger in every aspect: parameters, data, and compute. A good impression on where S-SMKE ranks among the
vision-language models we repeatedly compare to can be obtained from @vl_models_rank_overview.

Our most noteable achievements are:
- *WNLI Finetuning:* We outperform BERT @bert and DistilBERT @distilbert (by just 0.1 percentage point) on the WNLI task, which is likely
  due to the fact that it consists of just 635 training and 71 validation examples @wnli. Our text-only model is around
  half the size of BERT, making it less prone to overfitting on such a small dataset.
- *CIFAR-10 and CIFAR-100 Linear Evaluation:* Our image-only distilled model,
  DistilData2Vec2, is almost on par with BEiTv2 @beitv2
  on CIFAR-10 and CIFAR-100 @cifar_10_100 linear evaluation. This is remarkable, as BEiTv2 is also twice as large as
  our model. We consider the results on both benchmarks (CIFAR-10 and CIFAR-100) as reliable, as we performed the linear
  evaluation of BEiTv2 ourselves, using the exact same setup as for DistilData2Vec2.
- *COCO Image Retrieval:* S-SMKE outperforms CLIP @clip on COCO image retrieval, and the finetuned variant of S-SMKE 
  (indicated by "S-SMKE *$dagger$*") outperforms CLIP
  on all COCO retrieval metrics. It has to be noted that CLIP was neither pretrained nor finetuned on the COCO dataset,
  so S-SMKE has an advantage here. However, considering that CLIP is a much larger model trained with over 121$times$ the data,
  the result is still remarkable.

#figure(
  image(
  width: 50%,
  "../figures/vl_models_rank_overview.png"),
  caption: [
    Overview of vision-language model landscape. Bubble sizes represent the number of parameters (also shown in
    parentheses next to the model name). The number of image-text pairs used is in log scale, and the retrieval
    performance is the average of the R@1 scores on COCO @coco and Flickr30k @flickr30k. We denote red bubbles as
    the direct application of a pretrained model on image-text retrieval, and blue bubbles as models that were
    finetuned on the retrieval task after pretraining. Visual N-Grams @visual_n_grams was developed in 2017
    as a first proof-of-concept for vision-language models.
],
) <vl_models_rank_overview> 

On unimodal tasks, S-SMKE does not perform as well as our unimodal distilled models and other baselines. As mentioned
in the limitations of our method (@problem_fine_tuning_unimodal_tasks), the representations of our image and text encoder
are optimized for alignment, not modality-specific tasks. A loss in performance is therefore to be expected but not necessarily
unavoidable, since FLAVA @flava shows that including text-specific objectives during pretraining
can greatly improve performance when finetuning on text tasks. The same observation holds BEiT-3 @beit3 on vision tasks
(see @problem_fine_tuning_unimodal_tasks).

Considering the above, it is still striking that S-SMKE only reaches 14.2% accuracy on CoLA @cola, while our text-only model
F-DistilBERT reaches 55.1%, which is just 1.2 percentage points below BERT's @bert 56.3%. Unfortunately, we do not have
an explanation for this discrepancy. One possible factor is that we pretrain S-SMKE
on subsets of Conceptual Captions 3M and 12M @cc3m @cc12m. Since these datasets are largely uncurated and scraped from the web,
the captions passed through the text encoder might not be grammatically correct. As CoLA is a
task to assess the grammatical understanding of a model @cola, finetuning on a dataset with just 8551 samples might not be sufficient
to correct the model's understanding of grammar. An example of poor captioning in Conceptual Captions 12M can be seen in
@coco_vs_cc12m.

// #figure(
//   image(
//   width: 50%,
//   "../figures/polar_glue_results.png"),
//   caption: [
    
// ],
// ) <polar_glue_results> 

// #figure(
//   image(
//   width: 50%,
//   "../figures/polar_vision_results.png"),
//   caption: [
    
// ],
// ) <polar_vision_results> 

// #figure(
//   image(
//   width: 50%,
//   "../figures/polar_vl_results.png"),
//   caption: [
    
// ],
// ) <polar_vl_results>
