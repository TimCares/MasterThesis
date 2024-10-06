== Limitations and Insights <mm_kd_limitations>
While the proposed method is an efficient way to train comparatively small multimodal models, and can easily be
adapted to other modalities, e.g. audio, it has its limitations.

=== Performance on Unimodal Downstream Tasks <problem_fine_tuning_unimodal_tasks>
When finetuning S-SMKE on unimodal downstream tasks in @fine_tuning_unimodal_tasks, we found that the performance
is generally worse compared to the performance of the unimodal distilled models. While this is nothing unusual,
it is still a limitation considering that the goal of multimodal models is to excel at both unimodal and multimodal tasks.

Fortunately, this can be solved by not only training the multimodal model on aligning the modalities, but simultaneously
training the modality-specific encoders, in our case the image and text encoder, on modality-specific pretraining tasks.
A good example of this is the approach of BEiT-3 @beit3, which trains the whole model on alignment of image and text,
but also trains the image encoder to reconstruct masked images, and the text encoder to predict masked tokens. This way,
both encoders are encouraged to learn representations that are still specific to their modality, while providing representations
that can also be used for multimodal tasks and therefore alignment.
The result is that BEiT-3 reaches a finetuning accuracy of 85.4%
#footnote[This score was not published in the original paper, but can be found in the official BEiT-3 repository:
#link("https://github.com/microsoft/unilm/tree/master/beit3").]
on ImageNet-1K using its image encoder. This is better than
the performance of BEiTv2, which is an image-only model and reaches a finetuning accuracy of 85.0% @beitv2.
Since both the image encoder of BEiT-3 and BEiTv2 are based on the same ViT-B/16 @vit architecture, the results are directly comparable
and show that modality-specific tasks can strengthen the performance of multimodal models on unimodal tasks.

=== Modality-Specific Bias
Our method relies on knowledge distillation of a self-supervised *unimodal* image model as the teacher. The fact that there
has been no incentive for the teacher to learn a representation that is independent of the image modality makes it difficult
for the teacher to provide guidance to the student model on how to align the modalities. This is because the teacher representations are
not aligned and modality-agnostic themselves.
This has repeatedly been shown when comparing the loss between image-to-image and text-to-image distillation, where the former
is consistently lower. Interestingly, we were still able to outperform the approach of a supervised teacher, showing that
even though the ImageNet-1K classes, which we predict using KL-Divergence (see @transformer_shre), are real-world concepts independent
of the image modality, they might not capture the content of an image's caption better than
a image-specific representation (used with the self-supervised teacher), which is what we first assumed.

#figure(
  image(
  width: 100%,
  "../figures/kd_loss_shre_vs_ssmke.png"),
  caption: [
    Both approaches, the supervised as well as the self-supervised (ours), show that the image component is
    consistently lower than the text component of the knowledge distillation loss. Moreover, the loss behavior
    troughout the training process is very similar between both approaches, although the loss functions are different.
],
) <kd_loss_shre_vs_ssmke>

A glance at the comparison between the components of the knowledge distillation loss when using a supervised teacher 
shows that this approach suffers from the same problem as the self-supervised teacher
(see @kd_loss_shre_vs_ssmke). Here, the KL-Divergence
for the image-to-image loss $cal(L)^v_"KD"$ is also consistently lower than for the text-to-image loss $cal(L)^w_"KD"$, and the loss 
components between both approaches (Transformer SHRe and S-SMKE) generally perform very similar.

Consequently, a bias towards the teacher modality is not specific to S-SMKE, but generally a problem
when using an unimodal teacher for distilling knowledge to a multimodal model.

=== Fine-Grained Alignment
S-SMKE (and SHRe) processes image and text seperately, by performing a forward pass for the image and a forward pass
for the text. This is similar to CLIP @clip (see @clip_section). Because there is no attention mechanism between
individual image patches and text tokens both approaches miss a fine-grained alignment between the modalities.
Even though our model performs quite well on the retrieval task, even outperforming well-established research papers
on some metrics, there is still room for improvement. However, we believe that there is not much more performance to gain
for our approach. Wrong retrievals (mostly wrong image retrievals) are often still semantically
similar to the query, and only differ in little (token-level) details.
Since our representations are based on the global content of the image and text, those details are not captured in most
cases, leading to a "false" retrieval. Examples of this can be seen with retrievals on the full MSCOCO test set
in @coco25k_retrieval_examples,
which is the exact dataset we publish our retrieval results on (previous visualizations are based on the 1k subsets of SHRe @shre,
and are therefore a simpler task since there are less possible retrieval candidates). 

At this point it has to be noted that when it comes to a production-grade retrieval system, e.g. a system for searching images by text,
the retrievals of our model
can still be considered as correct, or "good enough", as most of them, even though they are flagged as "false" under
the evaluation metrics, are still semantically related to the query.

Still, the problem remains, and is actually relevant when it comes to applications where the fine-grained details are crucial.
This includes tasks like visual question answering and visual reasoning, benchmarked by the datasets VQAv2 @vqav2 and
NLVR2 @nlvr2, respectively.

For instance, in NLVR2 the task is to decide whether a given sentence about an image is either true or false.
The sentence, so the statement about the image, often focuses on fine-grained details for which there is no guarantee
that they are captured by just global representations. Further,
even if statements are false, then the differences between the image and the statement are often so subtle that the statement,
on a high level,
can still be considered as a caption for the image. Therefore, if say the cosine similarity between the global representations
of image and statement would be used as the binary classifier, then the model would likely fail on such tasks.
Examples of NLVR2 can be seen in @nlvr2_examples.

That simply aligning global representations is not enough to excel in such tasks was also shown by the authors
of ViLT @vilt, who were one of the first to propose a model that aligns image and text on a fine-grained level.
They found that when finetuning CLIP @clip on the NLVR2 dataset the model achieved an accuracy of only 51% @vilt. Considering that the task
is a binary classification, and both classes are equally distributed, this is only slightly better than random guessing
and therefore unusable for practical applications. Since our model works on a similar level of alignment, the same
will likely apply with our approach.
