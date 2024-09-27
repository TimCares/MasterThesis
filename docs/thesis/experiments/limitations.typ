=== Limitations and Insights <mm_kd_limitations>
While the proposed method is an efficient way to traing comparatively small multimodal models, and can easily be
adapted to other modalities, e.g. audio, it has two main limitations.

+ terrible on unimodal downstream tasks

First, our method relies on knowledge distillation of a self-supervised image model as the teacher. The fact that there
has been no incentive for the teacher to learn a representation that is independent of the image modality makes it difficult
to learn a representation that is truly modality-invariant and aligned across the modalities of the student model. This
has repeatedly been shown when comparing the loss between image-to-image and text-to-image distillation, where the former
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

A glance at the comparison between the components of the knowledge distillation loss (KL-Divergence) when using a supervised teacher 
also shows that this approach suffers from the same problem as when using a self-supervised teacher
(see @kd_loss_shre_vs_ssmke). Here, the KL-Divergence
for the image-to-image loss $cal(L)^v_"KD"$ is also consistently lower than for the text-to-image loss $cal(L)^w_"KD"$, and the loss 
components between both approaches (Transformer SHRe and S-SMKE) generally perform very similar.

We conclude that using an unimodal teacher in general is a limitation regarding alignment, and therefore performance. Still,
due to the fact that it does not require a multimodal teacher, the approach is far more flexible and can be applied to any
modality, as long as a suitable teacher is available.

Second, S-SMKE (and (Transformer) SHRe) processes image and text seperately, in a forward pass for the image and a forward pass
for the text. This is similar to CLIP @clip (see @clip_section). Because there is no attention mechanism between
individual image patches and text tokens, both approaches miss a fine-grained alignment between the modalities.
Even though our model performs quite well on the retrieval task, even outperforming well-established research papers
on some metrics, there is still room for improvement. However, we believe that there is not much more performance to gain
for our approach. Wrong retrievals are often still similar to the query, and only differ in little token-level details.
Since our representations are based on the global content of the image and text, those details are not captured in most
cases, leading to a "false" retrieval. Examples of this can be seen with example retrievals on the full MSCOCO test set
in @coco25k_retrieval_examples,
which is the exact dataset we publish our results on (previous visualizations are based on 1k subsets as in SHRe @shre,
and are therefore a simpler task since there are less possible retrieval candidates). 

At this point it has to be noted that when it comes to a production-grade retrieval system, e.g. a system for searching images by text,
the retrievals of our model
can still be considered as correct, or "good enough", as most of them, even though they are flagged as "false" under
the evaluation metrics, are still semantically related to the query.

Still, the problem remains, and is actually relevant when it comes to applications where the fine-grained details are crucial.
This includes tasks like visual question answering and visual reasoning, benchmarked by the datasets VQAv2 @vqav2 and
NLVR2 @nlvr2, respectively.

For instance, in NLVR2, the task is to decide whether a given sentence about an image is either true or false.
The sentence, so the statement about the image, often focuses on fine-grained details, for which there is no guarantee
that they are captured just by global representations. Furthermore, the statement can often not even be considered as
a caption, as is can also be a false statement about the image. Examples of this can be seen in @nlvr2_examples.

That simply aligning global representations is not enough to excel in such tasks was also shown by the authors
of ViLT @vilt, who were one of the first to propose a model that aligns image and text on a fine-grained level.
They found that when finetuning CLIP @clip on the NLVR2 dataset, using the dot-product of the global representations
of image and statement as the binary classifier, the model achieved an accuracy of only 51% @vilt. Considering that the task
is a binary classification, and both classes are equally distributed, this is only slightly better than random guessing
and therefore unusable for practical applications. Since our model works on a similar level of alignment, the same
will likely apply with our approach.