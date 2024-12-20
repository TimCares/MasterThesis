=== Increasing Negative Examples for ITC <memory_bank_section>

Our current approach utilizes Distributed Data Parallel (DDP) @pytorch_ddp with a batch size of 256 per GPU.
With two GPUs, the combined batch size is 512 and image/text features are gathered from all devices to increase the number of negative examples,
as described in VLMo @vlmo. This method, as demonstrated in @larger_batch_sizes_ddp, improves performance.

However, implementations of image-text models that leverage contrastive learning typically use much larger batch sizes, and therefore have much
more negative examples available. For instance, CLIP uses a batch size of 32,768 @clip. Achieving such large batch sizes would require more GPUs, as
batch sizes exceeding 256 (per device) lead to out-of-memory (OOM) errors on the GPU (NVIDIA RTX 4090).

// Although adding more GPUs to our setup is costly, the training time will be reduced proportionally to
// the number of GPUs used. For example, using two GPUs instead of one halves the training time. This is
// because DDP shades the whole training dataset between all devices, such that each device processes a
// different part of the dataset @pytorch_ddp (already mentioned in @larger_batch_sizes_ddp about DDP).
// From a cost perspective, this is acceptable, however, we decide against this approach, because we
// want to keep the number of GPUs manageable. Moreover, going with more GPUs would make the success
// of our approach dependent on the number of GPUs one has available, which is why we search for
// alternatives that allow us to increase the number of negative samples without requiring more GPUs.

One intuitive approach to increase the number of negative examples without a larger batch size is to use a queue-based memory bank.
The queue stores image and text embeddings, i.e. tokens $mono(["I_CLS"])$ and $mono(["T_CLS"])$,
from previous batches. With each new batch, the memory bank is updated
by adding the current batch embeddings and discarding those of the oldest batch.
These stored embeddings, combined with the embeddings of the current batch, are then used as negative examples.

Given a batch size of $B$ and a memory bank size of $U$, we can achieve $B + U - 1$ negative samples.
For instance, with a batch size of 256 and a memory bank size of 768, we can attain 1023 negative samples.
This configuration effectively simulates a contrastive loss with a batch size of 1024, and is comparable
to four GPUs with DDP and a batch size of 256 per GPU. However, it is important to note that this “simulation”
of larger batch sizes with a memory bank only applies to the number of negative examples:
The actual gradients are still computed using the effective batch size of 512 (... because we still use the
double GPU setup with a batch size of 256 per GPU).

Implementing this approach actually requires to maintain two distinct memory banks.
This is because we need to store old image representations (1), and old text representations (2).

Illustrated in @mb_results, we observed a significant drop in performance, and the resulting model is not usable. The accuracy on ImageNet-1K is 0.001%, and therefore corresponds to a random classification (1000 classes).
This suggests that the model did not learn anything useful, so simply increasing the number of negative examples via a memory bank does not help in learning richer representations.

#figure(
  table(
    columns: 4,
    stroke: none,
    table.hline(),
    table.header(
      table.cell(rowspan: 2, colspan: 1, align:horizon, [Memory Bank]),
      table.cell(rowspan: 2, colspan: 1, align:horizon, [ImageNet-1K]),
      table.cell(colspan: 2, align:horizon, [Retrieval]),
      [MSCOCO],
      [Flickr30K],
    ),
    table.hline(stroke: .6pt),
    [$times$], [*30.3*], [*67.2*], [*78.29*],
    [$checkmark$], [0.001], [0.12], [0.58],
    table.hline(),
  ),
  caption: [
    Using a memory bank to store negative examples leads to unusable results. Especially noticeable is an accuracy of 0.001% on ImageNet-1K,
    which corresponds to a random classification.
  ],
)<mb_results>

We suspect this drop in performance originates because of the following reasons:
When using an actual batch size of 512 without a memory bank all negative examples are generated
during the same step, and therefore by the same model with the same weights. This is the case even with DDP, as all model replicas
are synchronized, i.e. share the same weights.
Therefore, the representations share the same latent space and are consistent with each other. A similarity measure, in our
case the cosine similarity, can then provide the model with a meaning of distance between these,
in our case image and text, representations.

However, when using a memory bank most negative examples come from previous batches that were stored in the memory bank.
As the model's weights constantly change, especially at the beginning of training, there is a continuous shift in the representation space.
This shift is so pronounced that even representations from the immediate previous steps differ significantly from the current representations,
and a similarity measure will not provide meaningful information to the model.
To demonstrate, say we generate a representation
$bold(h)_(v, K, mono(["I_CLS"]))$ for an image, and store it in the memory bank. If we generate a new representation for the same image
in the next step, then both representations will not be the same, because the weights with which the representations have been generated
were not the same. Consequently, the cosine similarity will not be at its maximum value of 1,
even though it is the same image. 
It follows that the representations stored in the memory bank are not consistent with the representations generated in the current
training step, and comparing them with cosine similarity does not provide a meaningful measure of similarity to the model.

This can be thought of as a less extreme case of comparing the representations of an image-only and text-only model,
which are not associated with each other. In the beginning of our experiments,
we tested image-text retrieval with the Data2Vec2 image and text model (see @image_text_retrieval_shre_transformer_first),
and observed that
this approach is ineffective for image-text retrieval. The representations produced by both models do not have any
relationship with each other, and therefore the cosine similarity does not provide any meaningful information to the model.

While this effect is less pronounced with a memory bank, as the representations are still generated by the same model,
the shift in the model's weights is still significant enough to make the representations inconsistent with each other.


===== Relation to Initial Memory Bank Approach <original_memory_bank>
The memory bank was initially introduced by @memory_bank as a mapping of the complete training dataset:
The embedding of each sample in the dataset is stored in the memory bank (illustrated in @memory_bank_images).
For each batch/step, $K$ samples are randomly drawn from the memory bank to be used as negative examples.
The representations of samples in the current batch are then updated in the memory bank @memory_bank.
This approach is similar to ours, but faces the same problem: The representations come from an older variant of the model with different weights.
Even worse, the representation of an example in the dataset is updated only when it was last seen in a batch, which can be a long time ago for large datasets. However, the authors mitigate this issue by using proximal regularization, as shown in @proximal_regularization.

#figure(
  image(
  "../figures/memory_bank_images.png"),
  caption: [The memory bank, originally developed for self-supervised image representation learning, stores a
  128-dimensional embedding of each training example. Contrary to what can be observed, instead of the whole
  dataset, just $K$ samples are drawn from the memory bank in each iteration to be used as negative examples @memory_bank.
  ],
) <memory_bank_images>

$
-log h(i, bold(v)_i^(t-1))+lambda*||bold(v)_i^t-bold(v)_i^(t-1)||_2^2
$ <proximal_regularization>

While the term $-log h(i, bold(v)_i^(t-1))$ can be ignored for now, as it just denotes a form of contrastive loss, the other term
$lambda*||bold(v)_i^t-bold(v)_i^(t-1)||_2^2$ serves as the proximal regularization. It describes the mean squared error between the representation $bold(v)_i^t$ of a training example $i$ in the current batch, e.g. an image, and the representation of the same training example $bold(v)_i^(t-1)$ stored in the memory bank, which was updated the last time the image was in the current batch. This time is denoted as time step $t-1$.

The goal is to minimize @proximal_regularization, and therefore to also minimize the proximal regularization, which is the mean squared error.
The mean squared error is only at its minimum when both inputs are the same. Therefore, the proximal regularization
enforces the representation of a training example to not change too rapidly between updates, and allows for more
stable/consistent negative examples, depending on the value of weight $lambda$.
The authors report improved results with a value of $lambda=30$ @memory_bank, meaning that the proximal regularization term is 30 times more important than the contrastive loss term.

This forces the model to keep the representations of the training examples in the memory bank consistent, so
that a similarity measure can provide meaningful learning signals to the model. Our approach does not take this into account.

===== Momentum Encoder <momentum_encoder>

Inconsistent representations in a memory bank is a problem also identified by the authors of MoCo @moco,
which employ a different approach to address this issue.
They also use a queue-based memory bank, which is, similar to our approach, much smaller than the training dataset.
However, in MoCo, the negative examples in the memory bank are not updated by the model that is being trained,
but instead by a momentum encoder. The momentum encoder
is a copy of the model, but its weights are an exponential moving average of the actual model weights, defined in @ema_update @moco.
Consequently, the momentum encoder's weights are not updated by gradient descent, and
the negative examples do not come from the model that is trained, but only from the momentum encoder.

$
theta_k = m * theta_k + (1 - m) * theta_q
$ <ema_update>

With $theta_k$ being the momentum encoder weights, $theta_q$ the actual model weights, and $m$ the momentum coefficient,
the momentum encoder can be updated very slowly. Typical values for $m$ are usually between $m=0.99$ and $m=0.999$ @moco @mocov2 @mocov3 @albef.

The results are weights that change very slowly, which will also hold for the representations the momentum encoder produces.
This approach can be seen as maintaining consistency in the model that produces the negative examples,
rather than making the negative examples consistent themselves, as is done through the regularization
term in @proximal_regularization. An illustration of this method for our image-text contrastive learning can be seen in @mm_momentum_encoder.

#figure(
  image(
  width: 50%,
  "../figures/mm_momentum_encoder.png"),
  caption: [A momentum encoder generates negative examples for ITC, which are stored in a memory bank that
  discards the representations of the oldest batch when new ones are added after every step. Figure inspired by MoCo v2 @mocov2, image and
  text taken from COCO @coco.
  ],
) <mm_momentum_encoder>

Even though no gradients are needed to update the momentum encoder, it still requires additional GPU memory to keep it in memory,
which is the disadvantage of this variant.

===== Resolution

We can't use the memory bank style of @memory_bank, since we have 3,264,868 training examples (see @vl_dataset_summary). Each embedding has a size of 768, and storing them at full precision (float32) would require $3,264,868 * 768 * 4 "bytes" approx 10 "GB"$
of additional GPU memory. However, with our current setting we only have around 1.2 GB of GPU memory remaining.

We suspect that using a proximal regularization term, as in @proximal_regularization, could also stabilize our memory bank approach. However, we cannot apply it, since the term is based on the difference (MSE) between the representation of an individual training example when it was last updated in the memory bank, and its current representation in the batch. This requires the exact approach of @memory_bank, which we just deemed as infeasible.

In conclusion:

	1.	We cannot use a FIFO queue-based memory bank, as representations between samples are inconsistent.
	2.	We cannot use a memory bank that stores all training examples, as it would require too much additional GPU memory.
	3.	Proximal regularization is not applicable to a memory bank that is smaller than the training dataset.

The only alternative is to use a momentum encoder as in MoCo @moco, which is why we opt for this approach in the next experiment.
Our experimental setup remains the same, but we add a momentum encoder, which is a copy of our student model (as shown
in @mm_momentum_encoder). Oriented on ALBEF @albef, we use a memory bank of size 65,536 and a momentum factor of $m=0.995$. Both hyperparameters
also lead to good results in MoCo, where this approach was first introduced @moco.

However, we encounter an OOM error, which is not surprising considering the large memory bank size of 65k, and that we need to maintain two
of these in total (see @mm_momentum_encoder). Considering that the size of the memory bank is
crucial for performance (illustrated by MoCo @moco in @moco_vs_mb), we would like to keep it as large as possible.
Based on this goal, we apply an optimization to reduce the GPU memory consumption:

#figure(
  image(
  width: 50%,
  "../figures/moco_vs_mb.png"),
  caption: [Experiments done by the authors of MoCo show that increasing the memory bank size up to 65k is beneficial to performance @moco.
  ],
) <moco_vs_mb>

Usually, the forward pass of the momentum encoder is done after the forward pass
of the model that is trained, which is an approach MoCo @moco and ALBEF @albef perform.
It follows that during the forward pass of the momentum encoder GPU memory is already allocated
to store all activations of the actual model, as they are needed for the backward pass later.
Because we did not encounter an OOM error before using the momentum encoder, and we observed that the GPU memory
in previous experiments was almost fully utilized (up to 98%) without a momentum encoder,
we suspect that the forward pass of the momentum encoder is the reason for the overflow.

This can be remedied by performing the update and forward pass of the momentum encoder in each training step before any other work is done
(this includes the forward pass of the teacher and student).
This way, the activations of intermediate layers of the momentum encoder are freed before the forward pass of the actual model.
The result is the same performance, as the work done per step remains the same. Merely the operations in a step are reordered to
avoid the memory overflow. We illustrate this in @me_forward_comparison.

#figure(
  image(
  width: 100%,
  "../figures/me_forward_comparison.png"),
  caption: [Doing the forward pass of the momentum encoder after the forward pass of the model, when the model's activations are already stored on the GPU memory (left), leads to a cuda OOM error. This can be avoided by reordering the operations, so that the momentum encoder (EMA) update and forward pass are performed before the forward pass of the model (right). Results are based on NVIDIA RTX 4090 with 24 GB of GPU memory.
  ],
) <me_forward_comparison>


The result is shown in @itc_vs_mb. 
The performance does not exceed
that of the the standard gathering from all devices with just 511 negative examples (effective batch size of 512).
However, the experiment seems more promising to achieve a better retrieval performance with more epochs,
as it does not appear to saturate towards the end of training (see @coco_val_imagenet_itc_vs_mb).

We consider efficiency and simplicity as a key aspect of our work. Since adding a momentum encoder

1. increases the complexity of our approach,
2. increases the training time from approx. 7 hours to 10.3 hours, and
3. does not lead to a significant improvement in performance,

we decide to abandon this approach.
As a side note: The increase training time can be attributed to the additional forward pass of the momentum encoder,
and the large matrix multiplication resulting from the large memory bank.

#show table: set text(8pt)
#figure(
  table(
  columns: 4,
  stroke: none,
  table.hline(),
  table.header(
    table.cell(rowspan: 2, colspan: 1, align:horizon, [Momentum Encoder]),
    table.cell(rowspan: 2, colspan: 1, align:horizon, [ImageNet-1K]),
    table.cell(colspan: 2, align:horizon, [Retrieval]),
    [MSCOCO],
    [Flickr30K],
  ),
  table.hline(stroke: .6pt),
  [$times$], [*30.3*], [*67.2*], [*78.29*],
  [$checkmark$], [30.0], [64.28], [76.17],
  table.hline(),
),
    caption: [Comparison of the Standard ITC approach with a momentum encoder and memory bank of size 65,536. The momentum encoder
    approach does not exceed the performance of the standard ITC approach.]
) <itc_vs_mb>

#figure(
  image( "../figures/coco_val_imagenet_itc_vs_mb.png", width: 50%),
  caption: [
    Comparison of the average R@1 score for image and text retrieval on the MSCOCO val set throughout training.
    A momentum encoder shows a promising trend to achieve better retrieval performance towards the end of training.]
) <coco_val_imagenet_itc_vs_mb>
#show table: set text(12pt)