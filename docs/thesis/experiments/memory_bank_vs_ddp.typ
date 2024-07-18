#set math.equation(numbering: "(1)")
===== Memory Bank vs DDP
- current approach uses DDP with batch size of 256
- we use two GPUs, so combined batch size is 512
- negative samples are gathered from all devices to get more negative samples (following VLMo TODO: cite)
- as shown in experiments with supervised teacher (SHRe), significantly improves performance
- problem is, it is expensive -> we need two GPUs instead of one
- even though batch size is doubled, training time is not halved, due to distributed communication overhead
  - comes from gradient synchronization and gathering negative samples across devices

- search for alternatives, that allows for large number of negative samples, but without the need for two GPUs
- one alternative is to use a FIFO queue based memory bank

- memory bank stores image and text embeddings of previous batches
- with every batch, we update the memory bank with the current batch embeddings, while discarding the oldest embeddings
- embeddings from memory bank are then used, together with the embeddings of the current batch, as negative examples
- given a batch size of $N$, and a memory bank size of $M$, we can have $N - 1 + M$ negative samples
- given a batch size of 256, and a memory bank size of 256, we can have 511 negative samples, which is the same as having an
  effective batch size of 512, or using two GPUs with DDP and a batch size of 256, as done in the previous experiments

- doing this, we observe a significant drop in performance
- might come as a surprise, but is a well known
- problem lies is rapidly changing model weights during training
- when using actual batch size of 512, or DDP, all negative examples (the representations) come from the same model with the same weights
- when using a memory bank, the negative examples are stored from previous batches, or steps respectively
- means from an older model with different weights
- leads to a fast shift in embedding space, and in the embeddings in general
- so pronounced, that even representations from the immediate previous steps are so different to the represenations of the current step, that they are not useful as negative examples

- memory bank was initially introduced as a mapping of the complete training dataset
- embedding of each sample is stored in the memory bank
- for each batch, $K$ samples are drawn from the memory bank to be used as negative examples
- representations of samples that are currently in the batch are then updated in the memory bank
- has the same problem that representations come from an older model with different weights, especially if the dataset is large
- this is partly mitigated by using a proximal regularization, show in @proximal_regularization 

$
-log h(i, bold(v)_i^(t-1))+lambda*||bold(v)_i^t-bold(v)_i^(t-1)||_2^2
$ <proximal_regularization>

- term $-log h(i, bold(v)_i^(t-1))$ can be ignored for now, and just be considered as the contrastive loss for simplicity
- more interesting is proximal regularization term $lambda*||bold(v)_i^t-bold(v)_i^(t-1)||_2^2$
- is the mean squared error between the representation $bold(v)_i$ of a training example $i$ in the current batch, e.g. an image, at time step $t$, and the representation of the same training example at time step $t-1$ (the last update)
- enforces that representation of a training example does not change too rapidly
- $lambda$ controls how strong this regularization is
- allowes for more stable negative examples

#figure(
  image(
  "../figures/memory_bank_images.png"),
  caption: [
  ],
) <memory_bank_images>

- problem also identified by (TODO: cite MOCO)
- authors experiments, displaced in @moco_vs_mb, show that even a memory bank that maps all training examples performs significantly worse than using the actual batch size it tries to mimic
- especially with lower batch sizes
- this type of memory bank need to sample up to 65,536 ($K=65,536$) negative examples to match actual, comparatively small, batch size of 1024

#figure(
  image(
  width: 50%,
  "../figures/moco_vs_mb.png"),
  caption: [
  ],
) <moco_vs_mb>

- we can't use the memory bank style of (TODO: cite), since we have 3,264,868 training examples
  - each embedding has a size of 768, considering storing them at full precision, so float32, we would need $3,264,868 * 768 * 4 B > 10e^9 B = 10 "GB"$ of additional GPU memory

- we suspect that using a proximal regularization term, as in @proximal_regularization, might help to stabilize our memory bank approach
- however, we cannot use it -> the term is based on the representation of the same training example at the previous time step
- but our memory bank is significantly smaller than the training dataset, and older samples are dequeued, so the same training example will never be in the memory bank and at the same time in the current batch

- authors of MOCO propose a different approach to solving this problem
- they use a queue based memory bank, much smaller than the training dataset (cite MoCo v2), so same approach as we use
- but the negative examples in the memory bank are not updated by the model that is trained, but instead by a momentum encoder,
  which is a moving average of the actual model weights
- weight update of the momentum encoder is given by

$
theta_k = m * theta_k + (1 - m) * theta_q
$

- with $theta_k$ being the momentum encoder weights, $theta_q$ the actual model weights, and $m$ the momentum factor
- set to to $m=0.999$, meaning the momentum encoder is updated very slowly
- that way negative examples in the memory bank are updated by a model with similar weights as the model that is trained
- makes them more stable and consistent
- can be seen as keeping the model consistent that produces the negative examples, instead of making the negative examples themselves consistent
  through an additional regularization term (@proximal_regularization)

- disadvantage is that a copy of the same model that is being trained has to be kept in memory
- even though no gradients are needed for the momentum encoder, it still requires additional GPU memory

- in conclusion:
1. can't use FIFO queue based memory bank, as representations of negative examples are inconsistent
2. can't use memory bank that maps all training examples, as it would require too much additional GPU memory
3. proximal regularization not applicable to a memory bank that is smaller than the training dataset
4. momentum encoder required additional GPU memory, but might be a solution to stabilize the memory bank approach

- test momentum encoder -> works
- use just one gpu -> iterations per second less than with DDP, ddp with two gpus is more than twice as fast
- cost wise, two gpus are more expensive than one gpu, but training is more than twice as fast, so in the end, two gpus are cheaper
- we keep batch size of 256 per gpu, so 512 in total, and use DDP with two gpus
- combine this with ema
- because we use ddp and not deepspeed anymore, we get OOM errors
- introduce optimization:
  -> normal forward pass (kd+itc) -> backward and step -> free activations, gradients, and memory -> update ema -> forward pass with ema -> gather embeddings from
    ema over all devices -> update memory bank