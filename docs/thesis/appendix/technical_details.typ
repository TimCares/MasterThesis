== Technical Details
=== Software

All implementations in this research were conducted using PyTorch @pytorch and PyTorch Lightning @pytorch_lightning.
PyTorch Lightning is a high-level library built on top of PyTorch that provides functionalities such as checkpointing,
logging, and distributed training. These features are particularly important for the distributed data parallel (DDP)
training we perform (see Section @larger_batch_sizes_ddp). By leveraging PyTorch Lightning, we avoid the need
to manually implement these functionalities in PyTorch, which, although it offers a high-level API, can be more prone to errors.
This approach allows us to save time and focus on the actual implementation of our models, as errors in
vanilla PyTorch are likely and can be difficult to debug, especially in the context of distributed training.

All experiments were performed using Python 3.10, with PyTorch 2.2.0, PyTorch Lightning 2.4.0, and CUDA 12.1.1, on an Ubuntu 22.04 machine.

Given that research inevitably involves a significant amount of experimentation and iterative development,
we utilized the experiment tracking tool Weights & Biases#footnote[#link("https://wandb.ai")] to keep track of all experiments.
The code is available at #link("https://github.com/TimCares/EV-LP").

=== Hardware
To train the models and store the data, we used the GPU cloud platform runpod.io#footnote[#link("https://www.runpod.io")] <runpod_fn>.
This platform provides access to a wide range of GPU types, including the NVIDIA RTX 4090 24GB, which we use for training almost all
of our models. The reason for choosing this platform over traditional cloud providers like AWS or GCP is that the price per
GPU hour is significantly lower, which allows us to train more models for the same budget. For example, as of September 2024,
the price per GPU hour for an NVIDIA RTX 4090 24GB on runpod.io is \$0.69. A comparable GPU on AWS, e.g. the NVIDIA V100 16GB,
costs \$3.06
#footnote[Price is taken from the official AWS pricing page (#link("https://aws.amazon.com/de/ec2/instance-types/p3/"))
and based on the on-demand price for the region us-east (North-Virginia).]
per GPU hour. This price difference is despite the fact that the V100 was released in 2017, while the RTX 4090
was released in 2022. The RTX 4090 is also faster than the V100, with a higher memory bandwidth and more CUDA cores, so training
on the RTX 4090 is more cost-effective by a margin. This evaluation is significant, as there is no external funding for this work.

To store all of our data, which is a total of >900 GB, we use a network volume provided by runpod.io. This volume is mounted
on a virtual machine on start-up, allowing to access the data for training on the GPUs and provides high flexibility.

The RTX 4090 instances we used have 61 GB of memory and 8 virtual CPUs for one GPU, and around 132 GB of memory
with 16 virtual CPUs if using two GPUs, which is similar to that of AWS.

runpod.io also provides on-demand vm instances, as well as spot instances. The latter can be automatically terminated
if demand is high, but the price is significantly lower. On-demand instances are more expensive but are non-interruptible,
which is, even though we create a model checkpoint after each training epoch, important for long-running experiments, which
is the case for most training runs in this work. As of September 2024, the price for an on-demand instance with a single RTX 4090
is \$0.69 per GPU hour, while the price for a spot instance is just \$0.35 per GPU hour. The price for an instance increases
proportionally with the number of GPUs used, so for us, a two-GPU instance costs \$1.38 per GPU hour.

=== Costs Breakdown

#show table: set text(8pt)
#figure(
  table(
    columns: 6,
    stroke: 0.6pt,
    table.hline(),
    table.header(
      [*Model*],
      [*Data\ (Samples)*],
      [*Hardware*],
      [*Compute Time (hrs)*],
      [*Compute Cost (\$)*],
      [*Estim. AWS\ Cost (\$)*@v100_aws_cost],
    ),
    table.hline(stroke: .6pt),
    [DistilData2Vec2], [1.28M], [1$times$ RTX 4090], [6.9], [4.8], [21.1], 
    [C-DistilData2Vec2], [1.28M], [1$times$ RTX 4090], [6.9], [4.8], [21.1],
    [F-DistilBERT], [13M], [1$times$ RTX 4090], [27], [18.6], [82.6], 
    [SHRe], [3.3M], [1$times$ RTX 4090], [13.1], [9.1], [40.1],
    [Transformer SHRe], [3.3M], [2$times$ RTX 4090], [7.7], [10.6], [47.1],
    [S-SMKE], [3.3M], [2$times$ RTX 4090], [11.1], [15.3], [67.9], 
    [S-SMKE#sub[CTL\_MB]], [3.3M], [2$times$ RTX 4090], [11.2], [15.5], [68.5],
    [S-SMKE+], [3.3M], [2$times$ A100 80GB], [], [], [],
    table.hline(),
  ),
  caption: [
    Cost breakdown of all major models trained in this work.
  ],
)<cost_breakdown>
#show table: set text(11pt)

#show table: set text(8pt)
#figure(
  table(
    columns: 4,
    stroke: none,
    table.hline(),
    table.header(
      [],
      [*Amount*],
      [*Total Cost (\$)*],
      [*Estim. AWS Cost (\$)*],
    ),
    table.hline(stroke: .6pt),
    [Pretraining], [1,399h], [1,373], [4,281#footnote[Cost was calculated based on the on-demand price of one NVIDIA V100 32GB
    AWS in the region US East (Northern Virginia), which is, as of September 2024, \$3.06 per hour:
    #link("https://aws.amazon.com/ec2/instance-types/p3/"). The GPU is slightly slower than the RTX 4090, so the actual cost
    is likely higher.] <v100_aws_cost>], 
    [Finetuning], [145h], [142], [444@v100_aws_cost],
    [Data Transfer], [1TB], [39], [-],
    [Data Storage], [1TB], [190], [1,470#footnote[Cost was calculated based on the price for 1TB of storage on AWS S3 in the region (Europe)
    Frankfurt, which is, as of September 2024, \$0.0245 per GB per month: #link("https://aws.amazon.com/s3/pricing/")]], 
    table.hline(),
    [Total], [-], [1,744], [>6,195],
    table.hline(),
  ),
  caption: [
    Total costs for pretraining, finetuning, data transfer, and data storage on runpod.io@runpod_fn. We compare to the estimated costs
    on AWS for similar services.
  ],
)<cost_breakdown>
#show table: set text(11pt)
