== Hyperparameters <hyperparameters>

#figure(
  table(
    table.vline(x:1, stroke: .3pt),
    table.vline(x:2, stroke: .3pt),
    columns: 5,
    stroke: none,
    table.hline(),
    table.header(
      [*Type*],
      [*Hyperparameters*],
      [*ImageNet*],
      [*CIFAR10*],
      [*CIFAR100*],
    ),
    table.hline(stroke: .6pt),
    table.cell(rowspan: 10, align:horizon, [*Training*]), 
    [Epochs], table.cell(colspan: 3, align:horizon, [15]),
    [Batch size], table.cell(colspan: 3, align:horizon, [256]),
    [AdamW ε], table.cell(colspan: 3, align:horizon, [1e-8]),
    [AdamW β], table.cell(colspan: 3, align:horizon, [(0.9, 0.999)]),
    [Weight decay], table.cell(colspan: 3, align:horizon, [0.01]),
    [Base learning rate], table.cell(colspan: 3, align:horizon, [1e-3]),
    [Layer Decay], table.cell(colspan: 3, align:horizon, [0.81]),
    [Learning rate schedule], table.cell(colspan: 3, align:horizon, [Cosine]),
    [Warmup steps], table.cell(colspan: 3, align:horizon, [10% of total steps]),
    [Hardware], table.cell(colspan: 3, align:horizon, [1 $times$ RTX 4090 24GB]),
    table.hline(stroke: .6pt),
    table.cell(rowspan: 5, align:horizon, [*Mixup* @mixup*\/Cutmix* @cutmix]),
    [Mixup prob.], table.cell(colspan: 3, align:horizon, [0.8]),
    [Cutmix prob.], table.cell(colspan: 3, align:horizon, [1.0]),
    [Prob.], table.cell(colspan: 3, align:horizon, [0.9]),
    [Switch prob.], table.cell(colspan: 3, align:horizon, [0.5]),
    [Label smooting], table.cell(colspan: 3, align:horizon, [0.1]),
    table.hline(stroke: .6pt),
    table.cell(rowspan: 4, align:horizon, [*RandAugment* @randaugment]),
    [Magintude], table.cell(colspan: 3, align:horizon, [9]),
    [Magnitude std.], table.cell(colspan: 3, align:horizon, [0.5]),
    [Magnitude inc.], table.cell(colspan: 3, align:horizon, [1]),
    [\# ops], table.cell(colspan: 3, align:horizon, [2]),
    table.hline(stroke: .6pt),
    table.cell(rowspan: 3, align:horizon, [*RandomErase* @randerase]),
    [Prob.], table.cell(colspan: 3, align:horizon, [0.25]),
    [Mode], table.cell(colspan: 3, align:horizon, [pixel]),
    [\# erase], table.cell(colspan: 3, align:horizon, [1]),
  ),
  caption: [Hyperparameters used for the ImageNet-1K @imagenet, CIFAR10 @cifar_10_100, and CIFAR100 @cifar_10_100 finetuning of the distilled Data2Vec image model.
  We refer to the respective papers for details on the augmentation techniques @mixup @cutmix @randaugment @randerase.
  ],
)<distil_data2vec2_imagenet_finetuning_hyperparameters>

#bibliography("../references.bib")