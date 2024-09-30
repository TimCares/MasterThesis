== Evaluation on Unimodal Benchmarks <fine_tuning_unimodal_tasks>
What is often lost sight of in multimodal models is the performance on unimodal downstream tasks.
While the main goal of multimodal models is to learn a joint representation of text and images, a multimodal model should also excel
at unimodal tasks. In our case: image classification and text classification. When it comes to
unimodal downstream tasks, most papers on vision-language models, with the exception of
FLAVA @flava, exclusively focus on image classification and segmentation tasks, and do not evaluate the performance on
text classification tasks (e.g. BEiT-3 @beit3, VLMo @vlmo, and CoCa @coca).
This is surprising, as an adequate language understanding is crucial for any multimodal model, especially
when it comes to vision-lanuage reasoning like in NLVR2 @nlvr2 or VQAv2 @vqav2. Moreover, it is quite simple to test the language
understanding of a model by evaluating it on the GLUE benchmark @glue, which is what we already did once in the
language distillation experiments of @unimodal_kd_text.
We therefore evaluate our best multimodal models on both image and text classification tasks.

=== Vision <fine_tuning_image_classification>

For image classification, we take the image encoder of S-SMKE and finetune it using the same strategy as done
for the image-only distilled model: The output $bold(H)_(v, L_s)$ of the image encoder is pooled by taking the mean of all
patch representations, with the exception of the $mono(["I_CLS"])$ token. This pooled representation is then passed through a
layer normalization and a linear classification layer. The pytorch pseudocode is the same as for the image-only distilled model,
and can be found in @image_downstream_forward_pseudocode.
We do not use the shared encoder at the top of our multimodal models for the image classification task, as a shared representation
is not desired for image-specific tasks. The image encoder returns a representation specific to the image modality, which is what
we want to use for image classification. The hyperparameters (see @distil_data2vec2_imagenet_finetuning_hyperparameters),
and all other settings, are the same as for the image-only distilled model DistilData2Vec2,
and we refer to @unimodal_kd_data2vec2_finetuning for more details.

#show table: set text(8pt)
#figure(
  table(
  columns: 3,
  stroke: none,
  table.hline(),
  table.header(
  table.cell(rowspan: 2, colspan: 1, align:horizon, [Method]),
  table.cell(colspan: 2, align:horizon, [ImageNet-1K]),
    [Lin eval],
    [Finetune],
  ),
  table.hline(stroke: .6pt),
    [Data2Vec2 @data2vec2], [-], [84.5],
    [BEiTv2 @beitv2], [*80.1*], [*85.0*],
    [ResNet-101 @resnet], [-], [80.1],
    [FLAVA*$dagger$* @flava], [75.54], [-],
  table.hline(stroke: .3pt),
    [DistilData2Vec2], [56.2], [75.0],
    [S-SMKE*$dagger$*], [65.0], [75.5],
    [C-DistilData2Vec2], [#underline[71.1]], [#underline[76.1]],
  table.hline(),
),
    caption: [
      Using the image encoder of S-SMKE for image classification tasks leads to an increase in performance over DistilData2Vec2,
      but a decrease in performance when making the models directly comparable using C-DistilData2Vec2.
      *$dagger$* indicates the usage of an image encoder from vision-language models,
      and #underline[...] indicates the best performance among distilled models,
      and bold values indicate the best performance overall.
    ]
) <imagenet_finetune_results_s_smke>
#show table: set text(12pt)

#show table: set text(8pt)
#figure(
  table(
  columns: 5,
  stroke: none,
  table.hline(),
  table.header(
  table.cell(rowspan: 2, colspan: 1, align:horizon, [Method]),
  table.cell(colspan: 2, align:horizon, [CIFAR-10]),
  table.cell(colspan: 2, align:horizon, [CIFAR-100]),
    [Lin eval],
    [Finetune],
    [Lin eval],
    [Finetune],
  ),
  table.hline(stroke: .6pt),
    [FLAVA*$dagger$* @flava], [*93.44*], [-], [*78.37*], [-],
  table.hline(stroke: .3pt),
    [DistilData2Vec2], [68.4], [97.0], [46.2], [85.1],
    [S-SMKE*$dagger$*], [89.7], [97.6], [71.3], [85.2],
    [C-DistilData2Vec2], [#underline[93.2]], [*97.7*], [#underline[77.3]], [*87.2*],
  table.hline(),
),
  caption: [
    Results of performing linear evaluation and full finetuning of the image encoder from S-SMKE on CIFAR-10/100 @cifar_10_100.
  ],
)<cifar_finetune_results_s_smke>
#show table: set text(12pt)

The results on ImageNet-1K @imagenet and CIFAR-10/100 @cifar_10_100 are shown
in @imagenet_finetune_results_s_smke and @cifar_finetune_results_s_smke,
respectively. Incredibly, we observe an increase in performance, compared to DistilData2Vec2,
on all finetuning tasks, and a significant increase for linear evaluation over all datasets.

This is unusual, as DistilData2Vec2 is an image-only model, and S-SMKE is a multimodal model.
Single modality models usually perform better on their respective modality-specific tasks, as they only focus on one modality.
Even though the image encoder of S-SMKE is used for finetuning, which also technically only focuses on the image modality,
its representations are optimized for the alignment of text and images in the shared encoder, and should therefore not be as beneficial
for image-specific tasks as those of DistilData2Vec2.

However, both results of DistilData2Vec2 and S-SMKE are not directly comparable. In order to compare the two models directly,
everything about the DistilData2Vec2 and the image encoder of S-SMKE should be the same, however,
looking at @comparison_components_image_models, we see that S-SMKE uses a different teacher and loss function for the distillation.

#show table: set text(8pt)
#figure(
  table(
  columns: 4,
  stroke: none,
  table.hline(),
  table.header(
    [*Approach*],
    [Teacher],
    [Encoder init],
    [Loss],
  ),
  table.hline(stroke: .6pt),
    [*DistilData2Vec2*], [Data2Vec2 @data2vec2], [Data2Vec2 layer 1-6], [Data2Vec loss],
    [*Image S-SMKE*], [BEiTv2 @beitv2], [Data2Vec2 layer 1-6], [Contrastive Target loss],
  table.hline(),
),
  caption: [
    DistilData2Vec2 and S-SMKE are not directly comparable on visual downstream tasks, as they use different teachers and loss functions.
    "Image S-SMKE" refers to the image encoder of the trained S-SMKE model.
  ],
)<comparison_components_image_models>
#show table: set text(12pt)

To make both approaches directly comparable, so that we can directly see the impact of using an image encoder of a vision-language
model on an image-only task, we conduct the following experiment:
We distill DistilData2Vec2 again, but now train the student model with the contrastive target loss (with memory bank),
and change the teacher to BEiTv2.
We call this approach C-DistilData2Vec2. The only thing that differentiates C-DistilData2Vec2 from the image encoder of S-SMKE
is that the image encoder of S-SMKE is trained for the alignment of text and images, while C-DistilData2Vec2 is only trained
to replicate image representations. The hyperparameters and settings are the same as for DistilData2Vec2, and we refer to
@unimodal_kd_data2vec2_finetuning for more details.

The result of finetuning C-DistilData2Vec2 after the distillation
are also shown in @imagenet_finetune_results_s_smke and @cifar_finetune_results_s_smke.
Compared to DistilData2Vec2, we observe an
even more pronounced increase than through the image encoder of S-SMKE. In linear evaluation, C-DistilData2Vec2 increases the performance
by at least 15 percentage points over all benchamrks, and we even record an increase of over 25
percentage points on CIFAR-100 linear evaluation.

This experiment shows that having a unimodal image model that focuses only on image representations is better for image-specific tasks
than using an image encoder of a vision-language model. This is exactly what we expected earlier.

The most important benchmark to consider is the performance on ImageNet-1K, where S-SMKE achieves the lowest performance among all models.
Again, this is not surprising, as models like BEiTv2 @beitv2 and Data2Vec2 @data2vec2 are specific to images. More interesting is the
comparison with FLAVA @flava, which is a multimodal model that also uses its image encoder for downstream image classification tasks.
We can see that FLAVA outperforms S-SMKE on all tasks, but since the image encoder of FLAVA is a full ViT-B/16 @vit @flava model with 12 layers,
and the image encoder of S-SMKE has only 6 layers, FLAVA's results are based on a model twice as large as S-SMKE's image encoder.
We therefore consider the performance of S-SMKE on ImageNet-1K as acceptable, even though we do not consider it is obviously
not the strength of our model,
which lies more in image-text retrieval.

=== Language <fine_tuning_text_classification>

Analogue to image classification, we extract the text encoder of S-SMKE and finetune it on the GLUE benchmark @glue.
We follow the same strategy as for the text-only distilled model: From the output $bold(H)_(w, L_s)$ of the text encoder we take
the representation $bold(h)_(w, L_s, mono(["T_CLS"]))$ of the $mono(["T_CLS"])$ token, pass it through a linear pooling layer,
the weights of which come from a pretrained BERT model, through a dropout layer with $p=0.1$, and finally through a linear classification
layer. The pytorch pseudocode is the same as for the text-only distilled model F-DistilBERT,
and can be found in @text_downstream_forward_pseudocode.
Again, all hyperparameters and settings are the same as for F-DistilBERT, and we refer to @unimodal_kd_bert_finetuning
for more details, and to @distil_bert_glue_finetuning_hyperparameters for the hyperparameters.

#show table: set text(8pt)
#figure(
  table(
    columns: 11,
    stroke: none,
    table.hline(),
    table.header(
      [],
      [MNLI],
      [QNLI],
      [RTE],
      [MRPC],
      [QQP],
      [STS-B],
      [CoLA],
      [SST],
      [WNLI],
      [*Score*],
      table.hline(stroke: .6pt),
    ),
    table.cell(colspan: 11, align: left, [_Unimodal_]),
    [ELMo @elmo],[68.6], [71.1], [53.4], [76.7], [86.2], [70.4], [44.1], [91.5], [56.3*$dagger$*], [68.7],
    [BERT @bert],[*86.7$dagger$*], [*91.8$dagger$*], [*69.3$dagger$*], [*88.6$dagger$*], [*89.6$dagger$*],
    [*89.0$dagger$*], [*56.3$dagger$*], [*92.7$dagger$*], [53.5], [*79.5$dagger$*],
    table.hline(stroke: .3pt),
    table.cell(colspan: 11, align: left, [_Unimodal Distilled_]),
    [DistilBERT @distilbert],[#underline[82.2]], [#underline[89.2]], [59.9], [#underline[87.5]], [#underline[88.5]], [#underline[86.9]], [51.3], [91.3], [56.3], [#underline[77.0]],
    [F-DistilBERT (ours)],[81.2], [88.0], [#underline[67.64]], [85.0], [86.5], [81.0], [#underline[55.1]], [#underline[91.4]], [#underline[*56.4*]], [76.9],
    table.hline(stroke: .3pt),
    table.cell(colspan: 11, align: left, [_Multimodal_]),
    [CLIP @clip],[33.5], [50.5], [55.2], [65.0], [53.9], [16.0], [25.4], [88.2], [-], [(48.5)],
    [FLAVA @flava],[80.3*$dagger.double$*], [87.3*$dagger.double$*], [57.8*$dagger.double$*], [86.9*$dagger.double$*],
    [87.2*$dagger.double$*], [85.7*$dagger.double$*], [50.7*$dagger.double$*], [90.9*$dagger.double$*], [-], [(78.4)],
    [S-SMKE (ours)],[73.88], [78.71], [51.6], [79.9], [81.2], [57.5], [14.2], [83.5], [45.04*$dagger.double$*], [61.3],
    table.hline(),
  ),
  caption: [
    Results of finetuning the text encoder of S-SMKE on the GLUE benchmark @glue. We compare to unimodal, unimodal distilled, and multimodal models, where the latter use their text encoder for finetuning.
    $dagger$ indicates the best performance among unimodal models, underlined values indicate the best performance among unimodal distilled models,
    and $dagger.double$ indicates the best performance among multimodal models. Bold values indicate the best performance overall.
    Note that the score for FLAVA and CLIP are not directly comparible with others, as both works do not publish results
    on WNLI @wnli. CLIP generally does not publish results on GLUE directly, so we take the results reported by th authors of FLAVA @flava.
  ],
)<s_smke_glue_results>
#show table: set text(11pt)

The results, displayed in @s_smke_glue_results, show that we lose more than 16 percentage points in the overall GLUE score compared to
our F-DistilBERT, and even more compared to the original BERT model. While the reason for this decrease lies again in the fact that
the text encoder of S-SMKE is optimized for the alignment of text and images, and not for text-specific tasks, it certainly does not
explain the performance on CoLA @cola, which is the lowest among all models (14.2).
While we continously perform worse than FLAVA @flava, which again uses a text encoder with twice the number of layers of our text
encoder, S-SMKE outperforms CLIP @clip in all tasks, with the exception of CoLA.
A visually more appealing version of @s_smke_glue_results is shown when discussing all results in @discussion_of_results.