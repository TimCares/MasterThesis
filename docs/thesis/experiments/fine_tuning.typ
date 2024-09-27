=== Finetuning on Unimodal Tasks <fine_tuning_unimodal_tasks>
What is often lost sight of in multimodal models is the performance on unimodal downstream tasks.
While the main goal of multimodal models is to learn a joint representation of text and images, a multimodal model should also excel
at unimodal tasks. In our case: image classification and text classification. When it comes to
unimodal downstream tasks, most papers, with the exception of FLAVA @flava,
on vision-language model exclusively focus on image classification and segmentation tasks, and do not evaluate the performance on
text classification tasks (e.g. BEiT-3 @beit3, VLMo @vlmo, and CoCa @coca).
This is surprising, as an adequate language understanding is crucial for any multimodal model, especially
when it comes to vision-lanuage reasoning like in NLVR2 @nlvr2 or VQAv2 @vqav2. Moreover, it is quite simple to test the language
understanding of a model by evaluating it on the GLUE benchmark @glue, which is what we already did once in the
language distillation experiments of @unimodal_kd_text.
We therefore evaluate our best multimodal models on both image and text classification tasks.

For image classification, we take the image encoder of the multimodal model and finetune it using the same strategy as done
for the image-only distilled model: The output $bold(H)_(v, L_s)$ of the image encoder is pooled by taking the mean of all
patch representations, with the exception of the $mono(["I_CLS"])$ token. This pooled representation is then passed through a
layer normalization and a linear classification layer. The pytorch pseudocode is the same as for the image-only distilled model,
and can be found in @image_downstream_forward_pseudocode. Since the approach is exactly the same to the image-only distilled model,
and the image encoder of the multimodal model has the same architecture and number of layers as the image-only distilled model,
both models are directly comparable.
We do not use the shared encoder at the top of our multimodal models for the image classification task, as a shared representation
is not desired for image-specific tasks. The image encoder returns a representation specific to the image modality, which is what
we want to use for image classification. The hyperparameters (see @distil_data2vec2_imagenet_finetuning_hyperparameters),
and all other settings, are the same as for the image-only distilled model,
and we refer to @unimodal_kd_data2vec2_finetuning for more details.

Analogue to image classification, we extract the text encoder of the multimodal model and finetune it on the GLUE benchmark @glue.
We follow the same strategy as for the text-only distilled model: From the output $bold(H)_(w, L_s)$ of the text encoder, we take
the representation $bold(h)_(w, L_s, mono(["T_CLS"]))$ of the $mono(["T_CLS"])$ token, pass it through a linear pooling layer,
whe weights of which come from a pretrained BERT model, through a dropout layer with $p=0.1$, and finally through a linear classification
layer. The pytorch pseudocode is the same as for the text-only distilled model, and can be found in @text_downstream_forward_pseudocode.
Again, all hyperparameters and settings are the same as for the text-only distilled model, and we refer to @unimodal_kd_bert_finetuning
for more details, and to @distil_bert_glue_finetuning_hyperparameters for the hyperparameters. As with image classification, using the same
approach as for the text-only distilled model allows for a direct comparison between the text component from
the multimodal model and text-only distilled model.

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
    Results of finetuning the text encoder of S-SMKE on the GLUE benchmark @glue. We compare to unimodal, unimodal distilled, and multimodal models.
    $dagger$ indicates the best performance among unimodal models, underlined values indicate the best performance among unimodal distilled models,
    and $dagger.double$ indicates the best performance among multimodal models. Bold values indicate the best performance overall.
    Note that the score for FLAVA and CLIP are not directly comparible with others, as both works do not publish results
    on WNLI @wnli. CLIP generally does not publish results on GLUE directly, so we take the reported results from FLAVA @flava.
    A visually more appealing version is shown in @discussion_of_results.
  ],
)<s_smke_glue_results>
#show table: set text(11pt)