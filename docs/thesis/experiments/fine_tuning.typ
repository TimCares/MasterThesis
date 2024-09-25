== Finetuning on Unimodal Tasks <fine_tuning_unimodal_tasks>
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