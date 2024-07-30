#set math.equation(numbering: "(1)")
===== Ablation Study: Removing ITC
In the previous chapters, we made efforts to improve and extend the architecture and training of the model to achieve
better alignment of the modalities, which should go hand in hand with better retrieval performance.
However, we observed that contrastive learning is highly dependent on the granularity of the data (TODO: cite),
e.g. the quality of captions, and the number of negative examples (TODO: cite). While there are viable options to
overcome the latter, they are not always feasible, nor efficient, and the right hardware is required.
Therefore, we identify Image-Text Contrastive Learning (ITC) as a weak point of our approach.
The best approach would be to not use contrastive learning at all, which is why we will investigate the effects of an absence of ITC in this ablation study.

As an intersting side note, at the point of writing, the state-of-the-art (SOTA) vision-language model, BEiT-3, gives us a good reason to discard ITC. BEiT-3 pre-training is performed without contrastive learning, and the authors report SOTA results @beit3 after fine-tuning on retrieval tasks MSCOCO @coco and Flickr30K @flickr30k. Even without fine-tuning, BEiT-3 achives competitive results on Flickr30K, and even outperforms models trained using contrastive learning (see @beit3_flickr30k_zero_shot). 

#figure(
  image(
  width: 50%,
  "../figures/beit3_flickr30k_zero_shot.png"),
  caption: [Excerpt from the BEiT-3 paper @beit3, showing that BEiT-3 outperforms models trained with contrastive learning on the Flickr30K dataset without fine-tuning. As the results are based on the pre-trained model without fine-tuning, and the model has not been pre-trained with contrastive learning, nor was the Flickr30K test set used during pre-training, the retrieval application becomes #underline[*truly zero-shot*]. However, it is important to note that the performance reported in the paper are based on the vision Transformer giant (ViT-G/14) architecture, which a patch size of 14x14 @beit3 @vit-g, and naturally leads to better performance than smaller architectures.],
) <beit3_flickr30k_zero_shot>

Since we are removing ITC, we can also remove the additional projection introduced in (TODO: cite),
which was added to seperate the regression of the $mono(["I_CLS"])$ token of BEiT-2 (our teacher model) from
the contrastive learning, and lead to superior performance.
Consequently, the output cls token of our shared Transformer block will be used for the regression of the
$mono(["I_CLS"])$ token of BEiT-2 directly.

The advantage a feature-based Knowledge Distillation approach has in this case, is that even if we do not use contrastive learning, the representations will still be, to some extend, aligned.
This is because we are regressing the $mono(["I_CLS"])$ token of BEiT-2, not a probability distribution over the classes,
as in SHRe @shre, described in (TODO: cite). Therefore, if the model would reach a loss (MSE) of 0, then the representation between
the cls token output for a caption by the student model $mono(["T_CLS"]#sub[s])$, and the $mono(["I_CLS"]#sub[t])$ token of BEiT-2
for the fitting image would be exactly the same. If the same holds for the cls token output for the same image by the student model $mono(["I_CLS"]#sub[s])$, then $mono(["T_CLS"]#sub[s])$ and $mono(["I_CLS"]#sub[t])$ would be same, and aligned.

$
op("MSE")(mono(["T_CLS"]#sub[s]), mono(["I_CLS"]#sub[t])) = 0 &and op("MSE")(mono(["I_CLS"]#sub[s]), mono(["I_CLS"]#sub[t])) = 0 \
==> mono(["T_CLS"]#sub[s]) &= mono(["I_CLS"]#sub[t])
$ <proof_representation_alignment_to_itc>

The aforementioned does not hold in response-based Knowledge Distillation, as done in SHRe @shre, since the probability distribution over
classes is regressed. This ensures alignment on the level of categories, but not on the level of features: The cls token output would not need
to be the same for an image-text pair, as long as the probability distribution over the classes is the same @shre.

It follows from @proof_representation_alignment_to_itc, that we will use $mono(["T_CLS"]#sub[s])$ and $mono(["I_CLS"]#sub[s])$ for the retrieval tasks, we will use the cosine similarity between those as the similarity measure (as done before). 

When we first reproduced the architecure of "See, Hear, and Read: Deep Aligned Representations" (SHRe) in a supervised setting (TODO: cite),
we directly used ITC as defined in e.g. VLMo @vlmo, instead of using the alignment approach actually used in SHRe.
The alignment approach of SHRe is based on a pairwise cosine similarity between an image and text. The goal is to maximize the
cosine similarity between a matching image-text pair, and minimize the cosine similarity between a non-matching image-text pair.
What differentiates it from ITC, is that we are only intersted in the cosine similiarity between one image and one text, meaning
the score/similarity is not softmax-normalized over the cosine similarity between the same image and a set of negative (non-matching) texts.
This makes the approach independent of the number of negative examples: There is only ever one image and one text, which are compared.
Following the cosine embedding loss of pytorch @pytorch_cos_emb_loss, we define the loss as:

#figure(
  image(
  width: 50%,
  "../figures/cos_emb_loss.png"),
  caption: [@pytorch_cos_emb_loss],
) <cos_emb_loss>

Where $y=1$ denotes a matching pair, and $y=-1$ denotes a non-matching pair.
Finding positive ($y=1$) pairs is easy, as we can use the image-text pairs from the dataset.
However, finding negative ($y=-1$) pairs is not as straight forward, and we need to find a suitable strategy.
The best approach would be to use hard-negative mining, as done in VLMo for Image-Text Matching (ITM) @vlmo, but VLMo
selects hard-negatives using the cosine similarities from their ITC approach, which we do not have @vlmo.
Since we do not have any labels, there is no other way to find negative pairs than to just randomly select them.
Therefore, we resort to finding a random negative text $j$ for each image $i$ in the current batch, with $i eq.not j$.
Apart from this being the only option, it also has the advantage of being simple and fast. In total, we have $N$
positive pairs and $N$ negative pairs per batch, with $N$ being the batch size. The total alignment loss is defined in @sx3hre_align_loss.

$
cal(L)_("cos")^("Sx3HRe") = 
1/2 * 
(
cal(L)_("cos")(mono(["T_CLS"]#sub[s]), mono(["I_CLS"]#sub[s])) + 
cal(L)_("cos")(mono(["T_CLS"]#sub[s]), mono(["I_CLS"]#sub[s])) 
), \ 
j tilde.op op("Uniform")({1, ..., N} backslash {i})
$ <sx3hre_align_loss>

#figure(
  table(
  columns: (25%, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: none,
    table.hline(),
    table.header(
      table.cell(rowspan: 3, colspan: 1, align:horizon, [*Model*]),
      table.cell(colspan: 6, [*MSCOCO (5K test set)*]),
      table.cell(colspan: 6, [*Flickr30K (1K test set)*]),
      table.cell(colspan: 3, [Image $arrow.r$ Text]),
      table.cell(colspan: 3, [Text $arrow.r$ Image]),
      table.vline(stroke: .4pt),
      table.cell(colspan: 3, [Image $arrow.r$ Text]),
      table.cell(colspan: 3, [Text $arrow.r$ Image]),
      table.hline(start: 1, end: 4, stroke: .2pt),
      table.hline(start: 4, end: 7, stroke: .2pt),
      table.hline(start: 7, end: 10, stroke: .2pt),
      table.hline(start: 10, end: 13, stroke: .2pt),
      [R@1], [R@5], [R@10], [R@1], [R@5], [R@10], [R@1], [R@5], [R@10], [R@1], [R@5], [R@10]
    ),
    table.hline(stroke: .4pt),
    [FLAVA @flava], [42.74], [76.76], [-], [38.38], [67.47], [-], [67.7], [94.0], [-], [65.22], [89.38], [-],
    [CLIP @clip], [58.4], [81.5], [88.1], [37.8], [62.4], [72.2], [88.0],[98.7], [99.4], [68.7], [90.6], [95.2],
    [*Sx3HRe#sub[ITC]*], [41.36], [71.16], [82.0], [30.2], [59.46], [72.54], [9.5], [35.68], [50.18], [8.38], [37.54], [49.88],
    [*Sx3HRe#sub[\~ITC]*], [33.52], [59.34], [70.14], [11.26], [29.12], [40.40], [8.86], [29.88], [41.98], [4.45], [18.42], [26.82],
    [*Sx3HRe#sub[COS]*], [35.48], [60.74], [71.44], [12.46], [31.74], [43.71], [9.44], [31.18], [43.50], [4.8], [20.98], [30.08],
    table.hline(),
  ),
  caption: [\~ITC is full zero-shot on Flickr30K, and task zero-shot on MSCOCO. COS is task zero-shot on both datasets.],
)<image_text_retrieval_shre_first>

#bibliography("../references.bib")