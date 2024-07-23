#set math.equation(numbering: "(1)")
===== Cross-Modal Late Interaction (CMLI)
- currently $mono(["T_CLS"])$ and $mono(["I_CLS"])$ are used for global text and image features respectively in contrastive learning
- has the disadvantage that only global information are used for contrastive learning, and fine-grained, timestep specific, information is not considered
- can be a problem, if the real-world concepts described by image and text differ in small, yet important, details
- makes it difficult for the model to differentiate between similar concepts
- to address this, authors of FILIP introduces Cross-Modal Late Interaction (CMLI) for fine-grained interaction between text and image in contrastive learning @filip

- as shown in @cmli, there is no cosine similarity between $mono(["T_CLS"])$ and $mono(["I_CLS"])$ computed, but instead the cosine similarity between all image patches $[bold(v)_l^k]_(1 lt.eq k lt.eq N)$ and text tokens $[bold(w)_l^j]_(1 lt.eq j lt.eq M)$
- other special tokens such as the end-of-sequence token $mono(["EOS"])$ and padding token $mono(["PAD"])$ are also excluded, as they do not carry any semantic information, so cosine similarity is only computed between the actual text tokens and image patches
- for each image patch $[bold(v)_l^k]_(1 lt.eq k lt.eq N)$ we now have the cosine similarity with all text tokens $[bold(w)_l^j]_(1 lt.eq j lt.eq M)$, and vice versa
- for an image patch $k$, we now get the text token with the maximum cosine similarity to this image patch

$
m_k^("i2t") = op("argmax", limits: #true)_(1 lt.eq j lt.eq M) [bold(v)^k] [bold(w)^j]^T
$
and for each text token $j$, we get the image patch with the maximum cosine similarity to this text token
$
m_j^("t2i") = op("argmax", limits: #true)_(1 lt.eq k lt.eq N) [bold(v)^k] [bold(w)^j]^T
$

- the result can be seen in (2) of @cmli 
- with this approach, we achive an association between individual image patches and text tokens, which allows the model to find, fine-grained, matching patterns
- the actual similarity between an image and text is then the average of the maximum cosine similarity between the associated tokens, which can be used for image-text contrastive learning and image-text retrieval

$
s^("i2t")_(v,w) = 1/N sum_(k=1)^N [bold(v)_l^k] [bold(w)_l^(m_k^("i2t"))]^T
$

$
s^("t2i")_(v,w) = 1/M sum_(j=1)^M [bold(v)_l^(m_j^("t2i"))] [bold(w)_l^j]^T
$

#figure(
  image(
  width: 50%,
  "../figures/cmli.png"),
  caption: [For a token/patch, CMLI finds the semantic timestep with the highest match from the other modality. This enables the model to associate small details of image and text with each other. Notice how through the $max$-operation patches containing grass are always associated with the word "grass", and the words "sheep" and "head" are matched with the head of the sheep (associations created through $max$ are shown in (2)). The cosine similarity is then the average of all associations between an image-text pair. Figure inspired and adapted from @filip.
  ],
) <cmli>

- fine-grained alignment offers the opportunity to test image-language reasoning, an application non-referecing model previously were deemed unsuited for

- we identify the option to combine CMLI with vanilla ITC, and test the mean of both as a similarity measure


#bibliography("../references.bib")