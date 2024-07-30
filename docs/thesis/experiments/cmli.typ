#set math.equation(numbering: "(1)")
===== Cross-Modal Late Interaction (CMLI)
Until now, we used the global text and image representations $mono(["T_CLS"])$ and $mono(["I_CLS"])$, respectively, for
contrastive learning and the alignment loss of (TODO: cite removing itc).
This has the disadvantage that only global information is utilized, and fine-grained, token/patch-specific, information is not considered.
This can make retrieval, and alignment in general, difficult, especially if real-world concepts described by and image and text differ
in small, yet important, details.
To address this, the authors of FILIP @filip introduce Cross-Modal Late Interaction (CMLI) for a fine-grained comparison of text and image in contrastive learning.

As shown in @cmli, no cosine similarity between $mono(["T_CLS"])$ and $mono(["I_CLS"])$ is computed, but instead the cosine 
similarity between all image patches $[bold(v)_l^k]_(1 lt.eq k lt.eq N)$ and text tokens $[bold(w)_l^j]_(1 lt.eq j lt.eq M)$, 
with $N$ being the number of image patches, and $M$ being the number of text tokens. 
Specifically, $N$ and $M$ denote the number of patches/tokens in a sequence that are not the cls token
($mono(["I_CLS"])$/$mono(["T_CLS"])$) or padding token ($mono(["PAD"])$) @filip. The choice to exclude
padding tokens is obvious, as they do not carry any semantic information. The cls token is excluded, as it contains
"just" global information. The result is that we now have the cosine similarity between all image patches and text tokens of an image-text pair.

The next step is to find for each image patch $k$, the text token with the maximum cosine similarity to this image patch.

$
m_k^("i2t") = op("argmax", limits: #true)_(1 lt.eq j lt.eq M) [bold(v)^k] [bold(w)^j]^T
$

Likewise, for each text token $j$, we get the image patch with the maximum cosine similarity to this text token
$
m_j^("t2i") = op("argmax", limits: #true)_(1 lt.eq k lt.eq N) [bold(v)^k] [bold(w)^j]^T
$

This has an intersting effect: For each image patch, the semantically most similar text token is found, and vice versa 
for each text token - the result of this operation can be seen in (2) of @cmli.
Consequently, the model will be able to associate small details of an image with individual text tokens, and vise versa. 
The actual cosine similarity between an image-text pair is then the average of all associations between an image patch and a text token.

$
s^("i2t")_(bold(H)^v_(l),bold(H)^w_(l)) = 1/N sum_(k=1)^N [bold(v)_l^k] [bold(w)_l^(m_k^("i2t"))]^T
$

$
s^("t2i")_(bold(H)^v_(l),bold(H)^w_(l)) = 1/M sum_(j=1)^M [bold(v)_l^(m_j^("t2i"))] [bold(w)_l^j]^T
$

Here, for one image-text pair, $m_k^("i2t")$ denotes the index of the text token with the highest cosine similarity to image patch $k$,
and $m_j^("t2i")$ the index of the image patch with the highest cosine similarity to text token $j$. $s^("i2t")_(bold(H)^v_(l),bold(H)^w_(l))$
denotes the the similarity score between an image representation $bold(H)^v_(l)$ and text representation $bold(H)^w_(l)$.
Vice versa, $s^("t2i")_(bold(H)^v_(l),bold(H)^w_(l))$ denotes the similarity score between a text representation $bold(H)^w_(l)$ and an image representation $bold(H)^v_(l)$. $l$ can denote any layer of the model, but we will use, as done in FILIP @filip, the last layer of the model,
so if a model has $L$ layers, then $l=L$.

In contrast to the standard contrastive learning, this similarity measure is not necessarily symmetric,
as e.g. a text token might have a maximum cosine similarity to another image patch, than a image patch to the text token @filip.
The process in illustrated in @cmli.
 
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