#set math.equation(numbering: "(1)")
== Notations and Definitions <notions_and_definitions>

The architectures used in the experiments of this thesis are based on the Transformer architecture developed by Vaswani et al. @transformer.
For the Transformer blocks, we use the same same structure for both image and text.
As mentioned in (TODO: cite data preparation), text is tokenized into subwords using the GPT-2 byte-pair encoder also used in Data2Vec @data2vec @data2vec2. Before being passed into the Transformer, a start-of-sequence token $mono(["T_CLS"])$ is added to the beginning of the sequence, and an end-of-sequence token $mono(["T_SEP"])$ is added to the end of the sequence. Then, the sequence is embedded into 768-dimensional vectors, and a positional encoding is added
to the embeddings. In this thesis, we define a text sequence as follows:

$
bold(H)^s_(w, l)=[bold(h)^s_(w, l, mono(["T_CLS"])), bold(h)^s_(w, l, 1), ..., bold(h)^s_(w, l, M), bold(h)^s_(w, l, mono(["T_SEP"]))]
$

Because we use KD in some parts, representations will be superscripted with $s$ or $t$, for a student and teacher representation, respectively.

#bibliography("../references.bib")
