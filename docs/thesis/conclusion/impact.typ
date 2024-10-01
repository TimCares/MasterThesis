== Broader Impact
This work focuses on reducing the computational cost of training vision-language models by
leveraging pretrained unimodal models. While our approach does not achieve state-of-the-art
performance, which is not surprising given that we compare against models developed by large organizations
such as OpenAI, Meta, and Microsoft, it demonstrates the potential of utilizing existing components
to generate new model paradigms like multimodal models. We hope that this proof-of-concept will
inspire other researchers to explore efficient methods for training multimodal models and models in
general, thereby making the technology more accessible.

The current trend in deep learning emphasizes scaling models, data, and computational resources,
as illustrated by OpenAI’s neural scaling laws (see @neural_scaling_law_fig). However, we believe that
democratizing AI requires the development of efficient methods that enable smaller research groups and
individual researchers to contribute to the field without the need for extensive computational resources.
By lowering the barriers to entry, which our approach is characterized by, we can generate a more diverse set of approaches,
which may lead to new breakthroughs in the field.

In recent years, advances in AI have been driven largely by major organizations and have become increasingly closed-source.
A popular example is OpenAI’s GPT series, where, starting with GPT-3, less detail has been released about
the methodologies used to create these models. This shift towards reduced transparency contrasts with the
original ideals of open collaboration in AI research and makes it more challenging for researchers to build
upon and understand the work of others. Our work, for instance, relies on codebases published by the authors of
models such as FLAVA, Data2Vec, BERT, and BEiT-3. We believe that transparency is essential for advancing AI,
which is why we, like others, publish our entire codebase. By doing so, we aim to help others
advance their research and make AI more accessible.

Furthermore, our work towards a general and efficient appraoch to multimodal learning, which
has to be investigated more in future work, aims to advocate for the adoption of a philosophy centered on
*general modality-invariant representations*, that is unless explicitly undesired for certain tasks,
such as unimodal applications or cases where modality-specific details are crucial (e.g., image super-resolution).
This philosophy conveys that representations learned by a model should be independent of the modality on which they were trained.
We believe that the ultimate goal should be to generate representations that are both detailed,
capturing all relevant information about the input, and general, allowing for versatile use
across different modalities. Such modality-agnostic/invariant representations would
not only resemble human perception, where our understanding of real-world concepts is not bound to
a single modality (the concept of a "dog" is not dependent on images), but would also be beneficial in a variety of applications.

One potential application is universal modality translation, where a model generates a representation
of an input that can be used to translate the concept into another modality. Currently,
popular examples include image-to-text (image captioning) and text-to-image (image generation from text) models.
However, embracing the philosophy of general modality-invariant representations could extend
this capability to any combination of modalities, enabling the development of more general and versatile AI systems.
