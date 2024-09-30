= Introduction
== Motivation
Supervised Learning has been the fundamental technique for
training deep learning models since the development of the backpropagation
algorithm. This is not surprising, as it provides a model with clear ,domain-specific learning signals, making it is the most
direct and efficient way to solve problems or learn specific tasks.

However, the development of artificial intelligence using labeled
data has its limitations. As we aim to teach machine learning models
increasingly complex tasks, more labeled data is required.
Naturally, generating millions of labeled examples becomes 
more difficult as tasks grow in complexity, making the development of
such models less feasible.

Due to these challenges, *self-supervised learning* has gained increased attention from the scientific
community in the last few years. The approach does not
rely on creating labeled data by hand, e.g. through human annotation, but instead derives
them from the data itself. For example, to train a large language model (LLM) to understand written text,
words in a sentence are masked or deleted, and the model’s task is to predict these missing words.

#figure(
  placement: none,
  image(
  width: 75%,
  "../figures/language_masking.png"),
  caption: [
    An example for creating labels out of raw data. Any word in a sentence can
    potentially be masked, which is why training examples (including labels) can be created
    just from the data itself, without any human annotation. The labels for this example would
    be "moon" and "bright" (adapted from @self_supervised_learning_dark_matter).
  ],
) <language_masking_example>


This has three advantages: Firstly, labels do not need to be created by hand,
it is straightforward to randomly mask words in a sentence and use them as targets during training.
Secondly, massive amounts of text available on the internet enable the generation of massive training datasets.
Lastly, and most importantly, the model learns to understand and represent the world we live in.
This becomes clear with the example shown in @language_masking_example.
Here, the model must predict the words "moon" and "bright" based on the context/words
remaining after masking. To do so successfully, the model must learn that at night, it
is the moon that shines, not the sun, and that if the moon shines, it is usually bright.
This example illustrates an important characteristic of self-supervised learning:
It encourages the model to learn common sense and the world that we humans live in.

As a result of such self-supervised training, the model learns to represent input data in a
way that it can interpret and understand, typically as an n-dimensional vector, which is a process known as *representation learning*.

This is practical on tasks like text classification or image classification, where knowledge
gained from self-supervised can be transferred to downstream tasks. For instance, if a model has learned
to extract the meaning of a sentence and express it as a representation, then it
becomes significantly simpler to classify the sentiment of the sentence using this representation.

=== Multimodality
While representation learning has seen success in areas like computer vision and natural language processing, creating
interpretable representations of their respective input data, these models often lack the ability to understand that real-world
concepts are not bound to the method in which they are expressed. Consider @image_text_example: For a human, both
the image and the text express the same concept, a sheep on grass. To us, it does not matter whether this concept is expressed
in language or in an image, we understand that both represent the same idea. AI models, however, usually do not possess this ability.

#figure(
  image(
  width: 50%,
  "../figures/image_text_example.png"),
  caption: [
    The same real-world concept, expressed in different modalities. Most AI models are not able to understand that both
    expressions represent the same concept.
    Data comes from the COCO test set @coco.
  ],
) <image_text_example>

Most AI models are *unimodal*, meaning they can only understand one type of data, e.g. either text or images, but not both.
An image model, for instance, is able to reliably generate a representation of the image in @image_text_example, 
but cannot process the text. While a text model can generate a representation of the text, both representations will not be
the same. This is because, although both modalities express the same concept, the models have been trained independently 
and have no incentive to produce matching representations for the same concept, expressed in the other respective modality.

However, if the goal of representation learning is to capture the underlying concept, then
we would expect the representations to be the same (or at least very similar),
regardless of the modality in which the concept is expressed. Again, @image_text_example
shows the same concept in different modalities, so the representations should be aligned.
This is achieved through *multimodal representation learning*, where "multimodal" refers to the ability of the model to understand and
process multiple types of data, such as text and images.

=== Efficiency <introduction_efficiency>
In recent years, the idea of multimodal representation learning has received increased attention from the scientific community.
Methods like OpenAI's CLIP @clip, Meta's FLAVA @flava, or Microsoft's BEiT-3 @beit3 have shown that models are able to learn a joint representation
of multiple modalities, in this case text and images. However, these models, and deep learning models in general, have also become increasingly large
and computationally expensive to train. This trend can be partially attributed to the neural scaling laws formulated
by researchers from OpenAI @neural_scaling_law.

#figure(
  image(
  width: 100%,
  "../figures/neural_scaling_law.png"),
  caption: [
    Scaling compute, model parameters, and dataset size for large language models shows a continuous improvement in performance. Compute and
    parameters required for embedding tables is not considered, and "PF-days" refers to the number of petaFLOP-days required to train the model.
    Figure is taken directly from @neural_scaling_law.
  ],
) <neural_scaling_law_fig>

Moreover, multimodal models generally require more parameters than their unimodal counterparts, as they need to process multiple types of data.
Even though image and text representations should be aligned for the same concept, both modalities are
inherently different and require different processing steps. Consequently, multimodal models are usually larger and more computationally expensive
to train than unimodal models.

While this is not a problem for large enterprises or research institutions like OpenAI, Meta, or Microsoft, it poses a challenge for smaller
research groups or individual researchers. They rely increasingly on the progress made by large institutions, as they do not have
the resources to train such large models themselves. This situation limits the research on multimodal representation
learning that can be conducted by smaller groups and individuals, thereby reducing the diversity of research in the field.