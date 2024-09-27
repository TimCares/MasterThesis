= Introduction
== Motivation
Supervised Learning has been the most fundamental technique for
training Deep Learning Models since the development of the backpropagation
algorithm. This is not surprising, as it provides a model with clear domain-specific learning signals, which is the most
direct and efficient way of solving a problem or learning a task, respectively.

The development of artificial intelligence using labeled
data, however, has its limitations. To teach Machine Learning models, particularly Deep Learning models,
increasingly complex tasks, more labelled data is required.
Naturally, generating millions of labeled examples becomes 
more difficult as the underlying tasks become more complex, making the development of
such models more expensive and less feasible.

Because of this, self-supervised learning has received an increased attention from the scientific
community over the last few years. This is because self-supervised learning does not
rely on creating labeled data by hand, e.g. through human annotation, but receives it
from the context of the data. To train a Large-Language Model (LLM) to understand 
written text, for example, words in a sentence are masked or
deleted, respectively. The task of the model is then to predict those missing words.

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


This has three advantages: Firstly, labels do not need to be created by hand, as it is easy
to randomly mask words in a sentence and use them as the targets to predict during training.
Secondly, because there are massive amounts of text available on the internet, a massive amount of training
data can be generated.
And lastly but most importantly, the model learns to write text that represents the world we live in.
This becomes clear with the example seen in @language_masking_example.
Here the model would have to predict the words "moon" and "bright" based on the context/words
remaining after masking. In order to do so successfully, the model has to learn that only the moon shines at night, not the
sun, and that if the moon shines, it is usually bright.
This example illustrates an important characteristic of self-supervised learning:
It forces the model to learn common sense and the world that we humans live in.

The result of such a self-supervised training is that the model learns to represent the input data in a way that
it can interpret and understand, which is usually an n-dimensional vector. This is called representation learning.

This becomes practical on tasks like text classification or image classification, where the knowledge
the model has learned from the self-supervised task can be transferred to the downstream task: If during self-supervised training
a model has learned to
extract the meaning of a sentence and express it in a representation,
then it becomes significantly simpler to classify e.g. the sentiment of the sentence using this representation.

This also resembles the way humans solve tasks: By understanding/extracting the meaning of a sentence, which was learned before
(i.e. throughout our life), we can easily infer the sentiment of the sentence based on the meaning we have extracted.

=== The Problem: Multimodality
While representation learning has seen success in areas like computer vision and natural language processing, in both cases
creating an interpretable representation of the respective input data, they lack the ability to understand that real-world
concepts are not bound to the method in which they are expressed. Consider @image_text_example for example, for a human, both
the image and the text express the same concept: "A sheep on grass". To us, it does not matter whether this concept is expressed
in language or in an image, we understand that both express the same concept. AI models, however, usually do not have this ability.

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

They are usually what is called unimodal, meaning they can only understand one type of data, e.g. text or images, but not both.
An image model, for example, is able to reliably generate a representation of the image in @image_text_example, but it is not able to
process the text. While it is possible to generate a representation of the text, using a text model, both representations will not be
the same. This is because the way the same concept is expressed is completely different, and there has been no incentive for the models
to learn a representation that matches the representation of the other model for the same concept.

However, if the goal of representation learning is to learn a representation that represents the concept that is expressed, then
we would expect the representations to be the same, regardless of the modality in which the concept is expressed. Again, @image_text_example
shows the same concept, just expressed in different modalities, and the representations should therefore be the same (or at least very similar).
This is learned through multimodal representation learning, where "multimodal" refers to the ability of the model to understand and
process multiple types of data, e.g. text and images.

=== The Problem: Model Size
Over the last three years, the idea of multimodal representation learning has received increased attention from the scientific community.
Methods like OpenAI's CLIP @clip, Meta's FLAVA @flava, or Microsoft's BEiT-3 @beit3 have shown that models are able to learn a joint representation
of multiple modalities, in this case text and images, 

#figure(
  image(
  width: 100%,
  "../figures/neural_scaling_law.png"),
  caption: [
    @neural_scaling_law
  ],
) <neural_scaling_law_fig>