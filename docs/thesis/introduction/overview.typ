== Overview
Building upon the established motivation and goals, this thesis is organized as follows:

*Background*\
We begin by introducing the fundamental loss functions utilized in this work for ease of reference.
These include essential loss functions such as cross-entropy loss and Kullback-Leibler divergence,
which are critical for understanding the concept of *knowledge distillation*—a key component of our approach.
Following this, we provide an introduction to the Transformer architecture, which serves as the
backbone for all models in this work. We discuss how Transformers can be employed in multimodal models
and detail how self-supervised learning is applied to train these models effectively.

We continue by exploring relevant approaches to multimodal models, including the paper
"See, Hear, and Read: Deep Aligned Representation" @shre, which introduces the fundamental approach we build on.

*Data and Methodology*\
We present all datasets used in this work, explaining their collection processes and providing detailed descriptions where necessary.
Technical details regarding hardware, software, and data handling are provided in the appendix to ensure reproducibility.

Here, we also provide additional detials on how we set up our experiments, and how we proceed with our research.

*Experiments*\
#h(1cm)*Unimodal Knowledge Distillation*\
In section @unimodal_knowledge_distillation, we present the first part of our research, where we explore knowledge
distillation on unimodal image and text models. This chapter is essential for validating that the
knowledge distillation process works effectively in a unimodal setting and for providing a baseline for the
subsequent multimodal models. It lays the foundation for our approach to *multimodal knowledge distillation*,
presented in the following chapter, allowing us to incrementally increase complexity.

#h(1cm)*Multimodal Knowledge Distillation*\
Chapter 6 introduces our approach to multimodal knowledge distillation.
We begin by adapting the approach from “See, Hear, and Read: Deep Aligned Representation” @shre to the Transformer
architecture. We then present our method of transforming this approach into an *end-to-end self-supervised learning process*,
emphasizing the contributions and innovations of our work.

#h(1cm)*Iterative Enhancements*\
Following the adaptation of an end-to-end self-supervised approach,
we introduce additional techniques aimed at improving the performance of our approach,
presented in an iterative and incremental manner. Sections such as "Token-Type Embeddings",
"Enhancing Alignment", and "Quantizing Visual Features" detail these enhancements.
Throughout these sections, we repeatedly evaluate our approach on various multimodal benchmarks
and compare it with other state-of-the-art multimodal methods.

#h(1cm)*Limitations and Insights*\
In follows a detailed evaluation of the limitations of our approach, followed by ablation studies on teacher models and
dataset sizes. These studies aim to evaluate the impact of different components on the performance of our approach.

#h(1cm)*Evaluation on Unimodal Benchmarks*\
With our final approach established, we evaluate its performance on unimodal image and text benchmarks,
directly comparing it to the unimodal distilled models from section @unimodal_knowledge_distillation.
This evaluation is crucial for determining whether the multimodal models not only excel at aligning
modalities but also perform effectively on unimodal tasks.
Demonstrating parity or superiority would suggest that our multimodal models can serve as replacements for the earlier unimodal models.

#h(1cm)*Teacher Ablation Studies*\

#h(1cm)*Discussiuon, Conclusion, Outlook, and Impact*\
Finally, we discuss the results of our work, providing insights into the implications of our findings.
We provide an outlook on future research and discuss the potential impact of our contributions on multimodal representation learning.