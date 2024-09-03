#set math.equation(numbering: "(1)")
== Knowledge Distillation <knowledge_distillation>

Training large Deep Learning models is computationally expensive, and therefore finanically infeasible for researchers outside
of large corporations. Models often need more than 100 million parameters to achieve state-of-the-art (SOTA) performance, and training
those models requires a lot of computational resources, e.g. GPUs, time and data. For example, CoCa, a vision model
reaching SOTA performance of 91% validation accuracy on ImageNet-1K @imagenet @coca, has 2.1 billion parameters, was trained on more than
3 billion images @vlmo. Based on our approximation, it should have cost over 350 thousand USD to train
#footnote[Calculation done based on the price per TPU hour of the CloudTPUv4 on Google Cloud Platform with a three year commitment.
CoCa was trained for 5 days using 2048 CloudTPUv4s @coca. At a price of 1.449 USD per TPU hour (as of August 2024), the total cost is
$1.449 "USD/h" * 24 "h/day" * 5 "days" * 2048 "TPUs" = 356,106.24 "USD"$.].

One strategy to avoid high computational costs is transfer learning. Here, a, potentially large, pretrained model is used as a starting point
and finetuned on a specific task, for a potentially different use case. That way, features learned by the model during pretraining
can be reused, and the model can be adapted to the new task with less data.
The disadvantage of this approach is that the model size does not change,
so finetuning is still computationally expensive, especially for large models.
A viable strategy would be to only use a few layers from the pretrained model, but since the layers are then used in
a different model, they have to adapt to the new environment during finetuning. 
This, while the model will be smaller and more efficient, requires longer training times.

Another option is knowledge distillation (KD). Here, a smaller model is trained to replicate, or rather predict,
the outputs of a larger model for a given sample. The larger model is called the teacher, and the smaller model the student.
There are two strategies of KD usually used in practice: Response-based KD and feature-based KD @kd_survey, and we will
make use of both in this work.

Knowledge distillation has the advantage that the student model can be much smaller and have a different architecture compared
to the teacher model. Since the teacher is running in inference mode no backpropagation is needed, and thus no gradients have to be computed
(this is still required in simple transfer learning).
This makes KD faster and requires less memory compared to finetuning.
Most importantly, it has been empirically shown that student models much smaller than their teachers can achieve similar performance.
For example, the distilled model of BERT, DistilBERT, reduced the model size by 40% while retraining 97% of the
performance of the original model @distilbert.


==== Response-based Knowledge Distillation

In response-based KD, the teacher must provide a probability distribution over a set of classes for a given sample,
which is the prediction of the teacher.
The student model tries to replicate this probability distribution. This is also called soft targets, because the probability distribution
is, unless the teacher is 100% sure, not one-hot encoded, but rather a smooth distribution over the classes.
This increases the relative importance of logits with lower values, e.g. the classes with the second and third highest logits, and
Hinton et al. @kd argue that this makes the model learn hidden encoded information the teacher model has learned, which are not
represented when focusing on just the class with the highest logit/probability. This helps the student model to generalize better, especially
on less data, compared to a model trained from scratch @kd @kd_survey.

The loss function typically used in response-based KD is the KL-Divergence, introduced in @kl_divergence_section, measuring the difference between
two probability distributions. The mathematical formulation is as follows:
Let $f(dot)$ be the teacher model, $g(dot)$ the student model, and $bold(x)$ the input sample of any modalitiy, e.g. an image.
We define $bold(u)=g(bold(x))$ and $bold(z)=f(bold(x))$ as the output of the teacher and student model, respectively.
Those are the logits, and for a classification task of e.g. 1000 classes, vectores of length 1000 ($bold(u) in RR^1000 and bold(z) in RR^1000$).
A best practice is to divide the logits by a temperature parameter $tau$, before applying the softmax function @kd @kd_survey, which
smooothes the probability distribution further, as illustrated in @prob_dist_kd.

#figure(
  image("../figures/prob_dist_kd.png"),
  caption: [
    We present a similar figure as @target_dist.
    A temperature parameter
    $tau$ further smoothens an already soft distribution. In response-based KD, this soft distribution is the prediction of a teacher
    model for a given samle (middle). Further smoothing (right) increases the relative importance of classes with lower scores. Especially in
    distributions with large number of classes some classes will have semantic similarities.
    Consider the classes "German Shorthaired Pointer" and "Labrador Retriever" of ImageNet-1K @imagenet,
    which are both dog breeds. The temperature parameter brings the scores
    for those classes closer together, which helps the student model to learn the hidden encoded information the teacher model has
    learned. In the setting of the dog breeds that means: Both classes are related/similar.
  ],
) <prob_dist_kd>

The temperature $tau$ is usually a tuneable hyperparameter,
but research has shown that it can also be a learned parameter, especially in other settings such as contrastive learning @vlmo, introduced later.

$
p_i = exp(u_i/tau) / (sum_(j) exp(u_j/tau))
$ <kd_teacher_softmax>

$
q_i = exp(z_i/tau) / (sum_(j) exp(z_j/tau))
$ <kd_student_softmax>

$i$ and $j$ denote indices of the classes, and $p_i$ and $q_i$ the probabilities of class $i$ according to the teacher $g(dot)$ and
student model $f(dot)$, respectively. The goal is to minimize the difference between both distributions (the probabilities over all classes),
computed by the KL-Divergence.

Consequently, recalling the definition of the KL-Divergence in @kl_divergence, the training objective of response-based knowledge distillation
is to bring the probability distributions of the student model for a given sample, or rather for all sample in the training set,
as close as possible to the probability distributions of the teacher model for the respective sample(s). 

==== Feature-based Knowledge Distillation

In feature-based KD, the teacher model does not need to provide a probability distribution over classes.
Instead, the student model tries to replicate the (intermediate) activations of the teacher model,
and therefore does not necessarily have to only predict the final output (probability distribution) of the teacher model.

The activations of the teacher model are usually regressed using the Mean Squared Error (MSE) as the loss function
(defined in @mean_squared_error_section). The
Mean Absolute Error (MAE) can also be used as a criterion, although it is less common @kd_survey @data2vec @data2vec2.

How the activations of the teacher model are regressed by the student model can be adjusted to the specific use case.
However, this choice is greatly influenced by the architecture of the student model. For example, if the student model
has the same architecture as the teacher, but only half the number of layers, then the student model can't replicate the activations
of the teacher 1:1. In this case, other strategies have to be used, and we will introduce one of them in @unimodal_knowledge_distillation.
An illustration of response-based vs. feature-based KD is shown in @kd_fig.

#figure(
  image("../figures/kd.png"),
  caption: [
    Response-based knowledge distillation (a) requires a teacher to provide logits
    to generate a probability distribution, which is predicted by the student. Feature-based knowledge distillation (b)
    is used for predicting the actual activations of the teacher's layer(s).
    In both cases the weights of the teacher are frozen and the teacher is running in evaluation/inference mode.
    It is important to note that in most cases the number of layers between teacher and student differs ($L eq.not K$),
    making the direct regression of the teacher's activations non-trivial.
    Figure adapted and inspired by @kd_survey, image is taken from COCO train set @coco.
  ],
) <kd_fig>
