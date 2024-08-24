#set math.equation(numbering: "(1)")
=== Knowledge Distillation

Training large Deep Learning models is computationally expensive, and therefore finanically infeasible for researchers outside
of large corporations. Models often need more than 100 million parameters to achieve state-of-the-art (SOTA) performance, and training
those models requires a lot of computational resources, e.g. GPUs, time and data. For example, CoCa, a vision model
reaching SOTA performance of 91% validation accuracy on ImageNet-1K @imagenet @coca, has 2.1 billion parameters, was trained on more than
3 billion images @vlmo and, based on our approximation, should have cost over 350 thousand USD to train. 
#footnote[Calculation done based on the price per TPU hour of the CloudTPUv4 on Google Cloud Platform with a three year commitment.
CoCa was trained for 5 days using 2048 CloudTPUv4s. At a price of 1.449 USD per TPU hour (as of August 2024), the total cost is
$1.449 "USD/h" * 24 "h/day" * 5 "days" * 2048 "TPUs" = 356,106.24 "USD"$.].

One strategy to avoid high computational costs is transfer learning. Here, a, potentially large, pretrained model is used as a starting point,
and finetuned on a specific task, for a potentially different use case. The disadvantage of this approach is that the model size does not change,
so finetuning is still computationally expensive, especially for large models. A viable strategy would be to use few layers from the pretrained model, but since the environment in which those layers were trained is different from the one in which they are used during finetuning, this
approach requires longer training times.

Another option is knowledge distillation (KD). Here, a smaller model, the student model, is trained to replicate, or rather predict
the outputs of a larger model, the teacher model, for a given sample.
There are two strategies of KD usually used in practice: Response-based KD and feature-based KD @kd_survey. Both will be used in this work.

Knowledge distillation has the advantage that the student model can be much smaller, and have a different architecture, compared
to the teacher model. Since the teacher is running in inference mode no backpropagation is needed, and thus no gradients have to be computed.
This makes KD faster and requires less memory compared to finetuning.
Most importantly, it has been empirically shown that student models much smaller than their teachers can achieve similar performance.
For example, the distilled model of BERT, DistilBERT, reduced the model size by 40%, while maintaining 97% of the
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

The loss function typically used in response-based KD is the Kullback-Leibler divergence (KL), which measures the difference between
two probability distributions. The mathematical formulation is as follows:
Let $f$ be the teacher model, $g$ the student model, and $bold(x)$ the input sample of any modalitiy, e.g. an image or a text.
We omit the notation $bold(H)_(v, 0)$ and $bold(H)_(w, 0)$ for the input, as defined in (TODO: cite notation section),
as knowledge distillation is independent of the modality and model architecture. Therefore, the input can be imagined as
e.g. an image or text.

We define $bold(u)=g(bold(x))$ and $bold(z)=f(bold(x))$ as the output of the teacher and student model, respectively.
Those are the logits, and for a classification task of e.g. 1000 classes, vectores of length 1000.
A best practice is to divide the logits by a temperature parameter $tau$, before applying the softmax function @kd @kd_survey, which
smooothes the probability distribution further, as illustrated in @prob_dist_kd.

#figure(
  image("../figures/prob_dist_kd.png"),
  caption: [
    Comparison between the distribution for a classification task of 10 classes. The target distribution is used to train a model
    from scratch, while the model prediction over the classes can be used for knowledge distillation. The temperature parameter
    $tau$ further smoothens the distribution, and increases the relative importance of classes with lower scores. Especially in
    distributions with large number of classes, e.g. ImageNet-1K @imagenet, some classes will have semantic similarities, like
    "German Shorthaired Pointer" and "Labrador Retriever" @imagenet, which are both dog breeds. The temperature parameter brings the scores
    for those classes closer together, which helps the student model to learn the hidden encoded information the teacher model has
    learned. In the setting of the dog breeds, given above, that means: Both classes are related/similar.
  ],
) <prob_dist_kd>

The temperature $tau$ is usually a tuneable hyperparameter,
but research has shown that it can also be a learned parameter, especially in other settings such as contrastive learning (introduced later) @vlmo.

$
p_i = exp(u_i/tau) / (sum_(j) exp(u_j/tau))
$ <kd_teacher_softmax>

$
q_i = exp(z_i/tau) / (sum_(j) exp(z_j/tau))
$ <kd_student_softmax>

$i$ and $j$ denote indices of the classes, and $p_i$ and $q_i$ the probabilities of class $i$ according to the teacher $g$ and
student model $f$, respectively. The goal is to minimize the difference between the probabilities over all classes,
computed by the KL divergence:

$
cal(L)_("KD") = D_("KL")(bold(p) || bold(q)) = sum_(j)p_j log(frac(p_j, q_j))
$ <kl_divergence>

As in @kd_teacher_softmax and @kd_student_softmax, $j$ is the index of a class, and $p_j$ and $q_j$ the probabilities of class $j$ according to the teacher and student model, respectively @shre.

The KL-Divergence is a closely realted to the cross-entropy loss, which is typically used in classification tasks, and defined as follows:

$
cal(L)_("CE") = H(bold(p)) + D_("KL")(bold(p) || bold(q)) = -sum_(j)p_j log(p_j) + sum_(j)p_j log(frac(p_j, q_j))
$ <cross_entropy>

The difference between both is the target distribution $bold(p)$.
For classification tasks $bold(p)$ is the one-hot encoded ground truth, where the probability of the correct class is 1, and 0 for all
others (@prob_dist_kd).
Since $bold(p)$ is constant, minimizing the cross-entropy loss simplifies to

$
min log(frac(1, q_k))
$ <cross_entropy>

In knowledge distillation, the target distribution is the probability distribution of the teacher model, which is not one-hot encoded,
but rather a smooth distribution over the classes, as explained above.

==== Feature-based Knowledge Distillation

In feature-based KD, the teacher model does not need to provide a probability distribution over classes.
Instead, the student model tries to replicate the (intermediate) activations of the teacher model, so
the student model does not necessarily have to only predict the final output of the teacher model.

The activations of the teacher model are usually regressed, using the Mean Squared Error (MSE) as the loss function. The
Mean Absolute Error (MAE) can also be used as a criterion, although it is less common @kd_survey @data2vec @data2vec2.
We define the MSE as follows:

$
cal(L)_(K D) = op("MSE")(bold(p), bold(q)) = ||bold(p) - bold(q)||_2 = frac(1, K) sum_(j=1)^K (p_j-q_j)^2
$ <mean_squared_error>

An illustration of response-based vs. feature-based KD is shown in @kd_fig.

#figure(
  image("../figures/kd.png"),
  caption: [
    Response-based knowledge distillation (a) requires a teacher to provide logits,
    from which a probability distribution can be created, which is predicted by the student. Feature-based knowledge distillation (b)
    is used for predicting the actual activations of the teacher layer(s). Which teacher activations
    are regressed by which part of the student model is not fixed, and can be adjusted to the specific
    use case. An intuitive choice is to regress the activations of the teacher's last layer with the
    student's last layer.
    In both cases the weights of the teacher are frozen and the teacher is running in evaluation/inference mode.
    Figure adapted and inspired by @kd_survey, image is taken from COCO train set @coco.
  ],
) <kd_fig>

#bibliography("../references.bib")
