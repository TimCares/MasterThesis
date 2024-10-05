= Background <background>
== Basic Loss Functions <basic_loss_functions>
Throughout this work we will make use of various loss functions and extend them if necessary.
To avoid redundancy, we will define the basic concepts of them here for easier reference.

We denote bold symbols (e.g. $bold(v)$) as vectors, and $v_i$ as the $i$-th element of the respective vector.
Upper-cased bold symbols (e.g. $bold(M)$) denote matrices, and $M_(i j)$ the element in the $i$-th row and $j$-th column of the respective matrix.

=== Mean Squared Error <mean_squared_error_section>
The Mean Squared Error (MSE) is a loss function used in regression tasks and
describes the average of the squared differences between the prediction $bold(hat(y)) in RR^D$ and the target $bold(y) in RR^D$. Since
in this work the predictions and targets will exclusively be in the form of D-dimensional vectors, the MSE is defined as:

$
cal(L)_("MSE")(bold(y), bold(hat(y))) = ||bold(y)- bold(hat(y))||_2^2 = frac(1, D) sum_(d=0)^(D-1) (y_d - hat(y)_d)^2
$ <mean_squared_error>

=== Kullback-Leibler Divergence <kl_divergence_section>
The kullback-keibler divergence (KL-Divergence) is used to measure the difference between two probability distributions.
Specifically, in the context of machine learning, we are comparing a predicted probability distribution $bold(q) in RR^C$
with a target distribution $bold(p) in RR^C$.
Since we are using the KL-Divergence in the context of classification tasks, which are
discrete distributions over $C$ classes, the KL-Divergence is defined as:

$
cal(L)_("KD")(bold(p) || bold(q)) = D_("KL")(bold(p) || bold(q)) = sum_(j=0)^(C-1)p_j log frac(p_j, q_j)
$ <kl_divergence>

$p_j$ and $q_j$ are the probabilities of class $j$ according to the target and
predicted distribution, respectively.
For both distributions, there are potentially multiple classes with a non-zero probability:

$
forall j (p_j in [0, 1]) and sum_(j) p_j = 1
$ <kl_constraint>

@kl_constraint is defined analogously for $bold(q)$.

=== Cross-Entropy Loss <cross_entropy_section>
The cross-entropy Loss (CE) is quite similar to the KL-Divergence in that it compares two probability distributions in
classification tasks. It is defined as:

$
cal(L)_("CE")(bold(p), bold(q)) = H(bold(p), bold(q)) = H(bold(p)) + D_("KL")(bold(p) || bold(q))
= -sum_(j)p_j log p_j + sum_(j)p_j log frac(p_j, q_j)
$ <cross_entropy>

Here $H(bold(p))$ denotes the entropy of the target distribution $bold(p)$, and $D_("KL")(bold(p) || bold(q))$
the KL-Divergence between the target and predicted distribution.

The difference between KL-Divergence and cross-entropy is that the latter is used in traditional classification tasks,
where the target distribution $bold(p)$ is fixed and one-hot encoded,
meaning that there is only one correct class:

$
exists! i (p_i=1) and forall j(j eq.not i -> p_j=0)
$ <cross_entropy_constraint>

This strengthens the condition of the KL-Divergence, which we defined previously in @kl_constraint.
Since the goal is to minimize the cross-entropy loss $H(bold(p), bold(q))$, and $bold(p)$ is always one-hot encoded,
meaning one class $i$ has a probability of 1, all others 0 (see @cross_entropy_constraint),
the entropy of the target distribution $H(bold(p))$ will remain constant and does not affect the minimization.
Moreover, again given the constraint in @cross_entropy_constraint, only one term in the sum of the KL-Divergence is non-zero.
Consequently, we can simplify the cross-entropy loss, so that the training objective for classification tasks is:

$
min H(bold(p), bold(q)) &= H(bold(p)) + D_("KL")(bold(p) || bold(q)) \
&= D_("KL")(bold(p) || bold(q)) \
&= sum_(j)p_j log frac(p_j, q_j) \
&= log frac(1, q_i) \
&= -log q_i
$ <cross_entropy_minimization>

The cross entropy loss therefore minimizes the negative log-likelihood of the correct class $i$.

Often times, the prediction $bold(x)$ of a model is returned as raw logits, and not as probabilities.
To convert logits into probabilities the softmax function is used. For ease of use, without having to mention a 
softmax-normalization every time we make use of the cross-entropy loss, we redefine the cross-entropy loss
_actually used in this work_ as:

$
cal(L)_("CE")(bold(p), bold(x)) = H(bold(p), bold(x)) = - log exp(x_i)/(sum_(j) exp(x_j))
$ <cross_entropy_single_class_exp>

We denote $bold(x)$ as the raw logits (the model prediction), and $bold(p)$
as the one-hot encoded target distribution. $i$ is the index of the correct class,
and hence each element in $bold(x)$ corresponds to the raw logit for one class.

A comparison between the target distribution $bold(p)$ for cross-entropy and KL-Divergence
is shown in @target_dist.

#figure(
  image("../figures/target_dist.png", width: 75%),
  caption: [
    Comparison between different distributions over $C=10$ classes. The one-hot distribution (left)
    is used for classification tasks with the cross-entropy loss. The KL-Divergence is used
    when predicting a smooth distribution (right). A smooth distribution usually results from a model prediction,
    and is a popular target distribution for knowledge distillation, introduced in @knowledge_distillation.
  ],
) <target_dist>