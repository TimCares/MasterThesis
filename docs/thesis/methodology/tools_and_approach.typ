= Experimental Approach

In this work, we adopt an incremental experimental methodology that begins with the simplest possible approach and
incrementally builds upon the results and knowledge gained from each step. Our primary aim is to validate
the effectiveness of knowledge distillation (KD), which is central to our approach throughout this thesis.

We start by testing KD in a unimodal setting to ensure that the technique functions correctly within our experimental framework. Specifically, we perform knowledge distillation of both an image model (Data2Vec2 @data2vec2) and text model (BERT @bert). This
step leverages extensively researched methods in unimodal KD and serves to confirm that our implementation is effective.

Building on the success of unimodal KD, we then advance to our main objective: creating a multimodal model using
knowledge distillation from a unimodal teacher. Recognizing that distilling a multimodal model from a unimodal teacher
is more challenging than distilling between unimodal models of the same architecture, we begin this part by employing
a supervised teacher model. This means that the teacher model has been trained on labeled data and provides logits
that the student model can regress. This approach effectively reproduces the method presented in SHRe @shre to some extend,
which has been demonstrated to work as a proof-of-concept.

Once we validate that this approach works similarly for our models, we proceed to the next phase by advancing
to a self-supervised teacher. This step is crucial because our ultimate goal is to develop a model and procedure
for efficiently creating a multimodal model entirely reliant on unlabeled data. Consequently, the teacher model
and any pretrained modules used must not have been trained on labeled data. The aim of our experimental
approach is to demonstrate the feasibility of this concept, serving as a proof-of-concept for creating
efficient multimodal models without the need for any labeled data in the end-to-end training process.
