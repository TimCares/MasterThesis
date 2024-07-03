=== Differences to Unimodal Knowledge Distillation <differences_to_unimodal_knowledge_distillation>

- for multimodal Knowledge Distillation, we also need a teacher model
- question is, which model should be the teacher model, or rather, in which modality should the teacher have been (pre-)trained?
- in our case, should the teacher be an image model or a text model?
- why not both?, so why not have a teacher text model and a teacher image model, so two teachers?
- because then the model would have to regress two representations, one for the image and one for the text
- would mean for an image-text pair, the student would regress two representations, one for the image and one for the text
- because both teachers are not related, the representations of the image and text, we would like to learn, are not aligned and related in any way
- recall that a multimodal model always has at least one shared block at the end of the architecture
- constrains the model that the same representation has to be produced for an image and its corresponding text/caption
- only then we can align the representations of the image and text
- if we would now have two teachers, one for the image and one for the text, and the student would regress both representations, then we would have two targets for one image-text pair, but we can only predict one representation, which should be the same for the image and text
- also, with two targets the model would have to learn two different representations for the image and text, and would most likely not learn
anything meaningful, as it is not possible, and not desired, to learn two different representations at the same time, i.e for an image and its corresponding text
- so we only can take one teacher model, either an image model or a text model
- we select an image model as the teacher model
- also done by @shre, which pioneered the approach 
- seems to work good in practice, so we follow this approach
- makes sense, as there are a lot of supervised, and self-supvervised, image models available, and learning content from image
well researched
- also: VLMo initializes attention layers, which are shared between image and text, with weights from a pretrained image model @vlmo
- so it seems a model can learn text based on knowledge obtained from image pretraining
- consequently, we will use, as in unimodal distillation, a pretrained image model as the teacher model
  - so we still have just one teacher model
  - also good, because a second one would increase the computational cost, GPU memory requirements, and training time
- but why not directly use a multimodal model as the teacher model?
- because the goal is to learn a multimodal model (learn alignment of modalities, i.e. image and text) from scratch
- and use the fact that there are many pretrained unimodal (for us now image) models available
- recall: goal of this research/thesis is to learn it from just a unimodal teacher model!!!

- previously, in unimodal knowledge distillation, we were able to regress all time steps of the teacher model with the student model
- means the representation of each patch or text token, respectively
  - included the CLS/BOS token
- for multimodal Knowledge Distillation, we can't do this
  - we have to regress the whole image/text representation
  - and not the representation of each patch or text token
- has two reasons
1.
- the number of time steps (patches) an image model has is usually not the same as the number of time steps (text tokens) a text model has
- so we can't just regress all time steps
- also, text can vary in length, and we use padding -> embedding at the time steps where there is padding is not meaningful
- we would regress the representation of individual image patches -> if an image time step (patch) contains e.g. an eye, and the text token at the same time step is a padding token, then regressing this does not make sense
2.
- in order for this to work, a text token at a certain time step has to be related to the image patch at the same time step
- so if an image patch contains an eye, the text token at the same time step has to contain the word "eye"
- not possible, as result would be just a concatenation of words and no meaningful text
- also, text naturally contains fill words, e.g. "the", "a", "is", which is nothing that can be represented in any way in an image
- also those words do not have any meaning regarding real-world concepts, like a dog, cat, or car

- that is why we have to regress the global representation of the image and text
- means the CLS/BOS token -> goal of it is to aggregate as much information as possible, meaning it is a global representation of the image/text content
- is independent of the number of time steps (patches) or text tokens or what is going on in a certain time step
- necessary requirement: representation of CLS token that is returned by the teacher and regressed by the student has to be independent of the image modality
- means it should be abstract enought that it can be used to also describe the content of the caption of the image
- if the representation of the CLS token still contains image specific information, then the student model will not be able to align the representation of the caption with that of the image
  - based on the caption, it is impossible to predict the image-specific information still encoded in the representation of the CLS token
  - also not desired, representation should be independent of the modality
- this is something we will elaborate on in @self_supervised_teacher

- SHRe can be seen as a special case of regressing the CLS token @shre
- was published before the inception of Transformers @transformer
- uses ConvNets
- output of FFN head of deep ConvNets usually contains global information of the image due to the increased receptive field with more layers
- so in a sense, SHRe does exactly what we aim to do, just in a supervised way: it regresses the probability distribution of the Imagenet-1k classes
  - text and image of image-text pair should have the same probability distribution as the image predicted by the teacher model