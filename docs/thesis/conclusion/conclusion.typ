= Conclusion
== Summary of Contributions and Research
In this thesis, we presented an efficient end-to-end self-supervised approach to vision-language learning
that is significantly cheaper to train and smaller in size compared to existing multimodal models.
While previous approaches have relied on large-scale models, extensive datasets, and significant
compute, our method offers an alternative that aligns with the needs of smaller
research groups, thus filling a critical gap in the current literature.

Our approach is characterized by the use of pretrained unimodal encoders, which generate representations of their input that
are aligned by a randomly initialized shared encoder. We employ a contrastive loss function to enforce alignment
between image and text representations and utilize a self-supervised image model for simultaneous
knowledge distillation. This helps guide the alignment through high-level image representations
predicted using both an image and its caption.

Building upon the method of SHRe @shre, we adapted it to the Transformer architecture and employed a self-supervised
teacher instead of a supervised one. Our experiments demonstrate that this adaptation leads to increased performance
across all benchmarks. Furthermore, we showed that our approach achieves competitive performance with vision-language
models such as CLIP @clip and FLAVA @flava on retrieval tasks, and even outperforms them on certain scoring metrics,
all while using only 0.75% of the data used by CLIP and 4.3% of the data used by FLAVA.

The overall framework of our method is presented in @ev_lp. 
