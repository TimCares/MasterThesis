= Conclusion
== Summary of Contributions and Research
In this thesis, we presented an efficient end-to-end self-supervised approach to vision-language learning
that is significantly cheaper to train and smaller in size compared to existing multimodal models. The costs
are detailed in @cost_breakdown.
While previous approaches have relied on large-scale models, extensive datasets, and significant
compute, our method offers an alternative that aligns with the needs of smaller
research groups, thus filling a critical gap in the current literature. S-SMKE has less than half the parameters
of the models we compare to, is trained on only *0.75%* of the data used by CLIP @clip and *4.3%* of the data used by FLAVA @flava,
and achieves a cost reduction of *99.84%* compared to VLMo @vlmo and *99.98%* compared to CLIP @clip (more details in @cost_breakdown).

Our approach is characterized by the use of pretrained unimodal encoders, which generate representations of their input that
are then aligned by a randomly initialized shared encoder. We employ a contrastive loss function to enforce alignment
between image and text representations and utilize a self-supervised image model for simultaneous
knowledge distillation. This helps guide the alignment through high-level image representations
predicted using both an image and its caption.

Building upon the method of SHRe @shre, we adapted it to the Transformer architecture and employed a self-supervised
teacher instead of a supervised one. Our experiments demonstrate that this adaptation leads to increased performance
across all benchmarks. Furthermore, we showed that our approach achieves competitive performance with vision-language
models such as CLIP @clip and FLAVA @flava on retrieval tasks, and even outperforms them on a subset of the scoring metrics
we use.

The overall framework of our method is presented in @ev_lp. 
