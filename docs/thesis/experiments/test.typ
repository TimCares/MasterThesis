#set math.equation(numbering: "(1)")
$
bold(H)^w_(L-F)&=op("Encoder")_w (bold(H)^w_0) \
bold(H)^v_(L-F)&=op("Encoder")_v (bold(H)^v_0) \
bold(H)^w_(L)&=op("Encoder")_s (bold(H)^w_(L-F)) \
bold(H)^v_(L)&=op("Encoder")_s (bold(H)^v_(L-F)) \
bold(H)^w_(l)&=[bold(w)_l^mono(["T_CLS"]), bold(w)_l^1, ..., bold(w)_l^M, bold(w)_l^mono(["T_SEP"])]\
bold(H)^v_(l)&=[bold(v)_l^mono(["I_CLS"]), bold(v)_l^1, ..., bold(v)_l^N]\
$
- with $l in {1, ..., L-F, ..., L}$

- we define $bold(H)^w_(L)$ as the final output of the student model for the caption, and $bold(H)^v_(L)$ as the final output of the student model for the image, with $bold(H)^w_(L) in RR^((M+2) times D)$ and $bold(H)^v_(L) in RR^((N+1) times D)$

==== Image-Text Matching with Feature Fusion <image_text_matching_with_feature_fusion>
