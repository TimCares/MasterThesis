#!/bin/bash

coco_dir="../data/coco"

# ----------- cc-100 dataset -----------

# curl -O https://data.statmt.org/cc-100/en.txt.xz

# ----------- openwebtext dataset -----------

# download from: https://drive.google.com/drive/folders/1IaD_SIIB-K3Sij_-JjWoPy_UrWqQRdjx?usp=sharing

# tar -xvf url_subset*.tar

# tar -xvf openwebtext/*.xz

python clean_openwebtext.py

python multiprocessing_bpe_encoder.py \
        --encoder-json ${coco_dir}/encoder.json \
        --vocab-bpe ${coco_dir}/vocab.bpe \
        --inputs openwebtext.txt \
        --outputs openwebtext.bpe \
        --keep-empty

rm openwebtext.txt

fairseq-preprocess \
    --only-source \
    --srcdict ${coco_dir}/dict.txt \
    --trainpref openwebtext.bpe \
    --destdir ../../../data/language/openwebtext \
    --workers 10

rm openwebtext.bpe