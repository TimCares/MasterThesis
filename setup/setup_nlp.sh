#!/bin/bash

coco_dir="../data/coco"

# ----------- cc-100 dataset -----------

# curl -O https://data.statmt.org/cc-100/en.txt.xz

# ----------- enwik9 dataset -----------

mkdir -p ../data/language/enwik9

curl -O http://mattmahoney.net/dc/enwik9.zip

unzip enwik9.zip

rm enwik9.zip

perl clean_enwik9.pl enwik9 > enwik9.txt

rm enwik9

python multiprocessing_bpe_encoder.py \
        --encoder-json ${coco_dir}/encoder.json \
        --vocab-bpe ${coco_dir}/vocab.bpe \
        --inputs enwik9.txt \
        --outputs enwik9.bpe \
        --keep-empty

# Remove the original caption files
rm enwik9.txt

# Preprocess the data
echo "Preprocessing the data..."
fairseq-preprocess \
    --only-source \
    --srcdict ${coco_dir}/dict.txt \
    --trainpref enwik9.bpe \
    --destdir ../../../data/language/enwik9 \
    --workers 10

rm enwik9.bpe

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