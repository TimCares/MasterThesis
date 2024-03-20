#!/bin/bash

vg_dir="../data/vg"

mkdir -p ${vg_dir}

curl -O https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip

unzip images.zip

rm images.zip

curl -O https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

unzip images2.zip

rm images2.zip

curl -O https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip

unzip region_descriptions.json.zip

rm region_descriptions.json.zip

