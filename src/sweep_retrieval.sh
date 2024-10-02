#!/bin/bash

model="/workspace/models/Sx3HRe_ibn_mb/model-44632-4.0723-train.ckpt"
model_name="Sx3HRe"

for cfg in retrieval_coco retrieval_flickr30k; do
    python run_unimodal_train.py --config-path ../configs/fine_tuning --config-name $cfg \
        pretrained.model_path=$model pretrained.model_name=$model_name
done
