#!/bin/bash

model="/workspace/models/Sx3HRe_ibn_mb/model-44632-4.0723-train.ckpt"
model_name="Sx3HRe"

for cfg in imagenet cifar100 cifar10; do
    python run_unimodal_train.py --config-path ../configs/fine_tuning --config-name $cfg model.model_path=$model model.model_name=$model_name \
        model.linear_classifier=True # lin eval
    python run_unimodal_train.py --config-path ../configs/fine_tuning --config-name $cfg model.model_path=$model model.model_name=$model_name \
        model.linear_classifier=False # full finetuning
done
