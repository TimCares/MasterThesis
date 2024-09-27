#!/bin/bash

model="/workspace/models/Sx3HRe_ibn_mb/model-44632-4.0723-train.ckpt"
model_name="Sx3HRe"

for cfg in cola mnli mrpc qnli qqp rte sst stsb wnli; do
    python run_unimodal_train.py --config-path ../configs/fine_tuning --config-name $cfg model.model_path=$model model.model_name=$model_name
done
