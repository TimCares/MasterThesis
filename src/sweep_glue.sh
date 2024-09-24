#!/bin/bash

model="/workspace/models/text_kd/model-42000-0.0292-train.ckpt"

for cfg in cola mnli mrpc qnli qqp rte sst stsb wnli; do
    python run_unimodal_train.py --config-path ../configs/fine_tuning --config-name $cfg model.model_path=$model
done
