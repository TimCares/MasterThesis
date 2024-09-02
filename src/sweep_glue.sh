#!/bin/bash

for cfg in cola.yaml mnli_m.yaml mnli_mm.yaml mrpc.yaml qnli.yaml qqp.yaml rte.yaml sst.yaml stsb.yaml; do
    python run_unimodal_train.py --config-path ../configs/fine_tuning --config-name $cfg