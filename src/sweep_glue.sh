#!/bin/bash

for cfg in cola.yaml mnli_m.yaml mnli_mm.yaml mrpc.yaml qnli.yaml qqp.yaml rte.yaml sst.yaml stsb.yaml; do
    python run_text_finetuning.py --config-name $cfg