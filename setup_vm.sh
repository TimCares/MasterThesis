#!/bin/bash

sh setup/setup_global.sh

python setup_fairseq_fast.py build_ext --inplace

wandb login {{ RUNPOD_SECRET_WandB }}