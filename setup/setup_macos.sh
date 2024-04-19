#!/bin/bash

brew install miniforge

conda env create -f environment.yaml

echo "Setup complete. Please run 'conda activate mmrl' to use the environment."
