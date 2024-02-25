#!/bin/bash

brew install miniforge

conda env create -f environment.yml

echo "Setup complete. Please run 'conda activate mmrl' to use the environment."
