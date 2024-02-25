#!/bin/bash

wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

bash Miniforge3.sh -b -p "${HOME}/conda"

rm Miniforge3.sh

conda env create -f environment.yml

echo "Setup complete. Please run 'conda activate mmrl' to use the environment."