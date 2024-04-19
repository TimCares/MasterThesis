#!/bin/bash

mkdir -p ~/miniconda3

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash

echo "Conda installed. Please restart your terminal and create the environment using 'conda env create -f environment.yaml'."

echo "Then activate the environment using 'conda activate mmrl'."

#conda env create -f environment.yml
