#!/bin/bash

# script for installing all dependencies globally (good for vm instances)

pip install hydra-core --upgrade

pip install hydra_colorlog --upgrade

pip install lightning soundfile pydub rich bitarray sacrebleu timm torchtext==0.17.0

pip install tensorboardX pandas transformers datasets hapless pyarrow datasets wandb deepspeed==0.15.2 matplotlib gpustat

pip install -U scikit-learn

pip install -U "huggingface_hub[cli]"

pip install einops opt_einsum

# ffmpeg

wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz

tar xvf ffmpeg-git-amd64-static.tar.xz

# find the folder name
ffmpeg_dir=$(find . -maxdepth 1 -type d -name "ffmpeg-git-*-amd64-static" -print -quit)

# on error, check date of download and update the folder name accordingly
mv "$ffmpeg_dir/ffmpeg" "$ffmpeg_dir/ffprobe" /usr/local/bin/

rm -r ffmpeg-git-amd64-static.tar.xz

rm -r $ffmpeg_dir

# for deep speed
apt-get update
apt-get install libaio-dev

# cutlass, for deepspeed
apt-get install cmake
cd ..
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
export CUDACXX=/usr/local/cuda-12.1/bin/nvcc
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=89 # 89 -> RTX 3090
export CUTLASS_PATH=/root/cutlass

cd /usr/local/lib/python3.10/dist-packages/torch/lib
ln -s /usr/local/cuda/lib64/libcurand.so .