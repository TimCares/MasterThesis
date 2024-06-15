#!/bin/bash

# script for installing all dependencies globally (good for vm instances)

pip install hydra-core --upgrade

pip install hydra_colorlog --upgrade

pip install lightning soundfile pydub rich bitarray sacrebleu timm torchtext==0.17.0

pip install tensorboardX pandas transformers datasets hapless pyarrow datasets wandb deepspeed

pip install -U scikit-learn

pip install -U "huggingface_hub[cli]"

# ffmpeg

wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz

tar xvf ffmpeg-git-amd64-static.tar.xz

# find the folder name
ffmpeg_dir=$(find . -maxdepth 1 -type d -name "ffmpeg-git-*-amd64-static" -print -quit)

# on error, check date of download and update the folder name accordingly
mv "$ffmpeg_dir/ffmpeg" "$ffmpeg_dir/ffprobe" /usr/local/bin/

rm -r ffmpeg-git-amd64-static.tar.xz

rm -r $ffmpeg_dir
