#!/bin/bash

# script for installing all dependencies globally (good for vm instances)

pip install hydra-core --upgrade

pip install hydra_colorlog --upgrade

pip install lightning soundfile pydub rich bitarray sacrebleu timm torchtext==0.17.0 tensorboardX pandas

pip install -U scikit-learn

# ffmpeg

wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz

tar xvf ffmpeg-git-amd64-static.tar.xz

# on error, check date of download and update the folder name accordingly
mv ffmpeg-git-20240301-amd64-static/ffmpeg ffmpeg-git-20240301-amd64-static/ffprobe /usr/local/bin/