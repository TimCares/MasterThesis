#!/bin/bash

data_dir="../data/"

curl -O https://openslr.elda.org/resources/12/train-clean-100.tar.gz
# curl -O https://openslr.elda.org/resources/12/train-clean-360.tar.gz
# curl -O https://openslr.elda.org/resources/12/train-other-500.tar.gz

tar -xvf train-clean-100.tar.gz

rm train-clean-100.tar.gz

# tar -xvf train-clean-360.tar.gz

# rm train-clean-360.tar.gz

# tar -xvf train-other-500.tar.gz

# rm train-other-500.tar.gz

mv LibriSpeech ${data_dir}

python ../utils/wav2vec_manifest.py ${data_dir}LibriSpeech --dest ${data_dir}LibriSpeech --seed 42 --valid-percent 0
