#!/bin/bash

coco_dir="../../../data/coco"

# Download the COCO dataset
echo "Downloading the COCO dataset..."
echo "Train Dataset..."
curl -O http://images.cocodataset.org/zips/train2014.zip

echo "Validation Dataset..."
curl -O http://images.cocodataset.org/zips/val2014.zip

echo "Annotations and karpathy split meta data..."
curl -O http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip

# Unzip the dataset
echo "Unzipping the dataset..."
unzip train2014.zip
unzip val2014.zip

# Unzip the annotations and karpathy split meta data
unzip caption_datasets.zip

# Remove the zip files
echo "Removing the zip files..."
rm train2014.zip
rm val2014.zip
rm caption_datasets.zip

# Download the data2vec2 dictionary
echo "Downloading the data2vec2 dictionary..."
curl -O https://dl.fbaipublicfiles.com/fairseq/data2vec2/dict.txt

# Download the BPE Encoder
echo "Downloading the BPE Encoder..."
curl -O https://dl.fbaipublicfiles.com/fairseq/data2vec2/encoder.json

# Download the BPE Vocab
echo "Downloading the BPE Vocab..."
curl -O https://dl.fbaipublicfiles.com/fairseq/data2vec2/vocab.bpe

# Prepare file structure
echo "Preparing file structure..."
mkdir -p $coco_dir

# Move the dataset to the data directory
echo "Moving the dataset to the data directory..."
mv train2014 $coco_dir
mv val2014 $coco_dir

# Move the annotations and karpathy split meta data to the data directory
echo "Moving the annotations and karpathy split meta data to the data directory..."
mv caption_datasets/dataset_coco.json $coco_dir
rm -r caption_datasets

python create_karpathy_split.py
rm $coco_dir/dataset_coco.json

mv dict.txt $coco_dir
mv encoder.json $coco_dir
mv vocab.bpe $coco_dir

python write_captions.py

# Perform BPE encoding
echo "Performing BPE encoding..."

python multiprocessing_bpe_encoder.py \
        --encoder-json ${coco_dir}/encoder.json \
        --vocab-bpe ${coco_dir}/vocab.bpe \
        --inputs captions_train.txt \
        --outputs captions_train.bpe \
        --keep-empty

python multiprocessing_bpe_encoder.py \
        --encoder-json ${coco_dir}/encoder.json \
        --vocab-bpe ${coco_dir}/vocab.bpe \
        --inputs captions_val.txt \
        --outputs captions_val.bpe \
        --keep-empty

python multiprocessing_bpe_encoder.py \
        --encoder-json ${coco_dir}/encoder.json \
        --vocab-bpe ${coco_dir}/vocab.bpe \
        --inputs captions_test.txt \
        --outputs captions_test.bpe \
        --keep-empty

# Remove the original caption files
echo "Removing the original caption files..."
rm captions_train.txt
rm captions_val.txt
rm captions_test.txt

# Preprocess the data
echo "Preprocessing the data..."
n_cpus=$(nproc)
fairseq-preprocess \
    --only-source \
    --srcdict ${coco_dir}/dict.txt \
    --trainpref captions_train.bpe \
    --validpref captions_val.bpe \
    --testpref captions_test.bpe \
    --destdir ${coco_dir} \
    --workers n_cpus # will use all available CPU cores

# Remove the BPE encoded caption files
echo "Removing the BPE encoded caption files..."
rm captions_train.bpe
rm captions_val.bpe
rm captions_test.bpe

echo "Setup complete."

