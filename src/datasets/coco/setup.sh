#!/bin/bash

coco_dir="../../../data/coco"

# Download the COCO dataset
echo "Downloading the COCO dataset..."
echo "Train Dataset..."
curl -O http://images.cocodataset.org/zips/train2014.zip

echo "Validation Dataset..."
curl -O http://images.cocodataset.org/zips/val2014.zip

echo "Annotations and karpathy split meta data..."
curl -O https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip

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
mv dataset_coco.json $coco_dir
rm dataset_flickr8k.json dataset_flickr30k.json
