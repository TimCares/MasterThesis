#!/bin/bash

echo "If not done already, please first run coco/setup.sh to download the COCO dataset."

curl -O http://images.cocodataset.org/zips/test2015.zip

unzip test2015.zip

rm test2015.zip

curl -O https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip

unzip v2_Questions_Train_mscoco.zip

rm v2_Questions_Train_mscoco.zip

curl -O https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip

unzip v2_Questions_Val_mscoco.zip

rm v2_Questions_Val_mscoco.zip

curl -O https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip

unzip v2_Questions_Test_mscoco.zip

rm v2_Questions_Test_mscoco.zip

curl -O https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip

unzip v2_Annotations_Train_mscoco.zip

rm v2_Annotations_Train_mscoco.zip

curl -O https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip

unzip v2_Annotations_Val_mscoco.zip

rm v2_Annotations_Val_mscoco.zip

mkdir -p ../../../data/vqa

mv v2_* ../../../data/vqa

mv test2015 ../../../data/coco
