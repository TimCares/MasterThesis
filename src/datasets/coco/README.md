# COCO Captions

## Get access and prepare

Run 

```shell
sh setup.sh
```

### Data
Download ```Dataset_coco.json``` for Karpathy (test) split: 

```shell
curl -O http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
```

Download Train and Val from [here](https://cocodataset.org/#download) or via curl:

2014 Train images [83K/13GB]
```shell
curl -O http://images.cocodataset.org/zips/train2014.zip
```

2014 Val images [41K/6GB]
```shell
curl -O http://images.cocodataset.org/zips/val2014.zip
```

Add 'valrest' to 'train' using ```coco_captions.ipynb```

### Dict

```shell
curl -O https://dl.fbaipublicfiles.com/fairseq/data2vec2/dict.txt
```

### Byte-Pair Encoding (BPE) meta data (For text/captions encoding)

```shell
curl -O https://dl.fbaipublicfiles.com/fairseq/data2vec2/encoder.json
```

```shell
curl -O https://dl.fbaipublicfiles.com/fairseq/data2vec2/vocab.bpe
```
