import os
import json
import torch
from fairseq.data import Dictionary
from torchvision.datasets.folder import default_loader
from data_utils import get_transforms, _write_data_into_jsonl, curl_dataset
from bpe_encoder import BPEEncoder
from unimodal import BaseDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

class BaseImageText(BaseDataset):
    def __init__(
        self,
        data_path,
        batch_size,
        num_workers,
        num_max_bpe_tokens,
        transform_jitter=False,
        beit_transforms=False,
        no_transform=False,
        crop_scale=(0.6, 1.0),
    ):
        super().__init__(data_path, batch_size, num_workers)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.transform_jitter = transform_jitter
        self.beit_transforms = beit_transforms
        self.no_transform = no_transform
        self.crop_scale = crop_scale
        self.path_to_data = None
        
    def prepare_data(self):
        if not os.path.isfile(os.path.join(self.data_path, "dict.txt")):
            filename = curl_dataset("https://dl.fbaipublicfiles.com/fairseq/data2vec2/dict.txt")
            os.rename(filename, os.path.join(self.data_path, "dict.txt"))

        if not os.path.isfile(os.path.join(self.data_path, "encoder.json")):
            filename = curl_dataset("https://dl.fbaipublicfiles.com/fairseq/data2vec2/encoder.json")
            os.rename(filename, os.path.join(self.data_path, "encoder.json"))

        if not os.path.isfile(os.path.join(self.data_path, "vocab.bpe")):
            filename = curl_dataset("https://dl.fbaipublicfiles.com/fairseq/data2vec2/vocab.bpe")
            os.rename(filename, os.path.join(self.data_path, "vocab.bpe"))

        encoder_json_path = os.path.join(self.data_path, "encoder.json")
        vocab_bpe_path = os.path.join(self.data_path, "vocab.bpe")
        self.bpe_encoder = BPEEncoder(encoder_json_path, vocab_bpe_path)
        self.dictionary = Dictionary.load(os.path.join(self.data_path, "dict.txt"))

        self.bos_token_id = self.dictionary.bos()
        self.eos_token_id = self.dictionary.eos()
        self.pad_token_id = self.dictionary.pad()
        self.loader = default_loader
        self.transform = get_transforms(no_transform=self.no_transform,
                                        beit_transforms=self.beit_transforms,
                                        transform_jitter=self.transform_jitter,
                                        crop_scale=self.crop_scale)

    def setup(self, stage):
        index_files = self.get_index_files(stage)
        items = []
        self.index_files = index_files

        offset = 0
        for _index_file in index_files:
            index_file = os.path.join(self.path_to_data, _index_file)
            with open(index_file, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    data = json.loads(line)
                    items.append(data)
                print("Load %d image-text pairs from %s. " % (len(items) - offset, index_file))
                offset = len(items)
        self.items = items


    @staticmethod
    def get_index_files(split):
        raise NotImplementedError()

    def _get_image(self, image_path: str):
        image_path = os.path.join(self.data_path, image_path)
        image = self.loader(image_path)
        return self.transform(image)

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.bpe_encoder.encode(text_segment)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens

    def _get_image_text_example(self, index: int, data: dict):
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img

        text_segment = item["text_segment"]
        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        data["language_tokens"] = language_tokens
        data["padding_mask"] = padding_mask

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data)
        return data

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = '{' + "\n  Number of items: %s," % self.__len__()
        body += "\n  data root = %s," % self.data_path
        body += "\n  split = %s," % self.split
        body += "\n  dataset index files = %s" % str(self.index_files)
        body += "\n  num max bpe tokens = %s" % self.num_max_bpe_tokens
        body += "\n  transforms = ["
        for t in self.transform:
            body += "\n    %s" % str(t)
        body += "\n  ]"
        body += "\n}"

        return head + body
    
    def collater(self, samples):
        batch_tensors = {}
        for tensor_key in samples[0]:
            if isinstance(samples[0][tensor_key], torch.Tensor):
                batch_tensors[tensor_key] = torch.stack([d[tensor_key] for d in samples])
            else:
                batch_tensors[tensor_key] = torch.tensor([d[tensor_key] for d in samples], dtype=torch.long)

        return batch_tensors
    
    def train_dataloader(self):
        return DataLoader(self.dataset['train'],
                          collate_fn=self.collater,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=None,
                          shuffle=True,
                          drop_last=True,)
    
    def val_dataloader(self):
        return DataLoader(self.dataset['val'],
                          collate_fn=self.collater,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=None,
                          shuffle=True,
                          drop_last=True,)
    
    def test_dataloader(self):
        return DataLoader(self.dataset['test'],
                          collate_fn=self.collater,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=None,
                          shuffle=True,
                          drop_last=True,)
    

class COCOCaptions(BaseImageText):        
    def prepare_data(self):
        self.path_to_data = os.path.join(self.data_path, "coco")
        super().prepare_data()

        os.makedirs(self.path_to_data, exist_ok=True)
        for url in ["http://images.cocodataset.org/zips/train2014.zip",
                    "http://images.cocodataset.org/zips/val2014.zip",
                    "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"]:
        
            filename = curl_dataset(url)
            os.system(f"unzip {filename} -d {self.path_to_data}")
            os.remove(filename)
        os.remove('dataset_flickr8k.json')
        os.remove('dataset_flickr30k.json')

        self.make_coco_captioning_dataset_index()

    def setup(self, stage):
        super().setup(stage)

    @staticmethod
    def get_index_files(split):
        return (f"coco_captioning.{split}.jsonl", )

    def __getitem__(self, index: int):
        data = dict()
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img
        data["image_id"] = item["image_id"]

        text_segment = item["text_segment"]
        if text_segment is not None:
            language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
            data["language_tokens"] = language_tokens
            data["padding_mask"] = padding_mask
        return data
    
    def _make_captioning_coco_karpathy_dataset_index(
            self,
            split=("train", "restval"), 
            split_name="train",
        ):
        coco_karpathy_split_json_file = os.path.join(self.path_to_data, "dataset_coco.json")
        items = []
        image_counter = set()
        print("read %s" % coco_karpathy_split_json_file)
        with open(coco_karpathy_split_json_file, mode="r", encoding="utf-8") as reader:
            data = json.loads(reader.read())
            for item in data["images"]:
                if item["split"] in split:
                    image_path = os.path.join(item["filepath"], item["filename"])
                    if item["split"] in ["train", "restval"]:
                        for sent in item["sentences"]:
                            token_ids = self.bpe_encoder.encode(sent["raw"])
                            items.append({
                                    "image_path": image_path, 
                                    "text_segment": token_ids, 
                                    "image_id": item["cocoid"], 
                            })
                    else:
                        items.append({
                                    "image_path": image_path, 
                                    "text_segment": None, 
                                    "image_id": item["cocoid"], 
                        })
                    if image_path not in image_counter:
                        image_counter.add(image_path)
        print("Find %d images and %d image-text pairs for karpathy dataset %s split !" % \
            (len(image_counter), len(items), split_name))
        index_file = os.path.join(self.path_to_data, "coco_captioning.%s.jsonl" % split_name)
        _write_data_into_jsonl(items, index_file)


    def make_coco_captioning_dataset_index(self):
        self._make_captioning_coco_karpathy_dataset_index(split=("train", "restval"), split_name="train")
        self._make_captioning_coco_karpathy_dataset_index(split=("val", ), split_name="val")
        self._make_captioning_coco_karpathy_dataset_index(split=("test", ), split_name="test")

    # def make_nocaps_captioning_dataset_index(self):
    #     _make_nocaps_dataset_index(split="val")
    #     _make_nocaps_dataset_index(split="test")
    

class VisualGenome(BaseImageText):

    def prepare_data(self):
        super().prepare_data()
        self.path_to_data = os.path.join(self.data_path, "vg")
        os.makedirs(self.path_to_data, exist_ok=True)
        for url in ["https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
                    "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
                    "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip"]:
            filename = curl_dataset(url)
            os.system(f"unzip {filename} -d {self.path_to_data}")
            os.remove(filename)

        self.make_visual_genome_dataset_index()

    def setup(self, stage):
        super().setup(stage)

    @staticmethod
    def get_index_files(split):
        return (f"visual_genome.jsonl", ) # only for pretraining, so no splits

    def make_visual_genome_dataset_index(self):
        with open(os.path.join(self.path_to_data, "region_descriptions.json"), "r") as fp:
            region_descriptions = json.load(fp)

        items = []

        for image_meta in tqdm(region_descriptions, total=len(region_descriptions)):
            image_path = os.path.join(self.path_to_data, "VG_100K", f"{image_meta["id"]}.jpg")
            caption = ""
            for region in image_meta["regions"]:
                caption += region["phrase"] + " "
            
            token_ids = self.bpe_encoder.encode(caption.strip())
            # truncation will also be done when reading the data, but there we also substract 2 for the special tokens
            # so we already do it here to save time and memory
            token_ids = token_ids[:self.num_max_bpe_tokens - 2]
            items.append({
                "image_path": image_path, 
                "text_segment": token_ids,
                "image_id": image_meta["id"], 
            })

        _write_data_into_jsonl(items, os.path.join(self.path_to_data, "visual_genome.jsonl"))

    def __getitem__(self, index: int):
        data = dict()
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img
        data["image_id"] = item["image_id"]

        text_segment = item["text_segment"]
        if text_segment is not None:
            language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
            data["language_tokens"] = language_tokens
            data["padding_mask"] = padding_mask
        return data
    

class VQAv2(BaseImageText):
    def __init__(self, 
                 data_path,
                 split,
                 num_max_bpe_tokens,
                 transform_jitter=False,
                 beit_transforms=False,
                 no_transform=False,
                 task=None,
                 crop_scale=(0.6, 1.0)):
        
        super().__init__(data_path,
                         split,
                         num_max_bpe_tokens,
                         transform_jitter,
                         beit_transforms,
                         no_transform,
                         task,
                         crop_scale)
        
        ans2label_file = os.path.join(data_path, "answer2label.txt")
        ans2label = {}
        label2ans = []
        with open(ans2label_file, mode="r", encoding="utf-8") as reader:
            for i, line in enumerate(reader):
                data = json.loads(line)
                ans = data["answer"]
                label = data["label"]
                label = int(label)
                assert label == i
                ans2label[ans] = i
                label2ans.append(ans)
        
        self.ans2label = ans2label
        self.label2ans = label2ans

    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return ("vqa.train.jsonl", "vqa.trainable_val.jsonl")
        elif split == "val":
            return ("vqa.rest_val.jsonl", )
        elif split == "test":
            return ("vqa.test.jsonl", )
        elif split == "test-dev":
            return ("vqa.test-dev.jsonl", )            
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        if "labels" in self.items[index] and len(self.items[index]["labels"]) > 0:
            labels = [0.] * len(self.label2ans)
            for l, s in zip(self.items[index]["labels"], self.items[index]["scores"]):
                labels[l] = s
            data["labels"] = torch.FloatTensor(labels)
        else:
            data["qid"] = self.items[index]["qid"]
        return data
    

class NLVR2Dataset(BaseImageText):
    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        item = self.items[index]
        img_path = item["image2_path"]
        img = self._get_image(img_path)
        data["image2"] = img
        data["label"] = self.items[index]["label"]
        return data