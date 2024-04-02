import os
import json
import torch
from typing import Tuple
import pandas as pd
from torchvision.datasets.folder import default_loader
from datasets.data_utils import get_transforms, write_data_into_jsonl, download_and_unzip, convert_mp3_to_flac
from utils.glossary import normalize_word
from tqdm import tqdm
import os
import json
import random
import glob
from collections import defaultdict, Counter

from .base_datasets import BaseImageText, BaseTextAudio, BaseImageAudio
    

class COCOCaptions(BaseImageText):        
    def __init__(
            self,
            data_path,
            split,
            num_max_bpe_tokens,
            task="captioning",
            transform_jitter=False,
            beit_transforms=False,
            no_transform=False,
            crop_scale=(0.6, 1.0),
            ):
        assert task in ["captioning", "retrieval"]
        self.task = task
        if self.task == "retrieval": # yields no augmentation, as retrieval is zero-shot (testing)
            transform_jitter = False
            beit_transforms = False
            no_transform = True
        super().__init__(data_path, split, num_max_bpe_tokens, transform_jitter, beit_transforms, no_transform, crop_scale)
        self.path_to_data = os.path.join(self.data_path, "coco")        

        os.makedirs(self.path_to_data, exist_ok=True)
        urls = ["http://images.cocodataset.org/zips/train2014.zip",
                "http://images.cocodataset.org/zips/val2014.zip",
                "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"]
        download_and_unzip(urls=urls, store_at=self.path_to_data)
        os.remove(os.path.join(self.path_to_data, 'dataset_flickr8k.json'))
        os.remove(os.path.join(self.path_to_data, 'dataset_flickr30k.json'))

        self._make_coco_karpathy_dataset_index()

    def get_index_files(self):
        return (f"coco_{self.task}.{self.split}.jsonl", )

    def __getitem__(self, index: int):
        if self.task == "captioning":
            return self._get_item_captioning(index)
        else:
            return self._get_item_retrieval(index)
    
    def _get_item_captioning(self, index):
        data = dict()
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img
        data["id"] = item["id"]

        text_segment = item["text_segment"]
        if text_segment is not None:
            language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
            data["language_tokens"] = language_tokens
            data["padding_mask"] = padding_mask
        return data
    
    def _get_item_retrieval(self, index):
        data = super().__getitem__(index)
        data["id"] = self.items[index]["id"]
        return data
    
    def _make_coco_karpathy_dataset_index(self):
        if self.split == "train":
            karpathy_split = ("train", "restval")
        elif self.split == "val":
            karpathy_split = ("val", )
        elif self.split == "test":
            karpathy_split = ("test", )
        else:
            raise RuntimeError("split %s is not found!" % self.split)
        
        bpe_encoder = self.get_bpe_encoder()
        
        coco_karpathy_split_json_file = os.path.join(self.path_to_data, "dataset_coco.json")
        items = []
        image_counter = set()
        self.log("read %s" % coco_karpathy_split_json_file)
        self.log("task is %s" % self.task)
        with open(coco_karpathy_split_json_file, mode="r", encoding="utf-8") as reader:
            data = json.loads(reader.read())
            for item in data["images"]:
                if item["split"] in karpathy_split:
                    image_path = os.path.join(item["filepath"], item["filename"])
                    if self.task == "captioning":
                        if item["split"] in ["train", "restval"]:
                            items += self._encode_all(item, image_path, image_counter, bpe_encoder)
                        else:
                            items.append({
                                        "image_path": image_path, 
                                        "text_segment": None, 
                                        "id": item["cocoid"], 
                            })
                    else:
                        items += self._encode_all(item, image_path, image_counter, bpe_encoder)
                    if image_path not in image_counter:
                        image_counter.add(image_path)
        self.log("Find %d images and %d image-text pairs for karpathy dataset %s split !" % \
            (len(image_counter), len(items), self.split))
        index_file = os.path.join(self.path_to_data, f"coco_{self.task}.{self.split}.jsonl")
        write_data_into_jsonl(items, index_file)

    def _encode_all(self, item, image_path, image_counter, bpe_encoder):
        return [
            {
                "image_path": image_path,
                "text_segment": bpe_encoder.encode(sent["raw"]),
                "id": len(image_counter) if self.task=="retrieval" else item["cocoid"],
            }
            for sent in item["sentences"]
            ]

    # def make_nocaps_captioning_dataset_index(self):
    #     _make_nocaps_dataset_index(split="val")
    #     _make_nocaps_dataset_index(split="test")

class Flickr30Dataset(BaseImageText):
    def __init__(self, 
                 data_path,
                 split,
                 num_max_bpe_tokens,
                 ):
        # yields no augmentation, as Flickr is zero-shot retrieval (testing)
        super().__init__(data_path, split, num_max_bpe_tokens, transform_jitter=False, beit_transforms=False, no_transform=True)
        self.path_to_data = os.path.join(self.data_path, "flickr30k")

        os.makedirs(self.path_to_data, exist_ok=True)
        download_and_unzip(urls=["https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"], store_at=self.path_to_data)
        os.remove(os.path.join(self.path_to_data, 'dataset_flickr8k.json'))
        os.remove(os.path.join(self.path_to_data, 'dataset_coco.json'))
        
        self.make_flickr30k_dataset_index()

    def get_index_files(self):
        if self.split == "train":
            return (f"flickr30k.train.jsonl", )
        elif self.split == "val":
            return (f"flickr30k.val.jsonl", )
        elif self.split == "test":
            return (f"flickr30k.test.jsonl", )
        else:
            raise RuntimeError("split %s is not found!" % self.split)
        
    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        data["id"] = self.items[index]["id"]
        return data

    def make_flickr30k_dataset_index(self):

        with open(os.path.join(self.path_to_data, "dataset_flickr30k.json"), "r") as reader:
            captions = json.loads(reader.read())

        bpe_encoder = self.get_bpe_encoder()

        captions = captions["images"]
        index = []
        n_images = 0

        for each_item in captions:
            image_path = os.path.join("flickr30k-images", each_item["filename"])

            if each_item["split"] != self.split:
                continue

            for text_segment in each_item["sentences"]: 
                index.append({
                    "image_path": image_path, 
                    "text_segment": bpe_encoder.encode(text_segment["raw"]), 
                    "id": len(index), 
                })
            n_images = 0

        self.log(f"{n_images} images and {len(index)} image-text pairs!")
        write_data_into_jsonl(index, os.path.join(self.path_to_data, f"flickr30k.{self.split}.jsonl"))


class Flickr8KAudioDataset(BaseImageAudio):
    def __init__(self, 
                 data_path,
                 split,
                 transform_jitter:bool,
                 beit_transforms:bool,
                 no_transform:bool,
                 crop_scale:Tuple[float, float],
                 sample_rate:int,
                 max_sample_size:int,
                 min_sample_size:int,
                 normalize:bool,
                 pad:bool,
                 **precompute_mask_config
                 ):
        super().__init__(data_path=data_path,
                         split=split,
                         transform_jitter=False,
                         beit_transforms=False,
                         no_transform=True,
                         crop_scale=crop_scale, # ignored
                         sample_rate=sample_rate,
                         max_sample_size=max_sample_size,
                         min_sample_size=min_sample_size,
                         normalize=normalize,
                         pad=pad,
                         **precompute_mask_config)
        self.path_to_data = os.path.join(self.data_path, "flickr8k_audio")

        os.makedirs(self.path_to_data, exist_ok=True)
        download_and_unzip(urls=["https://groups.csail.mit.edu/sls/downloads/flickraudio/downloads/flickr_audio.tar.gz"], store_at=self.path_to_data,
                           archive_type="tar.gz")
        download_and_unzip(urls=["https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"], store_at=self.path_to_data)
        os.remove(os.path.join(self.path_to_data, 'dataset_flickr30k.json'))
        os.remove(os.path.join(self.path_to_data, 'dataset_coco.json'))

        self.transform = get_transforms(no_transform=True)
        self.loader = default_loader
        
        self.make_flickr8k_audio_dataset_index()

    def get_index_files(self):
        if self.split == "train":
            return (f"flickr8k_audio.train.jsonl", )
        elif self.split == "val":
            return (f"flickr8k_audio.val.jsonl", )
        elif self.split == "test":
            return (f"flickr8k_audio.test.jsonl", )
        else:
            raise RuntimeError("split %s is not found!" % self.split)

    def make_flickr8k_audio_dataset_index(self):
        with open(os.path.join(self.path_to_data, "dataset_flickr8k.json"), "r", encoding='utf-8') as reader:
            meta_data = json.loads(reader.read())["images"]

        image_to_audio = {}
        with open(filename, 'r') as file:
            for line in file:
                wav_file, jpg_file, _ = line.strip().split()
                if jpg_file in image_to_audio:
                    image_to_audio[jpg_file].append(wav_file)
                else:
                    image_to_audio[jpg_file] = [wav_file]
        
        index = []
        n_images = 0

        for each_item in meta_data:
            if each_item["split"] != self.split:
                continue
            filename = each_item["filename"]
            image_path = os.path.join(self.data_path, "flickr8k-images", filename)

            try:
                audio_names = image_to_audio[filename]
            except KeyError:
                raise KeyError(f"no audio for {filename}")

            for audio_name in audio_names:
                index.append({
                    "image_path": image_path, 
                    "audio_path": os.path.join(self.data_path, "flickr_audio", "wavs", audio_name), 
                    "id": each_item['imgid'],
                })
            n_images += 1

        self.log(f"{n_images} image and {len(index)} image-audio pairs!")
        write_data_into_jsonl(index, os.path.join(self.path_to_data, "flickr8k_audio.%s.jsonl" % self.split))
    

class VisualGenome(BaseImageText):
    def __init__(
            self,
            data_path,
            split,
            num_max_bpe_tokens,
            transform_jitter=False,
            beit_transforms=False,
            no_transform=False,
            crop_scale=(0.6, 1.0),
            ):
        super().__init__(data_path, split, num_max_bpe_tokens, transform_jitter, beit_transforms, no_transform, crop_scale)
        self.path_to_data = os.path.join(self.data_path, "vg")
        os.makedirs(self.path_to_data, exist_ok=True)
        urls = ["https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
                "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
                "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip"]
        download_and_unzip(urls=urls, store_at=self.path_to_data)

        urls = ["http://images.cocodataset.org/zips/train2014.zip",
                "http://images.cocodataset.org/zips/val2014.zip",
                "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"]
        download_and_unzip(urls=urls, store_at=self.path_to_data)

        self.make_visual_genome_dataset_index()

    def get_index_files(self):
        return (f"visual_genome.jsonl", ) # only for pretraining, so no splits

    def make_visual_genome_dataset_index(self):
        with open(os.path.join(self.path_to_data, "region_descriptions.json"), "r") as fp:
            region_descriptions = json.load(fp)

        bpe_encoder = self.get_bpe_encoder()

        items = []

        for image_meta in tqdm(region_descriptions, total=len(region_descriptions)):
            image_path = os.path.join(self.path_to_data, "VG_100K", f"{image_meta['id']}.jpg")
            caption = ' '.join([region["phrase"] for region in image_meta["regions"]])
            
            token_ids = bpe_encoder.encode(caption.strip())
            # truncation will also be done when reading the data, but there we also substract 2 for the special tokens
            # so we already do it here to save time and memory
            token_ids = token_ids[:self.num_max_bpe_tokens - 2]
            items.append({
                "image_path": image_path, 
                "text_segment": token_ids,
                "id": image_meta["id"], 
            })

        write_data_into_jsonl(items, os.path.join(self.path_to_data, "visual_genome.jsonl"))

    def __getitem__(self, index: int):
        data = dict()
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img
        data["id"] = item["id"]

        text_segment = item["text_segment"]
        if text_segment is not None:
            language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
            data["language_tokens"] = language_tokens
            data["padding_mask"] = padding_mask
        return data
    

class VQAv2(BaseImageText):
    def __init__(
            self,
            data_path,
            split,
            num_max_bpe_tokens,
            transform_jitter=False,
            beit_transforms=False,
            no_transform=False,
            crop_scale=(0.6, 1.0),
            ):
        super().__init__(data_path, split, num_max_bpe_tokens, transform_jitter, beit_transforms, no_transform, crop_scale)
        self.path_to_data = os.path.join(self.data_path, "coco")
        os.makedirs(self.path_to_data, exist_ok=True)

        download_and_unzip(urls=["http://images.cocodataset.org/zips/test2015.zip"], store_at=self.path_to_data)

        self.path_to_data = os.path.join(self.data_path, "vqa")
        os.makedirs(self.path_to_data, exist_ok=True)
        urls = ["https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
                "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
                "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
                "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
                "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"]
        download_and_unzip(urls=urls, store_at=self.path_to_data)

        self.make_vqa_dataset_index()

    def load(self):
        super().load()
        ans2label_file = os.path.join(self.path_to_data, "answer2label.txt")
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

    def get_index_files(self):
        if self.split == "train":
            return ("vqa.train.jsonl", "vqa.trainable_val.jsonl")
        elif self.split == "val":
            return ("vqa.rest_val.jsonl", )
        elif self.split == "test":
            return ("vqa.test.jsonl", )
        elif self.split == "test-dev":
            return ("vqa.test-dev.jsonl", )            
        else:
            raise RuntimeError("split %s is not found!" % self.split)

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
    
    def get_score(self, occurences):
        if occurences == 0:
            return 0.0
        elif occurences == 1:
            return 0.3
        elif occurences == 2:
            return 0.6
        elif occurences == 3:
            return 0.9
        else:
            return 1.0

    def make_vqa_dataset_index(self):
        with open(os.path.join(self.path_to_data, "v2_OpenEnded_mscoco_train2014_questions.json"), "r") as fp:
            questions_train2014 = json.load(fp)["questions"]
        with open(os.path.join(self.path_to_data, "v2_OpenEnded_mscoco_val2014_questions.json"), "r") as fp:
            questions_val2014 = json.load(fp)["questions"]
        with open(os.path.join(self.path_to_data, "v2_OpenEnded_mscoco_test2015_questions.json"), "r") as fp:
            questions_test2015 = json.load(fp)["questions"]
        with open(os.path.join(self.path_to_data, "v2_OpenEnded_mscoco_test-dev2015_questions.json"), "r") as fp:
            questions_test_dev2015 = json.load(fp)["questions"]

        with open(os.path.join(self.path_to_data, "v2_mscoco_train2014_annotations.json"), "r") as fp:
            annotations_train2014 = json.load(fp)["annotations"]
        with open(os.path.join(self.path_to_data, "v2_mscoco_val2014_annotations.json"), "r") as fp:
            annotations_val2014 = json.load(fp)["annotations"]

        annotations = dict()

        bpe_encoder = self.get_bpe_encoder()

        for split, questions in zip(
            ["train", "val", "test", "test-dev"],
            [questions_train2014, questions_val2014, questions_test2015, questions_test_dev2015],
        ):
            _annot = defaultdict(dict)
            for q in questions:
                question_text = q["question"]
                token_ids = bpe_encoder.encode(question_text)

                assert q["question_id"] not in _annot[q["id"]]
                _annot[q["id"]][q["question_id"]] = {
                    "question": question_text, 
                    "token_ids": token_ids, 
                }

            annotations[split] = _annot

        all_major_answers = list()

        for split, annots in zip(
            ["train", "val"], [annotations_train2014, annotations_val2014],
        ):
            # _annot = annotations[split]
            for q in annots:
                all_major_answers.append(q["multiple_choice_answer"])

        all_major_answers = [normalize_word(word) for word in all_major_answers]
        counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 9}
        ans2label = {k: i for i, k in enumerate(counter.keys())}
        label2ans = list(counter.keys())

        for split, annots in zip(
            ["train", "val"], [annotations_train2014, annotations_val2014],
        ):
            _annot = annotations[split]
            for q in annots:
                answers = q["answers"]
                answer_count = {}
                for answer in answers:
                    answer_ = answer["answer"]
                    answer_count[answer_] = answer_count.get(answer_, 0) + 1

                labels = []
                scores = []
                for answer in answer_count:
                    if answer not in ans2label:
                        continue
                    labels.append(ans2label[answer])
                    score = self.get_score(answer_count[answer])
                    scores.append(score)

                assert "labels" not in _annot[q["id"]][q["question_id"]]
                assert "question" in _annot[q["id"]][q["question_id"]]
                _annot[q["id"]][q["question_id"]]["labels"] = labels
                _annot[q["id"]][q["question_id"]]["scores"] = scores

        for split in ["train", "val"]:
            filtered_annot = dict()
            for ik, iv in annotations[split].items():
                new_q = dict()
                for qk, qv in iv.items():
                    if len(qv["labels"]) != 0:
                        new_q[qk] = qv
                if len(new_q) != 0:
                    filtered_annot[ik] = new_q
            annotations[split] = filtered_annot

        split2items = {}
        for split in ["train", "val", "test", "test-dev"]:
            annot = annotations[split]
            split_name = {
                "train": "train2014",
                "val": "val2014",
                "test": "test2015",
                "test-dev": "test2015",
            }[split]
            paths = list(glob.glob(f"{os.path.join(self.data_path, 'coco')}/{split_name}/*.jpg"))
            random.shuffle(paths)
            annot_paths = [path for path in paths \
                if int(path.split("/")[-1].split("_")[-1][:-4]) in annot]

            if len(paths) == len(annot_paths):
                self.log("all images have caption annotations")
            else:
                self.log("not all images have caption annotations")
            self.log(len(paths), len(annot_paths), len(annot))

            items = []
            for path in annot_paths:
                iid = int(path.split("/")[-1].split("_")[-1][:-4])
                _annot = annotations[split][iid]
                for qid in _annot:
                    q = _annot[qid]
                    if split in ["train", "val"]:
                        labels = q["labels"]
                        scores = q["scores"]
                    else:
                        labels, scores = [], []

                    items.append({
                        "image_path": os.path.join(split_name, path.split('/')[-1]), 
                        "text_segment": q["token_ids"], 
                        "labels": labels, 
                        "scores": scores, 
                        "qid": qid, 
                    })
            split2items[split] = items

            write_data_into_jsonl(items=items, jsonl_file=os.path.join(self.path_to_data, "vqa.%s.jsonl" % split))

        # Following ViLT, we use 1000 images of the original val set as the final val set        
        val_image2items = defaultdict(list)
        for item in split2items["val"]:
            val_image2items[item["image_path"]].append(item)
        
        self.log("Contains %d image and %d pairs for val set!" % (len(val_image2items), len(split2items["val"])))

        val_images = list(val_image2items.keys())
        random.shuffle(val_images)
        trainable_val = []
        rest_val = []
        for i, image_id in enumerate(val_images):
            if i < 1000:
                rest_val += val_image2items[image_id]
            else:
                trainable_val += val_image2items[image_id]
        
        write_data_into_jsonl(items=trainable_val, jsonl_file=os.path.join(self.path_to_data, "vqa.trainable_val.jsonl"))
        write_data_into_jsonl(items=rest_val, jsonl_file=os.path.join(self.path_to_data, "vqa.rest_val.jsonl"))

        with open(os.path.join(self.path_to_data, "answer2label.txt"), mode="w", encoding="utf-8") as writer:
            for ans in ans2label:
                to_json = {
                    "answer": ans, 
                    "label": ans2label[ans]
                }
                writer.write("%s\n" % json.dumps(to_json))
    

class NLVR2(BaseImageText):
    def __init__(
            self,
            data_path,
            split,
            num_max_bpe_tokens,
            transform_jitter=False,
            beit_transforms=False,
            no_transform=False,
            crop_scale=(0.6, 1.0),
            ):
        super().__init__(data_path, split, num_max_bpe_tokens, transform_jitter, beit_transforms, no_transform, crop_scale)
        self.path_to_data = os.path.join(self.data_path, "nlvr2")
        os.makedirs(self.path_to_data, exist_ok=True)
        # TODO: download and unzip the data
        #
        # self.make_dataset_index(nlvr_repo_path)

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        item = self.items[index]
        img_path = item["image2_path"]
        img = self._get_image(img_path)
        data["image2"] = img
        data["label"] = self.items[index]["label"]
        return data
    
    def get_index_files(self):
        if self.split == "train":
            return ("nlvr2.train.index.jsonl", )
        elif self.split == "val":
            return ("nlvr2.dev.index.jsonl", )
        elif self.split == "test":
            return ("nlvr2.test-P.index.jsonl", )
        else:
            raise RuntimeError("split %s is not found!" % self.split)

    def _preprocess_json(self, prefix, json_file, index_file):
        items = []
        bpe_encoder = self.get_bpe_encoder()
        with open(json_file, mode="r", encoding="utf-8") as reader:
            for line in reader:
                data = json.loads(line)
                path = os.path.join(prefix, str(data["directory"])) if "directory" in data else prefix
                path = os.path.join(path, "-".join(data["identifier"].split("-")[:-1]))
                token_ids = bpe_encoder.encode(data["sentence"])
                items.append({
                    "image_path": path + "-img0.png",
                    "image2_path": path + "-img1.png",
                    "text_segment": token_ids,
                    "label": 1 if data["label"] == "True" else 0,
                    "identifier": data["identifier"], 
                })
        write_data_into_jsonl(items, index_file)

    def make_dataset_index(self, nlvr_repo_path):
        if self.split == "train":
            prefix = "images/train"
            json_file = os.path.join(nlvr_repo_path, "nlvr2/data/train.json")
        elif self.split == "val":
            prefix = "dev"
            json_file = os.path.join(nlvr_repo_path, "nlvr2/data/dev.json")
        elif self.split == "test":
            prefix = "test1"
            json_file = os.path.join(nlvr_repo_path, "nlvr2/data/test1.json")
        else:
            raise RuntimeError("split %s is not found!" % self.split)
        
        index_file = os.path.join(self.path_to_data, self.get_index_files()[0])

        self._preprocess_json(prefix=prefix, json_file=json_file, index_file=index_file)


class CommonVoice(BaseTextAudio):
    def __init__(
            self,
            data_path:str,
            split:str,
            num_max_bpe_tokens:int,
            sample_rate:int,
            max_sample_size:int,
            min_sample_size:int,
            normalize:bool,
            pad:bool,
            **precompute_mask_config,
            ):
        super().__init__(data_path, split, num_max_bpe_tokens, sample_rate, max_sample_size, min_sample_size, normalize, pad,
                         **precompute_mask_config)

        cv_dir_name_pattern = os.path.join(self.data_path, 'cv-corpus-*')
        dir_name = [d for d in glob.glob(cv_dir_name_pattern) if os.path.isdir(d)][0]
        self.path_to_data = os.path.join(dir_name, 'en')
        self.path_to_clips = os.path.join(self.path_to_data, "clips")

        self.make_common_voice_dataset_index()

    def get_index_files(self):
        if self.split == "train":
            return (f"common_voice.train.jsonl", )
        else:
            return (f"common_voice.retrieval.jsonl", )    

    def make_common_voice_dataset_index(self):
        validated_data = pd.read_csv(os.path.join(self.path_to_data, 'validated.tsv'), sep='\t')[['client_id', 'sentence_id', 'path', 'sentence']]
        take_train = int(len(validated_data)*0.9)
        if self.split == "train":
            validated_data = validated_data.iloc[:take_train]
        else:
            validated_data = validated_data.iloc[take_train:] # for retrieval

        convert_mp3_to_flac(self.path_to_clips, validated_data['path'].tolist())

        pattern = os.path.join(self.path_to_clips, "*.mp3")

        # Use glob to find all files matching the pattern
        files = glob.glob(pattern)

        # Iterate over the list of file paths & remove each file
        for file in files:
            os.remove(file)
        self.log(f'Removed {len(files)} unused mp3 files.')

        bpe_encoder = self.get_bpe_encoder()

        items = []
        for i, row in validated_data.iterrows():
            path = os.path.join(self.path_to_clips, row['path'].replace('.mp3', '.flac'))
            token_ids = bpe_encoder.encode(row['sentence'])
            items.append({
                "audio_path": path,
                "text_segment": token_ids,
                "id": i,
                "client_id": row['client_id'],
                "sentence_id": row['sentence_id'],
            })
        write_data_into_jsonl(items, os.path.join(self.path_to_data, f"common_voice.{self.split}.jsonl"))
