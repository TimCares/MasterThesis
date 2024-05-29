import os
import json
import torch
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from torchvision.datasets.folder import default_loader
from datasets_.data_utils import get_transforms, write_data_into_jsonl, download_and_unzip, convert_mp3_to_flac
from datasets_.glossary import normalize_word
from tqdm import tqdm
import os
import json
import random
import glob
from collections import defaultdict, Counter
import soundfile as sf
import random

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
        super().__init__(data_path=data_path, 
                         split=split, 
                         num_max_bpe_tokens=num_max_bpe_tokens, 
                         transform_jitter=transform_jitter, 
                         beit_transforms=beit_transforms, 
                         no_transform=no_transform, 
                         crop_scale=crop_scale)
        self.path_to_data = os.path.join(self.data_path, "coco")        

        os.makedirs(self.path_to_data, exist_ok=True)
        if self.index_exists(dataset_path=self.path_to_data):
            return

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

        text_segment = item["text"]
        if text_segment is not None:
            language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
            data["text"] = language_tokens
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
        self.log("task is: %s" % self.task)
        with open(coco_karpathy_split_json_file, mode="r", encoding="utf-8") as reader:
            data = json.loads(reader.read())
            for item in data["images"]:
                if item["split"] in karpathy_split:
                    image_path = os.path.join(self.path_to_data, item["filepath"], item["filename"])
                    if self.task == "captioning":
                        if item["split"] in ["train", "restval"]:
                            items += self._encode_all(item, image_path, image_counter, bpe_encoder)
                        else:
                            items.append({
                                        "image_path": image_path, 
                                        "text": None, 
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
                "text": bpe_encoder.encode(sent["raw"]),
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
        super().__init__(data_path=data_path, 
                         split=split, 
                         num_max_bpe_tokens=num_max_bpe_tokens, 
                         transform_jitter=False, 
                         beit_transforms=False, 
                         no_transform=True)
        self.path_to_data = os.path.join(self.data_path, "flickr30k")

        os.makedirs(self.path_to_data, exist_ok=True)
        if self.index_exists(dataset_path=self.path_to_data):
            return
        
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
            image_path = os.path.join(self.path_to_data, "flickr30k-images", each_item["filename"])

            if each_item["split"] != self.split:
                continue

            for text_segment in each_item["sentences"]: 
                index.append({
                    "image_path": image_path, 
                    "text": bpe_encoder.encode(text_segment["raw"]), 
                    "id": len(index), 
                })
            n_images = 0

        self.log(f"{n_images} images and {len(index)} image-text pairs!")
        write_data_into_jsonl(index, os.path.join(self.path_to_data, f"flickr30k.{self.split}.jsonl"))


class Flickr8KAudioDataset(BaseImageAudio):
    def __init__(self, 
                 data_path,
                 split,
                 sample_rate:int,
                 max_sample_size:int,
                 min_sample_size:int,
                 normalize:bool,
                 pad:bool,
                 precompute_mask_config:Dict[str, Any]={},
                 ):
        super().__init__(data_path=data_path,
                         split=split,
                         transform_jitter=False,
                         beit_transforms=False,
                         no_transform=True,
                         sample_rate=sample_rate,
                         max_sample_size=max_sample_size,
                         min_sample_size=min_sample_size,
                         normalize=normalize,
                         pad=pad,
                         precompute_mask_config=precompute_mask_config)
        self.path_to_data = os.path.join(self.data_path, "flickr8k")

        os.makedirs(self.path_to_data, exist_ok=True)
        if self.index_exists(dataset_path=self.path_to_data):
            return
        
        download_and_unzip(urls=["https://groups.csail.mit.edu/sls/downloads/flickraudio/downloads/flickr_audio.tar.gz"], store_at=self.path_to_data,
                            archive_type="tar.gz")

        download_and_unzip(urls=["https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"], store_at=self.path_to_data)
        os.remove(os.path.join(self.path_to_data, 'dataset_flickr30k.json'))
        os.remove(os.path.join(self.path_to_data, 'dataset_coco.json'))

        self.transform = get_transforms(no_transform=True)
        self.loader = default_loader
        
        self.make_flickr8k_dataset_index()

    def get_index_files(self):
        if self.split == "train":
            return (f"flickr8k.train.jsonl", )
        elif self.split == "val":
            return (f"flickr8k.val.jsonl", )
        elif self.split == "test":
            return (f"flickr8k.test.jsonl", )
        else:
            raise RuntimeError("split %s is not found!" % self.split)

    def make_flickr8k_dataset_index(self):
        with open(os.path.join(self.path_to_data, "dataset_flickr8k.json"), "r", encoding='utf-8') as reader:
            meta_data = json.loads(reader.read())["images"]

        image_to_audio = {}
        n_skipped = 0
        with open(os.path.join(self.path_to_data, "flickr_audio", "wav2capt.txt"), 'r', encoding='utf-8') as reader:
            for line in reader:
                wav_file, jpg_file, _ = line.strip().split()

                audio, _ = sf.read(os.path.join(self.path_to_data, "flickr_audio", "wavs", wav_file), dtype="float32")
                audio = torch.from_numpy(audio).float()
                if len(audio) < self.min_sample_size:
                    n_skipped+=1
                    continue

                if jpg_file in image_to_audio:
                    image_to_audio[jpg_file].append(wav_file)
                else:
                    image_to_audio[jpg_file] = [wav_file]
        
        self.log(f"Skipped {n_skipped} audio samples with length < {self.min_sample_size}")
        index = []
        n_images = 0

        for each_item in meta_data:
            if each_item["split"] != self.split:
                continue
            filename = each_item["filename"]

            try:
                audio_names = image_to_audio[filename]
            except KeyError:
                raise KeyError(f"Flickr8kAudio: No audio for {filename}")

            for audio_name in audio_names:
                index.append({
                    "image_path": os.path.join(self.path_to_data, "flickr8k-images", filename), 
                    "audio_path": os.path.join(self.path_to_data, "flickr_audio", "wavs", audio_name), 
                    "id": each_item['imgid'],
                })
            n_images += 1

        self.log(f"{n_images} image and {len(index)} image-audio pairs!")
        write_data_into_jsonl(index, os.path.join(self.path_to_data, "flickr8k.%s.jsonl" % self.split))
    

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
            n_caption_groups:int=1,
            ):
        super().__init__(data_path=data_path, 
                         split=split, 
                         num_max_bpe_tokens=num_max_bpe_tokens, 
                         transform_jitter=transform_jitter, 
                         beit_transforms=beit_transforms, 
                         no_transform=no_transform, 
                         crop_scale=crop_scale)
        self.path_to_data = os.path.join(self.data_path, "vg")
        self.n_caption_groups = n_caption_groups
        os.makedirs(self.path_to_data, exist_ok=True)
        if self.index_exists(dataset_path=self.path_to_data):
            return
        
        urls = ["https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
                "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
                "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip"]
        download_and_unzip(urls=urls, store_at=self.path_to_data)

        urls = ["http://images.cocodataset.org/zips/train2014.zip",
                "http://images.cocodataset.org/zips/val2014.zip",
                "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"]
        download_and_unzip(urls=urls, store_at=self.path_to_data)

        self.bpe_encoder = self.get_bpe_encoder()

        self.make_visual_genome_dataset_index()

    def get_index_files(self):
        postfix = f'group_{self.n_caption_groups}' if self.n_caption_groups > 1 else ''
        return (f"visual_genome_{postfix}.jsonl", ) # only for pretraining, so no splits

    def make_visual_genome_dataset_index(self):
        with open(os.path.join(self.path_to_data, "region_descriptions.json"), "r") as fp:
            region_descriptions = json.load(fp)

        if self.n_caption_groups > 1:
            items = self._make_index_with_caption_groups(region_descriptions)
        else:
            items = self._make_index_with_single_caption(region_descriptions)
        
        write_data_into_jsonl(items, os.path.join(self.path_to_data, self.get_index_files()[0]))

    def _make_index_with_single_caption(self, region_descriptions):
        items = []
        for image_meta in tqdm(region_descriptions, total=len(region_descriptions)):
            image_path = os.path.join(self.path_to_data, "VG_100K", f"{image_meta['id']}.jpg")

            captions = [region["phrase"].strip() for region in image_meta["regions"]]
            
            items.append({
                "image_path": image_path, 
                "text": captions,
                "id": image_meta["id"], 
            })
            
        return items

    def _make_index_with_caption_groups(self, region_descriptions):

        def split_by_threshold(lst):
            groups = []
            current_group = []
            current_sum = 0
            
            for i in range(len(lst)):
                value = lst[i][0]
                if i != len(lst)-1: # if not last caption,...
                    value += 1 # ... then substract one because "and" token is later added to combine multiple captions
                if current_sum + value > self.num_max_bpe_tokens - 2: # -2 for bos and eos
                    groups.append(current_group)
                    current_group = [lst[i][1]]
                    current_sum = value
                else:
                    current_group.append(lst[i][1])
                    current_sum += value
            
            if current_group:
                groups.append(current_group)
            
            return groups
        
        items = []
        for image_meta in tqdm(region_descriptions, total=len(region_descriptions)):
            image_path = os.path.join(self.path_to_data, "VG_100K", f"{image_meta['id']}.jpg")

            n_tokens_list:List[Tuple[int, str]] = []
            for region in image_meta["regions"]:
                desc = region['phrase'].strip().lower()
                n_tokens = len(self.bpe_encoder.bpe.encode(desc))
                n_tokens_list.append((n_tokens, desc))
            
            caption_groups:List[List[str]] = split_by_threshold(n_tokens_list)

            s = self.n_caption_groups
            for caption_group_sample in [caption_groups[i:i + s] for i in range(0, len(caption_groups), s)]:
                if len(caption_group_sample) < s: # we drop the last group if it is not complete
                    continue
                item = {
                    "image_path": image_path,
                    "id": image_meta["id"],
                }
                # caption_group_sample is list of list of captions,
                # latter list will be reduced to one string/caption, 
                # first list will be used together with "repeat_interleave" for the image features
                for i, caption_group in enumerate(caption_group_sample):
                    item[f"text{i}"] = caption_group

                items.append(item)
        
        return items

    def __getitem__(self, index: int):
        data = dict()
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img
        data["id"] = item["id"]

        if self.n_caption_groups > 1:
            text_segment_keys = {k for k in item.keys() if k.startswith("text")}
            for i, key in enumerate(text_segment_keys):
                captions_to_use:List[str] = item[key].copy()
                random.shuffle(captions_to_use)
                full_caption = " and ".join(captions_to_use)
                text_segment = self.bpe_encoder.encode(full_caption)
                language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
                assert len(language_tokens) == self.num_max_bpe_tokens and len(padding_mask) == self.num_max_bpe_tokens
                data[f"text_{i}"] = language_tokens
                data[f"padding_mask_{i}"] = padding_mask
            
        else:
            text_segment = random.choice(item["text"]) # "item['text']" is a list of captions, so we randomly select one
            text_segment = self.bpe_encoder.encode(text_segment)
            language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
            data["text"] = language_tokens
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
        super().__init__(data_path=data_path, 
                         split=split, 
                         num_max_bpe_tokens=num_max_bpe_tokens, 
                         transform_jitter=transform_jitter, 
                         beit_transforms=beit_transforms, 
                         no_transform=no_transform, 
                         crop_scale=crop_scale)
        self.path_to_data = os.path.join(self.data_path, "coco")
        os.makedirs(self.path_to_data, exist_ok=True)
        if not os.path.exists(self.path_to_data, 'test2015'):
            download_and_unzip(urls=["http://images.cocodataset.org/zips/test2015.zip"], store_at=self.path_to_data)

        self.path_to_data = os.path.join(self.data_path, "vqa")
        os.makedirs(self.path_to_data, exist_ok=True)
        if self.index_exists(dataset_path=self.path_to_data):
            return
        
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
        if "targets" in self.items[index] and len(self.items[index]["targets"]) > 0:
            targets = [0.] * len(self.label2ans)
            for l, s in zip(self.items[index]["targets"], self.items[index]["scores"]):
                targets[l] = s
            data["targets"] = torch.FloatTensor(targets)
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
            paths = list(glob.glob(f"{self.path_to_data}/{split_name}/*.jpg"))
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
                        "image_path": path, 
                        "text": q["token_ids"], 
                        "target": labels, 
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
        super().__init__(data_path=data_path, 
                         split=split, 
                         num_max_bpe_tokens=num_max_bpe_tokens, 
                         transform_jitter=transform_jitter, 
                         beit_transforms=beit_transforms, 
                         no_transform=no_transform, 
                         crop_scale=crop_scale)
        self.path_to_data = os.path.join(self.data_path, "nlvr2")
        assert os.path.exists(os.path.join(self.path_to_data, 'images')), f"Data not found, "
        f"please download and add the NLVR2 data to this directory: {self.path_to_data}"
        
        self.make_dataset_index()

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        item = self.items[index]
        img_path = item["image2_path"]
        img = self._get_image(img_path)
        data["image2"] = img
        data["target"] = self.items[index]["label"]
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
                    "text": token_ids,
                    "target": 1 if data["label"] == "True" else 0,
                    "identifier": data["identifier"], 
                })
        write_data_into_jsonl(items, index_file)

    def make_dataset_index(self):
        if self.split == "train":
            prefix = "images/train"
            json_file = os.path.join(self.path_to_data, 'nlvr', "nlvr2/data/train.json")
        elif self.split == "val":
            prefix = "dev"
            json_file = os.path.join(self.path_to_data, 'nlvr', "nlvr2/data/dev.json")
        elif self.split == "test":
            prefix = "test1"
            json_file = os.path.join(self.path_to_data, 'nlvr', "nlvr2/data/test1.json")
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
            precompute_mask_config,
            ):
        super().__init__(data_path=data_path, 
                         split=split, 
                         num_max_bpe_tokens=num_max_bpe_tokens, 
                         sample_rate=sample_rate, 
                         max_sample_size=max_sample_size, 
                         min_sample_size=min_sample_size, 
                         normalize=normalize, 
                         pad=pad,
                         precompute_mask_config=precompute_mask_config)

        cv_dir_name_pattern = os.path.join(self.data_path, 'cv-corpus-*')
        dir_name = [d for d in glob.glob(cv_dir_name_pattern) if os.path.isdir(d)][0]
        self.path_to_data = os.path.join(dir_name, 'en')
        self.path_to_clips = os.path.join(self.path_to_data, "clips")

        if self.index_exists(dataset_path=self.path_to_data):
            return
        # else...
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
        n_skipped = 0
        for i, row in validated_data.iterrows():
            path = os.path.join(self.path_to_clips, row['path'].replace('.mp3', '.flac'))
            token_ids = bpe_encoder.encode(row['sentence'])

            audio, _ = sf.read(path, dtype="float32")
            audio = torch.from_numpy(audio).float()
            if len(audio) < self.min_sample_size:
                n_skipped+=1
                continue

            items.append({
                "audio_path": path,
                "text": token_ids,
                "id": i,
                "client_id": row['client_id'],
                "sentence_id": row['sentence_id'],
            })
        self.log(f"Skipped {n_skipped} audio samples with length < {self.min_sample_size}")
        write_data_into_jsonl(items, os.path.join(self.path_to_data, f"common_voice.{self.split}.jsonl"))


MULTIMODAL_DATASET_REGISTRY = {
    "coco_captions": COCOCaptions,
    "flickr30": Flickr30Dataset,
    "flickr8k_audio": Flickr8KAudioDataset,
    "visual_genome": VisualGenome,
    "vqa": VQAv2,
    "nlvr2": NLVR2,
    "common_voice": CommonVoice
}