import os
import json
import torch
from multimodal import BaseImageText

class VQAv2(BaseImageText):
    def __init__(self, 
                 data_path,
                 split,
                 num_max_bpe_tokens,
                 input_size,
                 transform_jitter=False,
                 beit_transforms=False,
                 no_transform=False,
                 task=None,
                 crop_scale=(0.6, 1.0)):
        
        super().__init__(data_path,
                         split,
                         num_max_bpe_tokens,
                         input_size,
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