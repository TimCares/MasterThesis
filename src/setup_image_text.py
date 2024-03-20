# from: https://github.com/microsoft/unilm/blob/master/beit3/datasets.py
import os
import json
import random
import glob
from collections import defaultdict, Counter
from tqdm import tqdm

import sys
sys.path.append('../../')
from config import COCO_CAPTIONS_PATH, VG_PATH
from src.datasets.bpe_encoder import BPEEncoder
from src.datasets.data_utils import _write_data_into_jsonl
from utils.glossary import normalize_word

# --------------------- VQA ---------------------

def get_score(occurences):
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

def make_vqa_dataset_index(data_path, bpe_encoder, annotation_data_path):
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_train2014_questions.json"), "r") as fp:
            questions_train2014 = json.load(fp)["questions"]
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_val2014_questions.json"), "r") as fp:
            questions_val2014 = json.load(fp)["questions"]
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_test2015_questions.json"), "r") as fp:
            questions_test2015 = json.load(fp)["questions"]
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_test-dev2015_questions.json"), "r") as fp:
            questions_test_dev2015 = json.load(fp)["questions"]

        with open(os.path.join(annotation_data_path, "v2_mscoco_train2014_annotations.json"), "r") as fp:
            annotations_train2014 = json.load(fp)["annotations"]
        with open(os.path.join(annotation_data_path, "v2_mscoco_val2014_annotations.json"), "r") as fp:
            annotations_val2014 = json.load(fp)["annotations"]

        annotations = dict()

        for split, questions in zip(
            ["train", "val", "test", "test-dev"],
            [questions_train2014, questions_val2014, questions_test2015, questions_test_dev2015],
        ):
            _annot = defaultdict(dict)
            for q in questions:
                question_text = q["question"]
                token_ids = bpe_encoder.encode(question_text)

                assert q["question_id"] not in _annot[q["image_id"]]
                _annot[q["image_id"]][q["question_id"]] = {
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
                    score = get_score(answer_count[answer])
                    scores.append(score)

                assert "labels" not in _annot[q["image_id"]][q["question_id"]]
                assert "question" in _annot[q["image_id"]][q["question_id"]]
                _annot[q["image_id"]][q["question_id"]]["labels"] = labels
                _annot[q["image_id"]][q["question_id"]]["scores"] = scores

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
            paths = list(glob.glob(f"{data_path}/{split_name}/*.jpg"))
            random.shuffle(paths)
            annot_paths = [path for path in paths \
                if int(path.split("/")[-1].split("_")[-1][:-4]) in annot]

            if len(paths) == len(annot_paths):
                print("all images have caption annotations")
            else:
                print("not all images have caption annotations")
            print(len(paths), len(annot_paths), len(annot))

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

            _write_data_into_jsonl(items=items, jsonl_file=os.path.join(data_path, "vqa.%s.jsonl" % split))

        # Following ViLT, we use 1000 images of the original val set as the final val set        
        val_image2items = defaultdict(list)
        for item in split2items["val"]:
            val_image2items[item["image_path"]].append(item)
        
        print("Contains %d image and %d pairs for val set!" % (len(val_image2items), len(split2items["val"])))

        val_images = list(val_image2items.keys())
        random.shuffle(val_images)
        trainable_val = []
        rest_val = []
        for i, image_id in enumerate(val_images):
            if i < 1000:
                rest_val += val_image2items[image_id]
            else:
                trainable_val += val_image2items[image_id]
        
        _write_data_into_jsonl(items=trainable_val, jsonl_file=os.path.join(data_path, "vqa.trainable_val.jsonl"))
        _write_data_into_jsonl(items=rest_val, jsonl_file=os.path.join(data_path, "vqa.rest_val.jsonl"))

        with open(os.path.join(data_path, "answer2label.txt"), mode="w", encoding="utf-8") as writer:
            for ans in ans2label:
                to_json = {
                    "answer": ans, 
                    "label": ans2label[ans]
                }
                writer.write("%s\n" % json.dumps(to_json))

# --------------------- COCO ---------------------

def _make_captioning_coco_karpathy_dataset_index(
        data_path, 
        bpe_encoder, 
        split=("train", "restval"), 
        split_name="train", 
):
    coco_karpathy_split_json_file = os.path.join(data_path, "dataset_coco.json")
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
                        token_ids = bpe_encoder.encode(sent["raw"])
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
    index_file = os.path.join(data_path, "coco_captioning.%s.jsonl" % split_name)
    _write_data_into_jsonl(items, index_file)


def make_coco_captioning_dataset_index(data_path, bpe_encoder):
    _make_captioning_coco_karpathy_dataset_index(data_path, bpe_encoder, split=("train", "restval"), split_name="train")
    #_make_captioning_coco_karpathy_dataset_index(data_path, bpe_encoder, split=("val", ), split_name="val")
    #_make_captioning_coco_karpathy_dataset_index(data_path, bpe_encoder, split=("test", ), split_name="test")

def make_nocaps_captioning_dataset_index(data_path):
    _make_nocaps_dataset_index(data_path, split="val")
    _make_nocaps_dataset_index(data_path, split="test")


# --------------------- NLVR2 ---------------------

def _get_index_files(split, task=None):
    if split == "train":
        return ("nlvr2.train.index.jsonl", )
    elif split == "val":
        return ("nlvr2.dev.index.jsonl", )
    elif split == "test":
        return ("nlvr2.test-P.index.jsonl", )
    else:
        raise RuntimeError("split %s is not found!" % split)

def _preprocess_json(preifx, json_file, bpe_encoder, index_file):
    items = []
    with open(json_file, mode="r", encoding="utf-8") as reader:
        for line in reader:
            data = json.loads(line)
            path = os.path.join(preifx, str(data["directory"])) if "directory" in data else preifx
            path = os.path.join(path, "-".join(data["identifier"].split("-")[:-1]))
            token_ids = bpe_encoder.encode(data["sentence"])
            items.append({
                "image_path": path + "-img0.png",
                "image2_path": path + "-img1.png",
                "text_segment": token_ids,
                "label": 1 if data["label"] == "True" else 0,
                "identifier": data["identifier"], 
            })
    _write_data_into_jsonl(items, index_file)

def make_dataset_index(data_path, bpe_encoder, nlvr_repo_path):
    _preprocess_json(
        preifx="images/train", json_file=os.path.join(nlvr_repo_path, "nlvr2/data/train.json"), 
        bpe_encoder=bpe_encoder, index_file=os.path.join(data_path, _get_index_files("train")[0]), 
    )
    _preprocess_json(
        preifx="dev", json_file=os.path.join(nlvr_repo_path, "nlvr2/data/dev.json"), 
        bpe_encoder=bpe_encoder, index_file=os.path.join(data_path, _get_index_files("val")[0]), 
    )
    _preprocess_json(
        preifx="test1", json_file=os.path.join(nlvr_repo_path, "nlvr2/data/test1.json"), 
        bpe_encoder=bpe_encoder, index_file=os.path.join(data_path, _get_index_files("test")[0]), 
    )

# --------------------- Visual Genome ---------------------
    
def make_visual_genome_dataset_index(data_path, bpe_encoder, visual_genome_path):
    with open(os.path.join(data_path, "region_descriptions.json"), "r") as fp:
        region_descriptions = json.load(fp)

    items = []

    for image_meta in tqdm(region_descriptions, total=len(region_descriptions)):
        image_path = os.path.join(visual_genome_path, "VG_100K", f"{image_meta["id"]}.jpg")
        caption = ""
        for region in image_meta["regions"]:
            caption += region["phrase"] + " "
        
        token_ids = bpe_encoder.encode(region["phrase"])
        # truncation will also be done when reading the data, but there we also substract 2 for the special tokens
        # so we already do it here to save time and memory
        token_ids = token_ids[:512 - 2]
        items.append({
            "image_path": image_path, 
            "text_segment": token_ids,
            "image_id": image_meta["id"], 
        })

    _write_data_into_jsonl(items, os.path.join(data_path, "visual_genome.jsonl"))
    

if __name__ == "__main__":
    encoder_json_path = os.path.join(COCO_CAPTIONS_PATH, "encoder.json")
    vocab_bpe_path = os.path.join(COCO_CAPTIONS_PATH, "vocab.bpe")
    bpe_encoder = BPEEncoder(encoder_json_path, vocab_bpe_path)
    # make_vqa_dataset_index(data_path=COCO_CAPTIONS_PATH, bpe_encoder=bpe_encoder, annotation_data_path=VQA_CAPTIONS_PATH)
    make_coco_captioning_dataset_index(data_path=COCO_CAPTIONS_PATH, bpe_encoder=bpe_encoder)

    make_visual_genome_dataset_index(data_path=VG_PATH, bpe_encoder=bpe_encoder, visual_genome_path=VG_PATH)