from torchvision.transforms import v2 as transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
import logging
import json
import os
from typing import List
import PIL
from torchvision.datasets.utils import download_url
from pydub import AudioSegment
import multiprocessing
from rich.progress import track

logger = logging.getLogger(__name__)


def _mp3_to_flac(mp3_file:str):
        audio = AudioSegment.from_mp3(mp3_file)
        audio = audio.set_frame_rate(16000)
        flac_file = mp3_file.replace(".mp3", ".flac")
        audio.export(flac_file, format="flac")
        os.remove(mp3_file)
    
def convert_mp3_to_flac(dir:str, files:List[str]):
    files = [os.path.join(dir, f) for f in files]
    len_before = len(files)
    files = [f for f in files if os.path.exists(f) and f.endswith(".mp3")]
    if len(files) != len_before:
        logger.info(f'Exclude {len_before - len(files)} files that are not mp3 files or do not exist.')
    
    with multiprocessing.Pool() as pool:
        list(track(pool.imap_unordered(_mp3_to_flac, files), total=len(files), description="Converting mp3 to flac"))
    logger.info(f"Converted {len(files)} mp3 files to flac files")

def write_data_into_jsonl(items, jsonl_file):
    with open(jsonl_file, mode="w", encoding="utf-8") as writer:
        for data in items:
            writer.write(json.dumps(data, indent=None))
            writer.write('\n')
    logger.info("Write %s with %d items !" % (jsonl_file, len(items)))

def download_and_unzip(urls:str, store_at:str="../data", archive_type:str="zip"):
    for url in urls:
        filepath = os.path.join(store_at, os.path.basename(url))
        if not os.path.exists(filepath):
            download_url(url=url, root=store_at)
            if archive_type == "zip":
                os.system(f"unzip {filepath} -d {store_at}")
            elif archive_type == "tar" or archive_type == "tar.gz" or archive_type == "xz":
                os.system(f"tar -xf {filepath} -C {store_at}")
            else:
                raise ValueError(f"Unsupported archive type: {archive_type}")
        os.remove(filepath)

def get_transforms(train,
                   color_jitter=None,
                   aa="rand-m9-mstd0.5-inc1",
                   reprob=0.25,
                   remode="pixel",
                   recount=1):

    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=244,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=aa,
            interpolation="bicubic",
            re_prob=reprob,
            re_mode=remode,
            re_count=recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    t.append(
        transforms.ToImage(),
        transforms.ToDtype(torch.uint8, scale=True),
        transforms.Resize(
            244, interpolation=PIL.Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(244))

    t.append(transforms.ToDtype(torch.float32, scale=True),)
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
