from torchvision.transforms import v2 as transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
import logging
import json
import os
from typing import List, Tuple
import PIL
from PIL import Image
from torchvision.datasets.utils import download_url
from pydub import AudioSegment
import multiprocessing
from rich.progress import track
from timm.data.transforms import RandomResizedCropAndInterpolation

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

def download_and_unzip(urls:List[str], store_at:str="../data", archive_type:str="zip"):
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

def get_transforms(
        pretraining,
        train,
        size:int=224,
        color_jitter=None,
        aa="rand-m9-mstd0.5-inc1",
        reprob=0.25,
        remode="pixel",
        recount=1,
        beit_transforms:bool=False,
        crop_scale:Tuple[float, float]=(0.08, 1.0)):
    
    if pretraining:
        return get_transforms_pretraining(
            train=train,
            size=size,
            beit_transforms=beit_transforms,
            color_jitter=color_jitter,
            crop_scale=crop_scale
        )
    else:
        return get_transforms_finetuning(
            train=train,
            size=size,
            color_jitter=color_jitter,
            aa=aa,
            reprob=reprob,
            remode=remode,
            recount=recount)


def get_transforms_pretraining(
    train:bool=True,
    size:int=224,
    beit_transforms:bool=False,
    color_jitter=None,
    crop_scale:Tuple[float, float]=(0.08, 1.0)):

    transform_prepare = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.uint8, scale=True),
            ]
    )

    if not train:
        transform_train = transforms.Resize((size, size), interpolation=PIL.Image.BICUBIC)
    elif beit_transforms:
        beit_transform_list = []
        # beit_transform_list.append(transforms.ColorJitter(0.4, 0.4, 0.4))
        beit_transform_list.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                RandomResizedCropAndInterpolation(
                    size=(size, size),
                    scale=crop_scale,
                    interpolation="bicubic",
                ),
            ]
        )
        transform_train = transforms.Compose(beit_transform_list)
    else:
        transform_train_list = [
            transforms.RandomResizedCrop(
                size=(size, size), scale=crop_scale, interpolation=3
            ),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
        ]
        if color_jitter is not None:
            transform_train_list.append(
                transforms.ColorJitter(color_jitter, color_jitter, color_jitter)
            )
        transform_train = transforms.Compose(transform_train_list)
    
    final_transform = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
            ),
        ]
    )

    return transforms.Compose([transform_prepare, transform_train, final_transform])

def get_transforms_finetuning(
        train,
        size:int=224,
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
            input_size=size,
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
    t = [
        transforms.ToImage(),
        transforms.ToDtype(torch.uint8, scale=True),
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),
        transforms.CenterCrop(size),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean, std)
    ]
    return transforms.Compose(t)

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=(5, 9)),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=(5, 9)),
            normalize,
        ])

    def __call__(self, image):
        return self.global_transfo1(image), self.global_transfo2(image)
