from torchvision.transforms import v2 as transforms
import torchvision.transforms.functional as F
import torch
import logging
import random
import math
import json
import os
from PIL import Image
from torchvision.datasets.utils import download_url

logger = logging.getLogger(__name__)

def _write_data_into_jsonl(items, jsonl_file):
    with open(jsonl_file, mode="w", encoding="utf-8") as writer:
        for data in items:
            writer.write(json.dumps(data, indent=None))
            writer.write('\n')
    print("Write %s with %d items !" % (jsonl_file, len(items)))

def load_tokenizer_data(store_at:str="../data"):
    for filename in ["dict.txt", "encoder.json", "vocab.bpe"]:
        if not os.path.isfile(os.path.join(store_at, filename)):
            url = f"https://dl.fbaipublicfiles.com/fairseq/data2vec2/{filename}"
            download_url(url=url, store_at=store_at)

def download_and_unzip(urls:str, store_at:str="../data"):
    for url in urls:
        filepath = os.path.join(store_at, url.split("/")[-1])
        if not os.path.isfile(filepath):
            download_url(url=url, root=store_at)
            os.system(f"unzip {filepath} -d {store_at}")
        os.remove(filepath)


class RandomResizedCropAndInterpolationWithTwoPic:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(
        self,
        size,
        second_size=None,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation="bilinear",
        second_interpolation="lanczos",
    ):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if second_size is not None:
            if isinstance(second_size, tuple):
                self.second_size = second_size
            else:
                self.second_size = (second_size, second_size)
        else:
            self.second_size = None
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            logger.warning("range should be of kind (min, max)")

        if interpolation == "random":

            self.interpolation = (Image.BILINEAR, Image.BICUBIC)
        else:
            self.interpolation = self._pil_interp(interpolation)

        self.second_interpolation = (
            self._pil_interp(second_interpolation)
            if second_interpolation is not None
            else None
        )
        self.scale = scale
        self.ratio = ratio

    def _pil_interp(self, method):

        if method == "bicubic":
            return Image.BICUBIC
        elif method == "lanczos":
            return Image.LANCZOS
        elif method == "hamming":
            return Image.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return Image.BILINEAR

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        if self.second_size is None:
            return F.resized_crop(img, i, j, h, w, self.size, interpolation)
        else:
            return F.resized_crop(
                img, i, j, h, w, self.size, interpolation
            ), F.resized_crop(
                img, i, j, h, w, self.second_size, self.second_interpolation
            )

def get_transforms(no_transform=False, beit_transforms=False,
                   transform_jitter=False, crop_scale=(0.08, 1.0)):
    
    transform_prepare = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.uint8, scale=True),
            ]
    )

    if transform_jitter:
        transform_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    if no_transform:
        transform_train = transforms.Resize((224, 224))
    elif beit_transforms:
        beit_transform_list = []
        beit_transform_list.append(transforms.ColorJitter(0.4, 0.4, 0.4))
        beit_transform_list.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=(224, 224),
                    second_size=None,
                    interpolation="bicubic",
                    second_interpolation=None,
                    scale=crop_scale
                ),
            ]
        )
        transform_train = transforms.Compose(beit_transform_list)
    else:
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=(224, 224), scale=crop_scale, interpolation=3
                ),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
            ]
        )
    final_transform = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    if transform_jitter:
        return transforms.Compose(
            [
                transform_prepare,
                transform_train,
                transform_jitter,
                final_transform,
            ]
        )
    else:
        return transforms.Compose([transform_prepare, transform_train, final_transform])
