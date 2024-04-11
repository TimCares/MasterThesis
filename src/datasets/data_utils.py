from torchvision.transforms import v2 as transforms
import torchvision.transforms.functional as F
import torch
import logging
import random
import math
import json
import os
from typing import List, Tuple
from PIL import Image
from torchvision.datasets.utils import download_url
from pydub import AudioSegment
import multiprocessing
from rich.progress import track

logger = logging.getLogger(__name__)


def compute_block_mask_2d(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    mask_prob_adjust: float = 0,
    inverse_mask: bool = False,
    require_same_masks: bool = True,
    expand_adjcent: bool = False,
    mask_dropout: float = 0,
    non_overlapping: bool = False,
) -> torch.Tensor:

    assert mask_length > 1

    B, L = shape

    d = int(L**0.5)

    if inverse_mask:
        mask_prob = 1 - mask_prob

    if non_overlapping:
        sz = math.ceil(d / mask_length)
        inp_len = sz * sz

        inp = torch.zeros((B, 1, sz, sz))
        w = torch.ones((1, 1, mask_length, mask_length))

        mask_inds = torch.multinomial(
            1 - inp.view(B, -1),
            int(inp_len * (mask_prob + mask_prob_adjust) * (1 + mask_dropout)),
            replacement=False,
        )
        inp.view(B, -1).scatter_(1, mask_inds, 1)

        mask = torch.nn.functional.conv_transpose2d(inp, w, stride=mask_length).squeeze(
            1
        )
        if mask.size(-1) > d:
            mask = mask[..., :d, :d]
    else:
        mask = torch.zeros((B, d, d))
        mask_inds = torch.randint(
            0,
            L,
            size=(
                B,
                int(
                    L
                    * ((mask_prob + mask_prob_adjust) / mask_length**2)
                    * (1 + mask_dropout)
                ),
            ),
        )
        mask.view(B, -1).scatter_(1, mask_inds, 1)
        centers = mask.nonzero(as_tuple=True)

        inds = ([], [], [])

        offset = mask_length // 2
        for i in range(mask_length):
            for j in range(mask_length):
                k1 = i - offset
                k2 = j - offset
                inds[0].append(centers[0])
                inds[1].append(centers[1] + k1)
                inds[2].append(centers[2] + k2)

        i0 = torch.cat(inds[0])
        i1 = torch.cat(inds[1]).clamp_(min=0, max=d - 1)
        i2 = torch.cat(inds[2]).clamp_(min=0, max=d - 1)

        mask[(i0, i1, i2)] = 1

    def get_nbs(b, m, w):
        all_nbs = torch.nn.functional.conv2d(m.unsqueeze(1), w, padding="same")
        all_nbs = all_nbs.clamp_max_(1).view(b, -1)
        return all_nbs

    if require_same_masks and expand_adjcent:
        w = torch.zeros((1, 1, 3, 3))
        w[..., 0, 1] = 1
        w[..., 2, 1] = 1
        w[..., 1, 0] = 1
        w[..., 1, 2] = 1

        all_nbs = get_nbs(B, mask, w)

    mask = mask.reshape(B, -1)

    if require_same_masks:
        n_masks = mask.sum(dim=-1)
        final_target_len = int(L * (mask_prob))
        target_len = int(final_target_len * (1 + mask_dropout))

        for i in range(len(mask)):
            n = n_masks[i]
            m = mask[i]
            r = 0
            while expand_adjcent and n < target_len:
                if r == 0:
                    nbs = all_nbs[i]
                else:
                    nbs = get_nbs(1, m.view(1, d, d), w).flatten()

                cands = (1 - m + nbs) > 1
                cand_sz = int(cands.sum().item())

                assert cand_sz > 0, f"{nbs} {cand_sz}"

                to_mask = torch.multinomial(
                    cands.float(), min(cand_sz, int(target_len - n)), replacement=False
                )
                m[to_mask] = 1
                assert to_mask.numel() > 0
                n += to_mask.numel()
                r += 1

            if n > final_target_len:
                to_unmask = torch.multinomial(
                    m, int(n - final_target_len), replacement=False
                )
                m[to_unmask] = 0
            elif n < final_target_len:
                to_mask = torch.multinomial(
                    (1 - m), int(final_target_len - n), replacement=False
                )
                m[to_mask] = 1

    if inverse_mask:
        mask = 1 - mask

    return mask


def compute_block_mask_1d(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    mask_prob_adjust: float = 0,
    inverse_mask: bool = False,
    require_same_masks: bool = True,
    expand_adjcent: bool = False,
    mask_dropout: float = 0,
    non_overlapping: bool = False,
) -> torch.Tensor:

    B, L = shape

    if inverse_mask:
        mask_prob = 1 - mask_prob

    if non_overlapping:
        sz = math.ceil(L / mask_length)

        inp = torch.zeros((B, 1, sz))
        w = torch.ones((1, 1, mask_length))

        mask_inds = torch.multinomial(
            1 - inp.view(B, -1),
            int(sz * (mask_prob + mask_prob_adjust) * (1 + mask_dropout)),
            replacement=False,
        )
        inp.view(B, -1).scatter_(1, mask_inds, 1)

        mask = torch.nn.functional.conv_transpose1d(inp, w, stride=mask_length).squeeze(
            1
        )
        if mask.size(-1) > L:
            mask = mask[..., :L]

    else:
        mask = torch.zeros((B, L))
        mask_inds = torch.randint(
            0,
            L,
            size=(
                B,
                int(
                    L
                    * ((mask_prob + mask_prob_adjust) / mask_length)
                    * (1 + mask_dropout)
                ),
            ),
        )

        mask.view(B, -1).scatter_(1, mask_inds, 1)
        centers = mask.nonzero(as_tuple=True)

        inds = ([], [])

        offset = mask_length // 2
        for i in range(mask_length):
            k1 = i - offset
            inds[0].append(centers[0])
            inds[1].append(centers[1] + k1)

        i0 = torch.cat(inds[0])
        i1 = torch.cat(inds[1]).clamp_(min=0, max=L - 1)

        mask[(i0, i1)] = 1

    def get_nbs(b, m, w):
        all_nbs = torch.nn.functional.conv1d(m.unsqueeze(1), w, padding="same")
        all_nbs = all_nbs.clamp_max_(1).view(b, -1)
        return all_nbs

    if require_same_masks and expand_adjcent:
        w = torch.ones((1, 1, 3))
        w[..., 1] = 0
        all_nbs = get_nbs(B, mask, w)

    mask = mask.view(B, -1)

    if require_same_masks:
        n_masks = mask.sum(dim=-1)
        final_target_len = int(L * (mask_prob))
        target_len = int(final_target_len * (1 + mask_dropout))

        for i in range(len(mask)):
            n = n_masks[i]
            m = mask[i]
            r = 0
            while expand_adjcent and n < target_len:
                if r == 0:
                    nbs = all_nbs[i]
                else:
                    nbs = get_nbs(1, m.unsqueeze(0), w).squeeze(0)

                cands = (1 - m + nbs) > 1
                cand_sz = int(cands.sum().item())

                assert cand_sz > 0, f"{nbs} {cand_sz}"

                to_mask = torch.multinomial(
                    cands.float(), min(cand_sz, int(target_len - n)), replacement=False
                )
                m[to_mask] = 1
                assert to_mask.numel() > 0
                n += to_mask.numel()
                r += 1

            if n > final_target_len:
                to_unmask = torch.multinomial(
                    m, int(n - final_target_len), replacement=False
                )
                m[to_unmask] = 0
            elif n < final_target_len:
                to_mask = torch.multinomial(
                    (1 - m), int(final_target_len - n), replacement=False
                )
                m[to_mask] = 1

    if inverse_mask:
        mask = 1 - mask

    return mask


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
