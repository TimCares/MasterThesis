import hydra
import os
import json
import torch
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data import DataLoader
from beit2.datasets import DataAugmentationForBEiT
from collections import namedtuple
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from datasets_ import UngroupedImageFolder
from utils import freeze_module, load_beit2_teacher
import cudf
from cuml.cluster import KMeans
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=os.path.join("..", "configs", "cluster"), config_name='k-means')
def main(cfg: DictConfig) -> None:
    if 'seed' in cfg and cfg.seed is not None:
        seed_everything(seed=cfg.seed, workers=True)
    else:
        logger.info('No seed set.')

    dataset, idx2file_mapper = generate_beit2_embeddings(cfg)

    logger.info("Converting dataset to cuDF for KMeans clustering.")
    dataset = cudf.DataFrame(dataset.cpu().numpy())

    kmeans = KMeans(**cfg.kmeans)
    logger.info("Fitting KMeans.")
    kmeans.fit(dataset)
    del dataset
    cluster_assignments = kmeans.labels_.to_pandas().values
    logger.info("Clustering done.")
    img2cluster = {idx2file_mapper[idx]: cluster for idx, cluster in cluster_assignments.items()}
    
    with open(os.path.join(cfg.output_dir, f"img2cluster_{cfg.kmeans.n_clusters}.json"), "w") as f:
        json.dump(img2cluster, f)
    logger.info("Finished embedding clustering.")

def generate_beit2_embeddings(cfg):
    BeitTransformsArgs = namedtuple('BeitTransformsArgs', 
                                    ['imagenet_default_mean_and_std', 'input_size',
                                     'second_input_size', 'min_crop_scale', 'train_interpolation',
                                     'second_interpolation',],)
    
    transforms_args = BeitTransformsArgs(imagenet_default_mean_and_std=True, input_size=224, second_input_size=None,
                                         min_crop_scale=0.9, train_interpolation='bicubic', second_interpolation='bicubic')
    
    beit2_transforms = DataAugmentationForBEiT(transforms_args)

    datasets = []
    for path in cfg.data.datasets.keys():
        ds = UngroupedImageFolder(img_dir=path, transform=beit2_transforms)
        datasets.append(ds)
    dataset = ConcatDataset(datasets)

    dataloader = DataLoader(
        dataset,
        collate_fn=datasets[0].collater,
        batch_size=cfg.data.dataloader.batch_size,
        num_workers=cfg.data.dataloader.num_workers,
        sampler=None,
        shuffle=False,
        drop_last=False,)

    beit2_kwargs = OmegaConf.to_container(cfg.beit2, resolve=True)
    sd_path = beit2_kwargs.pop("pretrained_path")

    beit2 = load_beit2_teacher(
        sd_path=sd_path,
        **beit2_kwargs,
    ).to('cuda')
    freeze_module(beit2)

    bsz = cfg.data.dataloader.batch_size
    n_samples = len(dataloader)*bsz

    bool_masked_pos = torch.zeros((images.shape[0], beit2.patch_embed.num_patches), 
                                  dtype=torch.bool).to('cuda')
    dataset = torch.empty((n_samples, beit2.embed_dim), dtype=torch.float32).to('cuda')
    idx2file_mapper = {}
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), desc="Iterating over batches"):
            images = batch['image'].to('cuda')
            
            target = beit2.forward_features(
                x=images,
                bool_masked_pos=bool_masked_pos,
            )[:, 0]
            start_idx = idx*bsz
            dataset[start_idx:start_idx+bsz] = target
            idx2file_mapper.update({k:v for k, v in zip(range(start_idx, start_idx+bsz), batch['file_id'].tolist())})
    return dataset, idx2file_mapper

if __name__ == "__main__":
    main()
