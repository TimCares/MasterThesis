import hydra
import os
import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning import seed_everything
from models import ImageCluster
from datamodules import DATAMODULE_REGISTRY, MultiDataModule
from typing import List
from pytorch_lightning import LightningDataModule
from beit2.norm_ema_quantizer import kmeans, l2norm
from einops import repeat
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=os.path.join("..", "configs", "cluster"), config_name='cluster')
def main(cfg: DictConfig) -> None:
    if 'seed' in cfg and cfg.seed is not None:
        seed_everything(seed=cfg.seed, workers=True)
    else:
        logger.info('No seed set.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader_args = cfg.data.dataloader
    common_args = cfg.data.common

    datamodules:List[LightningDataModule] = []
    for datamodule_key in cfg.data.datamodules.keys():
        dataset_args = cfg.data.datamodules[datamodule_key]
        with open_dict(dataset_args):
            dataset_args.update(dataloader_args)
            dataset_args.update(common_args)
        datamodules.append(DATAMODULE_REGISTRY[datamodule_key](**dataset_args))
        logger.info(f"Train datamodule {datamodule_key}: {dataset_args}")
    
    multi_datamodule = MultiDataModule(datamodules=datamodules, **dataloader_args)

    model = ImageCluster(cfg.model)
    model.to(device)

    for i in range(cfg.n_epochs):
        for batch in tqdm(multi_datamodule.train_dataloader(), desc=f"Epoch {i+1}, Train: Iterating over batches"):
            image = batch['image_teacher'].to(device)
            batch_kmeans(model, image)
        
        torch.save(model.state_dict(), cfg.save_path)

        codebook_cnt = torch.zeros(model.cfg.num_clusters, dtype=torch.float64).to(model.device)
        for batch in tqdm(multi_datamodule.val_dataloader(), desc=f"Epoch {i+1}, Val: Iterating over batches"):
            image = batch['image_teacher'].to(device)
            cluster_idx = model.get_cluster(image)['cluster_idx']
            codebook_cnt += torch.bincount(cluster_idx, minlength=model.cfg.num_clusters)
        
        zero_cnt = (codebook_cnt == 0).sum()
        logger.info(f"Zero count: {zero_cnt}")
        logger.info(f"Standard deviation: {codebook_cnt.std()}")
        del codebook_cnt


def batch_kmeans(model, image):
    if not model.initted:
        means, bins = kmeans(image, model.cfg.num_clusters, use_cosine_sim=True)
        model.cluster_prototypes = means
        model.initted = True
        return
    else:
        means = model.cluster_prototypes

    dists = model.get_cluster(image)['cluster_dist']
    
    dim, dtype = image.shape[-1], image.dtype

    buckets = dists.max(dim = -1).indices
    bins = torch.bincount(buckets, minlength = model.cfg.num_clusters)
    zero_mask = bins == 0
    bins_min_clamped = bins.masked_fill(zero_mask, 1)

    new_means = buckets.new_zeros(model.cfg.num_clusters, dim, dtype = dtype)
    new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), image)
    new_means = new_means / bins_min_clamped[..., None]

    means = torch.where(zero_mask[..., None], means, new_means)

    model.cluster_prototypes = model.cluster_prototypes * model.cfg.decay + means * (1 - model.cfg.decay)

    model.cluster_prototypes = l2norm(model.cluster_prototypes)



if __name__ == "__main__":
    main()
