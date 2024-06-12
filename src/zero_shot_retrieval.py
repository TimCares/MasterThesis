import logging
from rich.progress import track
from typing import *
import torch
import os
from models.mm_data2vec import AMMData2Vec, AMMData2VecPreTrainingLightningModule
from datamodules import DATAMODULE_REGISTRY
from utils import load_pretrained_d2v_model
from omegaconf import DictConfig, open_dict
import hydra

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def compute_recall(similarity_scores: torch.Tensor, k: int = 5):
    dataset_size = similarity_scores.size(0)
    targets = torch.arange(dataset_size).view(dataset_size, -1)
    _, topk_idx = torch.topk(similarity_scores, k)
    recall = targets.eq(topk_idx).sum()
    recall = recall / dataset_size
    return recall


@torch.no_grad()
def zero_shot_retrieval(model:AMMData2Vec, dataloader, device, name):
    img_embeds = []
    text_embeds = []

    for batch in track(dataloader):
        image = batch['image'].to(device)
        text = batch['text'].to(device)
        padding_mask = batch['padding_mask'].to(device) if 'padding_mask' in batch else None
        # encoding also normalizes the output
        img_emb = model.encode_image(image=image)
        text_emb = model.encode_text(text=text, padding_mask=padding_mask)
        img_embeds.append(img_emb.detach().cpu())
        text_embeds.append(text_emb.detach().cpu())

    img_embeds = torch.cat(img_embeds, 0)
    text_embeds = torch.cat(text_embeds, 0)

    similarity_scores = img_embeds @ text_embeds.t()
    similarity_scores_t = similarity_scores.t()

    img_to_txt_r1 = compute_recall(similarity_scores, k=1)
    img_to_txt_r5 = compute_recall(similarity_scores, k=5)
    txt_to_img_r1 = compute_recall(similarity_scores_t, k=1)
    txt_to_img_r5 = compute_recall(similarity_scores_t, k=5)

    logger.info(f"{name}: Image to Text Recall@1 {img_to_txt_r1}")
    logger.info(f"{name}: Image to Text Recall@5 {img_to_txt_r5}")
    logger.info(f"{name}: Text to Image Recall@1 {txt_to_img_r1}")
    logger.info(f"{name}: Text to Image Recall@5 {txt_to_img_r5}")


@torch.no_grad()
def d2v_zero_shot_retrieval(dataloader, device, name):
    d2v_image = load_pretrained_d2v_model('/workspace/models/base_imagenet.pt')
    d2v_image = d2v_image.to(device)
    d2v_image.eval()
    d2v_text = load_pretrained_d2v_model('/workspace/models/nlp_base.pt')
    d2v_text = d2v_text.to(device)
    d2v_text.eval()

    img_embeds = []
    text_embeds = []

    for batch in track(dataloader):
        image = batch['image'].to(device)
        text = batch['text'].to(device)
        padding_mask = batch['padding_mask'].to(device) if 'padding_mask' in batch else None
        
        img_emb = d2v_image.extract_features(
            source=image,
            mode=None, # determined automatically in model
            padding_mask=None,
            mask=False,
            remove_extra_tokens=False,
        )['x'][:, 0]

        text_emb = d2v_text.extract_features(
            source=text,
            mode=None, # determined automatically in model
            padding_mask=padding_mask,
            mask=False,
            remove_extra_tokens=False,
        )['x'][:, 0]
        
        img_embeds.append(img_emb.detach().cpu())
        text_embeds.append(text_emb.detach().cpu())

    img_embeds = torch.cat(img_embeds, 0)
    text_embeds = torch.cat(text_embeds, 0)

    similarity_scores = img_embeds @ text_embeds.t()
    similarity_scores_t = similarity_scores.t()

    img_to_txt_r1 = compute_recall(similarity_scores, k=1)
    img_to_txt_r5 = compute_recall(similarity_scores, k=5)
    txt_to_img_r1 = compute_recall(similarity_scores_t, k=1)
    txt_to_img_r5 = compute_recall(similarity_scores_t, k=5)

    logger.info(f"{name}: Image to Text Recall@1 {img_to_txt_r1}")
    logger.info(f"{name}: Image to Text Recall@5 {img_to_txt_r5}")
    logger.info(f"{name}: Text to Image Recall@1 {txt_to_img_r1}")
    logger.info(f"{name}: Text to Image Recall@5 {txt_to_img_r5}")


@hydra.main(version_base=None, config_path=os.path.join("..", "configs", "zero_shot"))
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    datamodules = []

    shared_args = cfg.data.shared_args

    for datamodule_key in cfg.data.datamodules.keys(): # coco and flickr30
        dataset_args = cfg.data.datamodules[datamodule_key]
        with open_dict(dataset_args):
            dataset_args.update(shared_args)
        dm = DATAMODULE_REGISTRY[datamodule_key](**dataset_args)
        datamodules.append((datamodule_key, dm))
    
    if cfg.eval_d2v:
        logger.info("Evaluating Data2Vec model")
        for name, dm in datamodules:
            d2v_zero_shot_retrieval(dm.test_dataloader(), device, name)
    else:
        logger.info("Evaluating KD model")
        path = os.path.join(cfg.pretrained_path, cfg.model_version, 'last.ckpt')
        model = AMMData2VecPreTrainingLightningModule.load_from_checkpoint(path).model
        model = model.to(device)
        model.eval()

        for name, dm in datamodules:
            logger.info(f"Zero-shot retrieval on: {name}")
            zero_shot_retrieval(model, dm, device, name)

if __name__ == "__main__":
    main()