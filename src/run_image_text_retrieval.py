import logging
from rich.progress import track
from typing import *
import torch
import os
from pytorch_lightning import LightningModule
import sys
sys.path.append('beit2')
from models import MODEL_REGISTRY
from datamodules import DATAMODULE_REGISTRY
from utils import load_pretrained_d2v_model
from omegaconf import DictConfig, open_dict
import hydra
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def compute_average_median_rank(images:torch.Tensor, texts:torch.Tensor, iids:torch.Tensor, tiids:torch.Tensor):
    mask = iids.unsqueeze(1) == tiids.unsqueeze(0)
    masks = mask.chunk(5)
    image_chunks = images.chunk(5)
    
    amrs_ir = []
    amrs_tr = []
    for i in range(5):
        avg_median_rank_tr, avg_median_rank_ir = amr_for_chunk(image_chunks[i], texts, masks[i])
        amrs_tr.append(avg_median_rank_tr)
        amrs_ir.append(avg_median_rank_ir)
    return sum(amrs_tr) / len(amrs_tr), sum(amrs_ir) / len(amrs_ir)

def amr_for_chunk(images, texts, mask):
    selected_indices = []
    matching_candidates_indices = torch.stack([torch.nonzero(row).squeeze()[:5] for row in mask])
    # [:5] -> some images have >5 captions

    median_ranks_ir = []
    median_ranks_tr = []
    for i in range(matching_candidates_indices.shape[1]):
        selected_indices = matching_candidates_indices[:, i]
        selected_texts = texts[selected_indices]

        scores = images @ selected_texts.T
        median_rank_tr = scores.argsort(dim=1, descending=True).argsort(dim=1).diagonal().add(1).float().quantile(0.5).item()
        median_ranks_tr.append(median_rank_tr)
        median_rank_ir = scores.argsort(dim=0, descending=True).argsort(dim=0).diagonal().add(1).float().quantile(0.5).item()
        median_ranks_ir.append(median_rank_ir)

    avg_median_rank_tr = sum(median_ranks_tr) / len(median_ranks_tr)
    avg_median_rank_ir = sum(median_ranks_ir) / len(median_ranks_ir)
    return avg_median_rank_tr, avg_median_rank_ir

# following stems mostly from the BEiT3 repo: https://github.com/microsoft/unilm/blob/master/beit3/engine_for_finetuning.py
def compute_scores(img_embeds, text_embeds, img_ids, compute_amr=False):
    image_feats = {} # collect all unique image features, and create mapping based on id
    for feats, ids in zip(img_embeds, img_ids):
        for i, _idx in enumerate(ids):
            idx = _idx.item()
            if idx not in image_feats:
                image_feats[idx] = feats[i]

    tiids = torch.cat(img_ids, dim=0)
    iids = []
    sorted_tensors = []
    for key in sorted(image_feats.keys()):
        sorted_tensors.append(image_feats[key].view(1, -1))
        iids.append(key)

    img_embeds = torch.cat(sorted_tensors, dim=0)
    text_embeds = torch.cat(text_embeds, dim=0)

    scores = img_embeds @ text_embeds.t()
    iids = torch.LongTensor(iids).to(scores.device)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    eval_result = {
        "tr_r10": tr_r10.item() * 100.0, 
        "tr_r5": tr_r5.item() * 100.0, 
        "tr_r1": tr_r1.item() * 100.0, 
        "ir_r10": ir_r10.item() * 100.0, 
        "ir_r5": ir_r5.item() * 100.0, 
        "ir_r1": ir_r1.item() * 100.0, 
        "average_score": 100.0 * (tr_r1 + tr_r5 + tr_r10 + ir_r1 + ir_r5 + ir_r10).item() / 6.0,
    }

    if compute_amr:
        tr_amr, ir_amr = compute_average_median_rank(img_embeds, text_embeds, iids, tiids)
        eval_result['tr_amr'] = tr_amr
        eval_result['ir_amr'] = ir_amr

    logger.info(f'* Eval result = {json.dumps(eval_result)}')
    return eval_result


@torch.no_grad()
def zero_shot_retrieval(model, dataloader, device, compute_amr=False):
    img_embeds = []
    text_embeds = []
    img_ids = []

    for batch in track(dataloader):
        image = batch['image'].to(device)
        text = batch['text'].to(device)
        padding_mask = batch['padding_mask'].to(device) if 'padding_mask' in batch else None
        # encoding also normalizes the output
        img_emb = model.encode_image(image=image)['x']
        text_emb = model.encode_text(text=text, padding_mask=padding_mask)['x']
        img_embeds.append(img_emb)
        text_embeds.append(text_emb)
        img_ids.append(batch['id'].to(device))

    return compute_scores(img_embeds=img_embeds, text_embeds=text_embeds, img_ids=img_ids, compute_amr=compute_amr)


@torch.no_grad()
def d2v_zero_shot_retrieval(dataloader, device):
    d2v_image = load_pretrained_d2v_model('/workspace/models/base_imagenet.pt')
    d2v_image = d2v_image.to(device)
    d2v_image.eval()
    d2v_text = load_pretrained_d2v_model('/workspace/models/nlp_base.pt')
    d2v_text = d2v_text.to(device)
    d2v_text.eval()

    img_embeds = []
    text_embeds = []
    img_ids = []

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

        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        
        img_embeds.append(img_emb)
        text_embeds.append(text_emb)
        img_ids.append(batch['id'].to(device))

    compute_scores(img_embeds=img_embeds, text_embeds=text_embeds, img_ids=img_ids)


@hydra.main(version_base=None, config_path=os.path.join("..", "configs", "retrieval"), config_name='coco_flickr')
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
            dm.prepare_data()
            dm.setup('test')
            d2v_zero_shot_retrieval(dm.test_dataloader(), device)
    else:
        logger.info("Evaluating KD model")
        path = os.path.join(cfg.pretrained_path, cfg.model_version)
        model_cls:LightningModule = MODEL_REGISTRY[cfg.model_name]['module']
        model = model_cls.load_from_checkpoint(path).model
        model = model.to(device)
        model.requires_grad_(False)
        model.eval()

        for name, dm in datamodules:
            dm.prepare_data()
            dm.setup('test')
            logger.info(f"Zero-shot retrieval on: {name}")
            zero_shot_retrieval(model, dm.test_dataloader(), device, compute_amr=name=='coco_captions')

if __name__ == "__main__":
    main()