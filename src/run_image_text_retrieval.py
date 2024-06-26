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
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


# following stems from the BEiT3 repo: https://github.com/microsoft/unilm/blob/master/beit3/engine_for_finetuning.py
def compute_scores(img_embeds, text_embeds, img_ids):
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

    logger.info(f'* Eval result = {json.dumps(eval_result)}')


@torch.no_grad()
def zero_shot_retrieval(model:AMMData2Vec, dataloader, device, name):
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

    compute_scores(img_embeds=img_embeds, text_embeds=text_embeds, img_ids=img_ids)


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
            d2v_zero_shot_retrieval(dm.test_dataloader(), device, name)
    else:
        logger.info("Evaluating KD model")
        path = os.path.join(cfg.pretrained_path, cfg.model_version, 'last.ckpt')
        model = AMMData2VecPreTrainingLightningModule.load_from_checkpoint(path).model
        model = model.to(device)
        model.eval()

        for name, dm in datamodules:
            dm.prepare_data()
            dm.setup('test')
            logger.info(f"Zero-shot retrieval on: {name}")
            zero_shot_retrieval(model, dm.test_dataloader(), device, name)

if __name__ == "__main__":
    main()