import torch
import torch.nn.functional as F
from utils import load_pretrained_d2v_model
import os
from hydra import compose, initialize
from omegaconf import OmegaConf, open_dict
from datamodules import DATAMODULE_REGISTRY
from typing import *
from torch.utils.data import DataLoader
from rich.progress import track
import logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def _get_zero_shot_retrieval_embeddings(model, dataloader:DataLoader, device:str) -> Tuple[torch.Tensor, torch.Tensor]:
    agg_strategies = ['CLS', 'mean', 'mean_without_CLS']
    X_data_dict = {strategy: [] for strategy in agg_strategies}
    y = []
    
    for batch in track(dataloader):
        source = batch[batch['modes'][0].name.lower()].to(device)
        padding_mask = batch['padding_mask'].to(device) if 'padding_mask' in batch else None
        # encoding also normalizes the output
        pred = model.extract_features(
                source=source,
                mode=None, # determined automatically in model
                padding_mask=padding_mask,
                mask=False, # we are creating targets from a teacher model for the student model, so no mask
                remove_extra_tokens=False,
            )
        out = pred['x']
        
        for agg_strategy in agg_strategies:
            if agg_strategy=="CLS":
                out_reduced = out[:, 0, :]
            elif agg_strategy=="mean":
                if 'padding_mask' in pred and pred['padding_mask'] is not None:
                    non_padded_avg = []
                    for i in range(out.size(0)):
                        non_padded_avg.append(out[i][~pred['padding_mask'][i]].mean(dim=0)) # list of B*(tensors of shape (C,))
                    out_reduced = torch.stack(non_padded_avg) # list of B*(tensors of shape (C,)) -> BC
                else:
                    out_reduced = out.mean(dim=1)
            else:
                if 'padding_mask' in pred and pred['padding_mask'] is not None:
                    out_reduced = out[:, 1:, :]
                    padding_mask = pred['padding_mask'][:, 1:]
                    non_padded_avg = []
                    for i in range(out_reduced.size(0)):
                        non_padded_avg.append(out_reduced[i][~padding_mask[i]].mean(dim=0)) # list of B*(tensors of shape (C,))
                    out_reduced = torch.stack(non_padded_avg) # list of B*(tensors of shape (C,)) -> BC
                else:
                    out_reduced = out[:, 1:, :].mean(dim=1)

            out_reduced = F.normalize(out_reduced, dim=-1)

            X_data_dict[agg_strategy].append(out_reduced.cpu())

        y.append(batch['target'])

    for agg_strategy in agg_strategies:
        X_data_dict[agg_strategy] = torch.cat(X_data_dict[agg_strategy], dim=0)

    y = torch.cat(y, dim=0).cpu()

    return X_data_dict, y

def _get_all_match(similarity_scores:torch.Tensor,
                   m_bank_targets:torch.Tensor,
                   q_targets:torch.Tensor,
                   k:int) -> Tuple[float, float]:
    
    _, indices = similarity_scores.topk(k, dim=-1)
    return (m_bank_targets[indices]==q_targets.unsqueeze(1)).all(dim=-1).sum().div(len(q_targets))

def _get_any_match(similarity_scores:torch.Tensor,
                   m_bank_targets:torch.Tensor,
                   q_targets:torch.Tensor,
                   k:int) -> Tuple[float, float]:
    # equivalent to top-k accuracy
    _, indices = similarity_scores.topk(k, dim=-1)
    return (m_bank_targets[indices]==q_targets.unsqueeze(1)).any(dim=-1).sum().div(len(q_targets))


def unimodal_zero_shot_retrieval(model,
                                 train_loader:DataLoader,
                                 test_loader:DataLoader,
                                 device:str,
                                 name:str) -> None:
    
    memory_bank, memory_bank_targets = _get_zero_shot_retrieval_embeddings(model=model, dataloader=train_loader, device=device)
    query, query_targets = _get_zero_shot_retrieval_embeddings(model=model, dataloader=test_loader, device=device)

    for agg_strategy in ['CLS', 'mean', 'mean_without_CLS']:

        similarity_scores = query[agg_strategy] @ memory_bank[agg_strategy].t()
        similarity_scores_t = similarity_scores.t()
        train_test_top1 = _get_all_match(similarity_scores, memory_bank_targets, query_targets, k=1)
        test_train_top1 = _get_all_match(similarity_scores_t, query_targets, memory_bank_targets, k=1)

        logger.info(f"{agg_strategy} {name} train-test top1: {train_test_top1.item()}")
        logger.info(f"{agg_strategy} {name} test-train top1: {test_train_top1.item()}")

        try:
            train_test_top3 = _get_all_match(similarity_scores, memory_bank_targets, query_targets, k=3)
            test_train_top3 = _get_all_match(similarity_scores_t, query_targets, memory_bank_targets, k=3)

            logger.info(f"{agg_strategy} {name} train-test top3: {train_test_top3.item()}")
            logger.info(f"{agg_strategy} {name} test-train top3: {test_train_top3.item()}")
        except:
            pass # if data has less than 5 classes, top5-accuracy will throw an error


def compute_recall(similarity_scores: torch.Tensor, k: int = 5) -> float:
    dataset_size = similarity_scores.size(0)
    targets = torch.arange(dataset_size).view(dataset_size, -1)
    _, topk_idx = torch.topk(similarity_scores, k)
    recall = targets.eq(topk_idx).sum()
    recall = recall / dataset_size
    return recall

@torch.no_grad()
def _get_zero_shot_pair_retrieval_embeddings(model, dataloader:DataLoader, device:str) -> Tuple[torch.Tensor, torch.Tensor]:
    agg_strategies = ['CLS', 'mean', 'mean_without_CLS']
    X_data_dict = {strategy: {i: [] for i in range(2)} for strategy in agg_strategies}
    
    for batch in track(dataloader):
        for i in range(2): # for each element in the pair
            source = batch[f"{batch['modes'][0].name.lower()}{i}"].to(device) # "text0", "text1"
            padding_mask = batch[f'padding_mask{i}'].to(device) if f'padding_mask{i}' in batch else None # "padding_mask0", "padding_mask1"
            # encoding also normalizes the output
            pred = model.extract_features(
                source=source,
                mode=None, # determined automatically in model
                padding_mask=padding_mask,
                mask=False, # we are creating targets from a teacher model for the student model, so no mask
                remove_extra_tokens=False,
            )
            out = pred['x']
            
            for agg_strategy in agg_strategies:
                if agg_strategy=="CLS":
                    out_reduced = out[:, 0, :]
                elif agg_strategy=="mean":
                    if f'padding_mask{i}' in pred and pred[f'padding_mask{i}'] is not None:
                        non_padded_avg = []
                        for i in range(out.size(0)):
                            non_padded_avg.append(out[i][~pred[f'padding_mask{i}'][i]].mean(dim=0)) # list of B*(tensors of shape (C,))
                        out_reduced = torch.stack(non_padded_avg) # list of B*(tensors of shape (C,)) -> BC
                    else:
                        out_reduced = out.mean(dim=1)
                else:
                    if f'padding_mask{i}' in pred and pred[f'padding_mask{i}'] is not None:
                        out_reduced = out[:, 1:, :]
                        padding_mask = pred[f'padding_mask{i}'][:, 1:]
                        non_padded_avg = []
                        for i in range(out_reduced.size(0)):
                            non_padded_avg.append(out_reduced[i][~padding_mask[i]].mean(dim=0)) # list of B*(tensors of shape (C,))
                        out_reduced = torch.stack(non_padded_avg) # list of B*(tensors of shape (C,)) -> BC
                    else:
                        out_reduced = out[:, 1:, :].mean(dim=1)

                out_reduced = F.normalize(out_reduced, dim=-1)

                X_data_dict[agg_strategy][i].append(out_reduced.cpu())

    for agg_strategy in agg_strategies:
        for i in range(2):
            X_data_dict[agg_strategy][i] = torch.cat(X_data_dict[agg_strategy][i], dim=0)

    return X_data_dict


def unimodal_zero_shot_pair_retrieval(model,
                                      train_loader:DataLoader,
                                      device:str,
                                      name:str) -> None:
    
    memory_bank = _get_zero_shot_pair_retrieval_embeddings(model=model, dataloader=train_loader, device=device)

    for agg_strategy in ['CLS', 'mean', 'mean_without_CLS']:

        similarity_scores = memory_bank[agg_strategy][0] @ memory_bank[agg_strategy][1].t()
        similarity_scores_t = similarity_scores.t()

        pair0_to_1_r1 = compute_recall(similarity_scores, k=1)
        pair0_to_1_r5 = compute_recall(similarity_scores, k=5)
        pair1_to_0_r1 = compute_recall(similarity_scores_t, k=1)
        pair1_to_0_r5 = compute_recall(similarity_scores_t, k=5)

        logger.info(f"{agg_strategy} unimodal-{name}-pair0_1-retrieval--zeroshot-recall@1: {pair0_to_1_r1}")
        logger.info(f"{agg_strategy} unimodal-{name}-pair0_1-retrieval--zeroshot-recall@5: {pair0_to_1_r5}")
        logger.info(f"{agg_strategy} unimodal-{name}-pair1_0-retrieval--zeroshot-recall@1: {pair1_to_0_r1}")
        logger.info(f"{agg_strategy} unimodal-{name}-pair1_0-retrieval--zeroshot-recall@5: {pair1_to_0_r5}")


def perform_representation_test(model, datamodules, pair, device) -> None:
    for name, datamodule in datamodules.items():
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        datamodule.setup(stage='test')
        
        if not pair:
            unimodal_zero_shot_retrieval(
                model=model,
                train_loader=datamodule.train_dataloader(),
                test_loader=datamodule.test_dataloader(),
                device=device,
                name=name,
                )
        else:
            unimodal_zero_shot_pair_retrieval(
                model=model,
                train_loader=datamodule.train_dataloader(),
                device=device,
                name=name,
                )
        

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_path = os.path.join('..', 'configs', 'training')

    try:
        initialize(version_base=None, config_path=cfg_path)
    except ValueError:
        pass # already initialized
    cfg = compose(config_name="simple_kd_mm_d2v")

    val_cfg = cfg.zero_shot_val
    zero_shot_modules = dict()
    val_dataloader_args = val_cfg.dataloader
    for name in val_cfg.datamodules:
        with open_dict(val_dataloader_args):
            # override general dataloader args with dataloader specific args (if present)
            args = OmegaConf.merge(val_dataloader_args, val_cfg.datamodules[name])

        zero_shot_modules[name] = DATAMODULE_REGISTRY[name](**args)
        
    d2v = load_pretrained_d2v_model(state_dict_path=os.path.join(cfg.model.pretrained_path, cfg.model.pretrained.image))
    d2v = d2v.to(device)

    image_datamodules = {key: value for key, value in zero_shot_modules.items() if 'cifar' in key}

    perform_representation_test(model=d2v, datamodules=image_datamodules, pair=False, device=device)

    d2v = load_pretrained_d2v_model(state_dict_path=os.path.join(cfg.model.pretrained_path, cfg.model.pretrained.audio))
    d2v = d2v.to(device)
    
    audio_datamodules = {key: value for key, value in zero_shot_modules.items() if 'speech' in key}

    perform_representation_test(model=d2v, datamodules=audio_datamodules, pair=False, device=device)

    d2v = load_pretrained_d2v_model(state_dict_path=os.path.join(cfg.model.pretrained_path, cfg.model.pretrained.text))
    d2v = d2v.to(device)

    text_datamodules = {key: value for key, value in zero_shot_modules.items() if 'qqp' in key}

    perform_representation_test(model=d2v, datamodules=text_datamodules, pair=True, device=device)

if "__main__" == __name__:
    main()