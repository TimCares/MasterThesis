import torch
import torch.nn.functional as F
from utils import load_pretrained_d2v_model
import os
from hydra import compose, initialize
from omegaconf import OmegaConf, open_dict
from datamodules import DATAMODULE_REGISTRY
from typing import *
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import top_k_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from rich.progress import track
import logging

logger = logging.getLogger(__name__)

@torch.no_grad()
def _get_knn_data(model, data_loader:DataLoader, device:str) ->Tuple[Dict[str, np.ndarray], np.ndarray]:
    agg_strategies = ['CLS', 'mean', 'mean_without_CLS']
    X_data_dict = {strategy: [] for strategy in agg_strategies}
    y = []

    for batch in track(data_loader): 
        source = batch[batch['modes'][0].name.lower()].to(device)
        padding_mask = batch['padding_mask'].to(device) if 'padding_mask' in batch else None
        pred = model.extract_features(
                source=source,
                mode=None, # determined automatically in model
                padding_mask=padding_mask,
                mask=False, # we are creating targets from a teacher model for the student model, so no mask
                remove_extra_tokens=False,
            )
        out = pred['x']
        
        y.append(batch['target']) # only append once

        for agg_strategy in agg_strategies:
            if agg_strategy=="CLS":
                out_reduced = out[:, 0, :].squeeze(1)
            elif agg_strategy=="mean":
                non_padded_avg = []
                for i in range(out.size(0)):
                    non_padded_avg.append(out[i][~pred['padding_mask'][i]].mean(dim=0)) # list of B*(tensors of shape (C,))
                out_reduced = torch.stack(non_padded_avg) # list of B*(tensors of shape (C,)) -> BC
                # out_reduced = out.mean(dim=1)
            else:
                out_reduced = out[:, 1:, :]
                padding_mask = pred['padding_mask'][:, 1:]
                non_padded_avg = []
                for i in range(out_reduced.size(0)):
                    non_padded_avg.append(out_reduced[i][~padding_mask[i]].mean(dim=0)) # list of B*(tensors of shape (C,))
                out_reduced = torch.stack(non_padded_avg) # list of B*(tensors of shape (C,)) -> BC
                #out_reduced = out[:, 1:, :].mean(dim=1)

            out_reduced = F.normalize(out_reduced, dim=-1)

            X_data_dict[agg_strategy].append(out_reduced.cpu())

    for agg_strategy in agg_strategies:
        X_data_dict[agg_strategy] = torch.cat(X_data_dict[agg_strategy], dim=0).numpy()

    y = torch.cat(y, dim=0).cpu().numpy()

    return X_data_dict, y

def make_knn_predictions(model:Callable,
                         n_neighbors:int,
                         train_loader:DataLoader,
                         test_loader:DataLoader,
                         device:str,
                         name:str,) -> None:
    
    X_train, y_train = _get_knn_data(model=model, data_loader=train_loader, device=device)
    X_test, y_test = _get_knn_data(model=model, data_loader=test_loader, device=device)

    for agg_strategy in ['CLS', 'mean', 'mean_without_CLS']:

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        logger.info(f"{agg_strategy}: Training KNN with {n_neighbors} neighbors")
        knn.fit(X_train[agg_strategy], y_train)
    
        results = {}

        logger.info(f"{agg_strategy}: Predicting with KNN")
        y_hat_test = knn.predict_proba(X_test[agg_strategy])
        acc = accuracy_score(y_test, y_hat_test.argmax(axis=1)) # .argmax(axis=1) -> convert class scores to class labels
        results[f"unimodal-{name}-knn--zeroshot-top1-acc"] = acc
        logger.info(f"{name}, zero-shot: top1-accuracy: {acc}")
        try: # if data has less than 5 classes, top5-accuracy will throw an error
            acc5 = top_k_accuracy_score(y_test, y_hat_test, k=5)
            results[f"unimodal-{name}-knn--zeroshot-top5-acc"] = acc5
            logger.info(f"{name}, zero-shot: top5-accuracy: {acc5}")
        except:
            pass

        for key in results:
            print(f"{agg_strategy} {name} {key}: {results[key]}")

def perform_representation_test(model, datamodules, n_neighbors, device) -> None:
    for name, datamodule in datamodules.items():
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        datamodule.setup(stage='test')
        
        make_knn_predictions(
            model=model,
            n_neighbors=n_neighbors,
            train_loader=datamodule.train_dataloader(),
            test_loader=datamodule.test_dataloader(),
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

        zero_shot_modules[name[1:]] = DATAMODULE_REGISTRY[name[1:]](**args)
        
    d2v = load_pretrained_d2v_model(state_dict_path=os.path.join(cfg.model.pretrained_path, cfg.model.pretrained.image))
    d2v = d2v.to(device)

    image_datamodules = {key: value for key, value in zero_shot_modules.items() if 'cifar' in key}

    perform_representation_test(model=d2v, datamodules=image_datamodules, n_neighbors=val_cfg.n_neighbors, device=device)

    d2v = load_pretrained_d2v_model(state_dict_path=os.path.join(cfg.model.pretrained_path, cfg.model.pretrained.audio))
    d2v = d2v.to(device)
    
    audio_datamodules = {key: value for key, value in zero_shot_modules.items() if 'speech' in key}

    perform_representation_test(model=d2v, datamodules=audio_datamodules, n_neighbors=val_cfg.n_neighbors, device=device)

    d2v = load_pretrained_d2v_model(state_dict_path=os.path.join(cfg.model.pretrained_path, cfg.model.pretrained.text))
    d2v = d2v.to(device)

    text_datamodules = {key: value for key, value in zero_shot_modules.items() if 'imdb' in key}

    perform_representation_test(model=d2v, datamodules=text_datamodules, n_neighbors=val_cfg.n_neighbors, device=device)

if "__main__" == __name__:
    main()