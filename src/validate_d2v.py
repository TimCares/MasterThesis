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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def _get_knn_data(model, data_loader:DataLoader, device:str, agg_strategy:str) ->Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for batch in data_loader: 
        source = batch[batch['modes'][0].name.lower()].to(device)
        padding_mask = batch['padding_mask'].to(device) if 'padding_mask' in batch else None
        out = model.extract_features(
                source=source,
                mode=None, # determined automatically in model
                padding_mask=padding_mask,
                mask=False, # we are creating targets from a teacher model for the student model, so no mask
                remove_extra_tokens=False,
            )['x']
        
        if agg_strategy=="CLS":
            out = out[:, 0, :].squeeze(1)
        elif agg_strategy=="mean":
            out = out.mean(dim=1)
        else:
            out = out[:, 1:, :].mean(dim=1)

        out = F.normalize(out, dim=-1)

        X.append(out.cpu())
        y.append(batch['target'])
    # shape after "cat" will be (n_samples, embed_dim)
    X = torch.cat(X, dim=0)
    X = X.numpy()
    y = torch.cat(y, dim=0).cpu().numpy()
    return X, y

def make_knn_predictions(model:Callable,
                         n_neighbors:int,
                         train_loader:DataLoader,
                         test_loader:DataLoader,
                         device:str,
                         name:str,
                         agg_strategy:str) -> Tuple[Callable, float]:
    
    X_train, y_train = _get_knn_data(model=model, data_loader=train_loader, device=device, agg_strategy=agg_strategy)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    logger.info(f"Training KNN with {n_neighbors} neighbors")
    knn.fit(X_train, y_train)

    X_test, y_test = _get_knn_data(model=model, data_loader=test_loader, device=device, agg_strategy=agg_strategy)
 
    results = {}

    logger.info(f"Predicting with KNN")
    y_hat_test = knn.predict_proba(X_test)
    acc = accuracy_score(y_test, y_hat_test.argmax(axis=1)) # .argmax(axis=1) -> convert class scores to class labels
    results[f"unimodal-{name}-knn--zeroshot-top1-acc"] = acc
    logger.info(f"{name}, zero-shot: top1-accuracy: {acc}")
    try: # if data has less than 5 classes, top5-accuracy will throw an error
        acc5 = top_k_accuracy_score(y_test, y_hat_test, k=5)
        results[f"unimodal-{name}-knn--zeroshot-top5-acc"] = acc5
        logger.info(f"{name}, zero-shot: top5-accuracy: {acc5}")
    except:
        pass

    return knn, results

def perform_representation_test(model, datamodules, n_neighbors) -> None:
    for agg_strategy in ['CLS', 'mean', 'mean_without_CLS']:
        for name, datamodule in datamodules.items():
            datamodule.prepare_data()
            datamodule.setup(stage='fit')
            datamodule.setup(stage='test')
            
            _, metrics = make_knn_predictions(
                model=model,
                n_neighbors=n_neighbors,
                train_loader=datamodule.train_dataloader(),
                test_loader=datamodule.test_dataloader(),
                device=device,
                name=name,
                agg_strategy=agg_strategy
                )
            for key in metrics:
                print(f"{agg_strategy} {name} {key}: {metrics[key]}")


def main():
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
        with open_dict(val_cfg.datamodules[name]):
            args = OmegaConf.merge(val_cfg.datamodules[name], val_dataloader_args)

        zero_shot_modules[name[1:]] = DATAMODULE_REGISTRY[name[1:]](**args)
        
    d2v = load_pretrained_d2v_model(state_dict_path=os.path.join(cfg.model.pretrained_path, cfg.model.pretrained.image))
    d2v = d2v.to(device)

    image_datamodules = {key: value for key, value in zero_shot_modules.items() if 'cifar' in key}

    perform_representation_test(model=d2v, datamodules=image_datamodules, n_neighbors=val_cfg.n_neighbors)

    d2v = load_pretrained_d2v_model(state_dict_path=os.path.join(cfg.model.pretrained_path, cfg.model.pretrained.audio))
    d2v = d2v.to(device)
    
    audio_datamodules = {key: value for key, value in zero_shot_modules.items() if 'speech' in key}

    perform_representation_test(model=d2v, datamodules=audio_datamodules, n_neighbors=val_cfg.n_neighbors)

    d2v = load_pretrained_d2v_model(state_dict_path=os.path.join(cfg.model.pretrained_path, cfg.model.pretrained.text))
    d2v = d2v.to(device)

    text_datamodules = {key: value for key, value in zero_shot_modules.items() if 'imdb' in key}

    perform_representation_test(model=d2v, datamodules=text_datamodules, n_neighbors=val_cfg.n_neighbors)

if "__main__" == __name__:
    main()