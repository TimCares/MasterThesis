from bpe_encoder import BPEEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import top_k_accuracy_score
from typing import Tuple, Callable
from sklearn.neighbors import KNeighborsClassifier
import logging
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import Callback, LightningDataModule
from pytorch_lightning.utilities import rank_zero_only
from typing import Dict
import sys
sys.path.append('..')
from data.imagenet_zeroshot_data import (
    imagenet_classnames,
    openai_imagenet_template,
)
from rich.progress import track
from datasets.data_utils import get_bpe_encoder
from fairseq.data.dictionary import Dictionary

logger = logging.getLogger(__name__)


def _zero_shot_classifier(model, device, tokenizer:BPEEncoder, num_max_bpe_tokens, 
                          dictionary:Dictionary, *args, **kwargs):
    zeroshot_weights = []
    for classname in track(imagenet_classnames, description="Building classifier"):
        texts = tokenizer.encode_lines(
            [template(classname) for template in openai_imagenet_template],
            tokens_per_sample=num_max_bpe_tokens-2,
            to_tensor=False,
        )
        padding_masks = []
        for i in range(len(texts)):
            texts[i] = [dictionary.bos()] + texts[i] + [dictionary.eos()]
            length = len(texts[i])
            texts[i] = texts[i] + [dictionary.pad()] * (num_max_bpe_tokens - length)
            padding_mask = [0] * length + [1] * (num_max_bpe_tokens - length)
            padding_masks.append(padding_mask)

        texts = torch.tensor(texts, dtype=torch.long)
        padding_masks = torch.tensor(padding_masks, dtype=torch.long)
        assert texts.size(1) == num_max_bpe_tokens
        assert padding_masks.size(1) == num_max_bpe_tokens

        texts = texts.to(device)
        padding_masks = padding_masks.to(device)
        class_embeddings = model.encode_text(texts) # TODO padding_masks
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)

    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def _accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


@torch.no_grad()
@rank_zero_only
def run_multimodal_zero_shot(model:Callable,
                             dataloader:DataLoader,
                             device,
                             text_transform, 
                             name,
                             *args,
                             **kwargs):
    logger.info(f"Starting multimodal {name} Zero-Shot Eval")
    logger.info("Building classifier")
    classifier = _zero_shot_classifier(model, device, text_transform)
    logger.info("Classifier built")
    top1, top5, n = 0.0, 0.0, 0.0
    for sample in track(dataloader, description=f"Zero-shot eval: {name}"):
        images = sample["image"]
        target = sample["label"]
        images = images.to(device)
        target = target.to(device)

        # predict
        # if hasattr(model, "module"):
        #     image_features = model.module.encode_image({"image": images})
        # else:
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ classifier

        # measure accuracy
        acc1, acc5 = _accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

    top1 = top1 / n
    top5 = top5 / n
    results = {}
    results[f"multimodal-{name}--zeroshot-val-top1"] = top1
    results[f"multimodal-{name}--zeroshot-val-top5"] = top5
    return results


@torch.no_grad()
def _get_knn_data(model, data_loader:DataLoader) ->Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for batch in data_loader:
        source = batch[batch['modes'][0].name.lower()].to(model.device)
        padding_mask = batch['padding_mask'].to(model.device) if 'padding_mask' in batch else None
        out = model.encode_modality(mode=batch['modes'], source=source, padding_mask=padding_mask,
                                    normalize=True) # norm output

        X.append(out)
        y.append(batch['target'])
    # shape after "cat" will be (n_samples, 1, embed_dim)
    # so transform to (n_samples, embed_dim) with "squeeze"
    X = torch.cat(X, dim=0).squeeze(1)
    X = X.cpu().numpy()
    y = torch.cat(y, dim=0).cpu().numpy()
    return X, y


@rank_zero_only # only needed in a distributed setting
def make_knn_predictions(model:Callable,
                         n_neighbors:int,
                         train_loader:DataLoader,
                         test_loader:DataLoader,
                         name:str) -> Tuple[Callable, float]:
    
    X_train, y_train = _get_knn_data(model=model, data_loader=train_loader)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    logger.info(f"Training KNN with {n_neighbors} neighbors")
    knn.fit(X_train, y_train)

    X_test, y_test = _get_knn_data(model=model, data_loader=test_loader)
 
    logger.info(f"Predicting with KNN")
    y_hat_test = knn.predict_proba(X_test)
    acc = accuracy_score(y_test, y_hat_test.argmax(axis=1)) # .argmax(axis=1) -> convert class scores to class labels
    acc5 = top_k_accuracy_score(y_test, y_hat_test, k=5)
    logger.info(f"{name}, zero-shot: top1-accuracy: {acc}, top5-accuracy: {acc5}")

    results = {}
    results[f"unimodal-{name}-knn--zeroshot-top1-acc"] = acc
    results[f"unimodal-{name}-knn--zeroshot-top5-acc"] = acc5
    return knn, results


class ZeroShotCallback(Callback):
    """
    datamodules: Dict[str, LightningDataModule] -> Dict of LightningDataModule, keys are the names of the LightningDataModule.
    """
    def __init__(self, n_neighbors:int, datamodules: Dict[str, LightningDataModule], data_path:str, num_max_bpe_tokens:int,
                 is_multimodal_aligned:bool, *args, **kwargs):
        super().__init__()
        self.datamodules = datamodules
        self.n_neighbors = n_neighbors
        self.encoder = get_bpe_encoder(data_path)
        self.num_max_bpe_tokens = num_max_bpe_tokens # TODO: utilize later...
        self.dictionary = Dictionary.load(os.path.join(data_path, 'dict.txt')) # TODO: utilize later...
        self.is_multimodal_aligned = is_multimodal_aligned # TODO: utilize later...

    @torch.no_grad()
    def on_validation_start(self, trainer, pl_module, **kwargs) -> None:
        for name_key in self.datamodules.keys():
            self.datamodules[name_key].prepare_data()
            self.datamodules[name_key].setup(stage='train')
            self.datamodules[name_key].setup(stage='test')
            _, metrics = make_knn_predictions(
                model=pl_module.model,
                n_neighbors=self.n_neighbors,
                train_loader=self.datamodules[name_key].train_dataloader(),
                test_loader=self.datamodules[name_key].test_dataloader(),
                name=name_key,
            )
            if metrics is not None:
                for metric_key in metrics:
                    self.log(
                        f"val/{metric_key}",
                        metrics[metric_key],
                        prog_bar=True,
                        logger=True,
                        rank_zero_only=True,
                    )
