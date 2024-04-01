from bpe_encoder import BPEEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import top_k_accuracy_score
from typing import Tuple, Callable
from sklearn.neighbors import KNeighborsClassifier
import logging
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Callback, LightningDataModule
from pytorch_lightning.utilities import rank_zero_only
from typing import Dict
from data import (
    imagenet_classnames,
    openai_imagenet_template,
)
from rich.progress import track
import data_utils
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
        class_embeddings = model.encode_text(texts) # TODO add parameters, and [:, 0, :], if not done in the model (no causal mask!), and use padding_masks etc.
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
@rank_zero_only # only needed in a distributed setting
def make_knn_predictions(model:Callable,
                         n_neighbors:int,
                         train_loader:DataLoader,
                         test_loader:DataLoader,
                         name:str) -> Tuple[Callable, float]:
    X_train = []
    y_train = []
    for batch in train_loader:
        batch = batch.to(model.device)
        X_train.append(model(batch[0])) # TODO add parameters, and [:, 0, :], if not done in the model (no causal mask!)
        y_train.append(batch[1])
    X_train = torch.cat(X_train, dim=0)
    X_train = X_train / X_train.norm(p=2, dim=-1, keepdim=True) # normalize
    X_train = X_train.cpu().numpy()
    y_train = torch.cat(y_train, dim=0).cpu().numpy()

    X_test = []
    y_test = []
    for batch in test_loader:
        batch = batch.to(model.device)
        X_test.append(model(batch[0])) # TODO add parameters, and [:, 0, :], if not done in the model (no causal mask!)
        y_test.append(batch[1])
    X_test = torch.cat(X_test, dim=0)
    X_test = X_test / X_test.norm(p=2, dim=-1, keepdim=True) # normalize
    X_test = X_test.cpu().numpy()
    y_test = torch.cat(y_test, dim=0).cpu().numpy()
 
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    logger.info(f"Training KNN with {n_neighbors} neighbors")
    knn.fit(X_train, y_train)
    logger.info(f"Predicting with KNN")
    y_hat_test = knn.predict_proba(X_test)
    acc = accuracy_score(y_test, y_hat_test.argmax(axis=1)) # .argmax(axis=1) -> convert class scores to class labels
    acc5 = top_k_accuracy_score(y_test, y_hat_test, k=5)
    logger.info(f"{name}, zero-shot: top1-accuracy: {acc}, top5-accuracy: {acc5}")

    results = {}
    results[f"unimodal-{name}-knn--zeroshot-test-top1"] = acc
    results[f"unimodal-{name}-knn--zeroshot-test-top5"] = acc5
    return knn, results


class ZeroShotCallback(Callback):
    """
    datamodules: Dict[str, LightningDataModule] -> Dict of LightningDataModule, keys are the names of the LightningDataModule.
    """
    def __init__(self, n_neighbors:int, datamodules: Dict[str, LightningDataModule], data_path:str, num_max_bpe_tokens:int,
                 dictionary, is_multimodal_aligned:bool, *args, **kwargs):
        super().__init__()
        self.datamodules = datamodules
        self.n_neighbors = n_neighbors
        self.encoder = data_utils.get_bpe_encoder(data_path)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.dictionary = dictionary
        self.is_multimodal_aligned = is_multimodal_aligned

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
