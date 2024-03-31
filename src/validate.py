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


logger = logging.getLogger(__name__)

# mach bei meinem anwendungsfall keinen sinn, also vielleicht später genau den code von flava verwenden
# def _zero_shot_classifier(model,
#                           train_loader:DataLoader,
#                           device,
#                           *args, 
#                           **kwargs):
#     zeroshot_weights = []
#     for batch in train_loader:
#         batch = batch.to(device)
#         embeddings = model(batch['input']) # TODO add parameters, and [:, 0, :], if not done in the model (no causal mask!)
#         embeddings /= embeddings.norm(dim=-1, keepdim=True)
#         zeroshot_weights.append(embeddings)
# 
#     zeroshot_weights = torch.cat(zeroshot_weights, dim=0).to(device)
#     return zeroshot_weights
# 
# 
# def _accuracy(output, target, topk=(1,)):
#     pred = output.topk(max(topk), 1, True, True)[1].t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#     return [
#         float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
#         for k in topk
#     ]
# 
# 
# @rank_zero_only
# def make_zero_shot(model,
#                    train_loader:DataLoader,
#                    test_loader:DataLoader,
#                    name:str, 
#                    device,
#                    *args, 
#                    **kwargs):
#     logger.info("Starting ImageNet Zero-Shot Eval")
#     logger.info("Building classifier")
#     classifier = _zero_shot_classifier(model, device, train_loader)
#     logger.info("Classifier built")
#     top1, top5, n = 0.0, 0.0, 0.0
#     for sample in test_loader:
#         images = sample["image"]
#         target = sample["label"]
#         images = images.to(device)
#         target = target.to(device)
# 
#         image_features = model(images) # TODO add parameters, and [:, 0, :], if not done in the model (no causal mask!)
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         logits = 100.0 * image_features @ classifier
# 
#         # measure accuracy
#         acc1, acc5 = _accuracy(logits, target, topk=(1, 5))
#         top1 += acc1
#         top5 += acc5
#         n += images.size(0)
# 
#     top1 = top1 / n
#     top5 = top5 / n
#     results = {}
#     results[f"{name}--zeroshot-val-top1"] = top1
#     results[f"{name}--zeroshot-val-top5"] = top5
#     return results

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
    results[f"{name}--zeroshot-test-top1"] = acc
    results[f"{name}--zeroshot-test-top5"] = acc5
    return knn, results


class ZeroShotCallback(Callback):
    """
    datamodules: Dict[str, LightningDataModule] -> Dict of LightningDataModule, keys are the names of the LightningDataModule.
    """
    def __init__(self, n_neighbors:int, datamodules: Dict[str, LightningDataModule], *args, **kwargs):
        super().__init__()
        self.datamodules = datamodules
        self.n_neighbors = n_neighbors

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
