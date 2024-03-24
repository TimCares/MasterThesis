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
from torchvision.transforms import v2 as transforms
from torchvision.datasets import CIFAR10, CIFAR100


logger = logging.getLogger(__name__)

@torch.no_grad()
@rank_zero_only
def make_knn_predictions(model:Callable,
                         n_neighbors:int,
                         train_loader:DataLoader,
                         test_loader:DataLoader,
                         name:str) -> Tuple[Callable, float]:
    X_train = []
    y_train = []
    for batch in train_loader:
        X_train.append(batch[0])
        y_train.append(batch[1])
    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train, dim=0)

    X_test = []
    y_test = []
    for batch in test_loader:
        X_test.append(batch[0])
        y_test.append(batch[1])
    X_test = torch.cat(X_test, dim=0)
    y_test = torch.cat(y_test, dim=0)

    y_train = y_train.cpu().numpy()
    y_test = y_test.cpu().numpy()       
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    logger.info(f"Training KNN with {n_neighbors} neighbors")
    knn.fit(model(X_train).cpu().numpy(), y_train)
    logger.info(f"Predicting with KNN")
    y_hat_test = knn.predict_proba(model(X_test).cpu().numpy())
    acc = accuracy_score(y_test, y_hat_test.argmax(axis=1)) # .argmax(axis=1) -> convert class scores to class labels
    acc5 = top_k_accuracy_score(y_test, y_hat_test, k=5)
    logger.info(f"{name}, zero-shot: top1-accuracy: {acc}, top5-accuracy: {acc5}")

    results = {}
    results[f"{name}--zeroshot-test-top1"] = acc
    results[f"{name}--zeroshot-test-top5"] = acc5
    return knn, results


class ZeroShotCallback(Callback):
    """
    datasets: Dict[str, LightningDataModule] -> Dict of dataloaders, keys are the names of the dataloaders.
    """
    def __init__(self, n_neighbors:int, datasets: Dict[str, LightningDataModule], *args, **kwargs):
        super().__init__()
        self.dataset = datasets
        self.n_neighbors = n_neighbors

    @torch.no_grad()
    def on_validation_start(self, trainer, pl_module, **kwargs) -> None:
        for name_key in self.dataset:
            _, metrics = make_knn_predictions(
                model=pl_module.model,
                n_neighbors=self.n_neighbors,
                train_loader=self.dataset.train_dataloader(),
                test_loader=self.dataset.val_dataloader(),
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


class CIFARDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32, type: str = "cifar10"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.type = type
        self.transform = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.Resize((224, 224)),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def prepare_data(self):
        if self.type == "cifar10":
            CIFAR10(self.data_dir, train=True, download=True)
            CIFAR10(self.data_dir, train=False, download=True)
        else:
            CIFAR100(self.data_dir, train=True, download=True)
            CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.type == "cifar10":
                self.train = CIFAR10(self.data_dir, train=True, transform=self.transform)
                self.val = CIFAR10(self.data_dir, train=False, transform=self.transform)
            else:
                self.train = CIFAR100(self.data_dir, train=True, transform=self.transform)
                self.val = CIFAR100(self.data_dir, train=False, transform=self.transform)
        if stage == "test" or stage is None:
            if self.type == "cifar10":
                self.test = CIFAR10(self.data_dir, train=False, transform=self.transform)
            else:
                self.test = CIFAR100(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)