from bpe_encoder import BPEEncoder
from typing import Tuple, Callable
import logging
import torch
import numpy as np
from multimodal_data2vec import KDMMData2Vec
from torch.utils.data import DataLoader
from pytorch_lightning import Callback, LightningDataModule, Trainer, LightningModule
from pytorch_lightning.utilities import rank_zero_only
from typing import Dict
import sys
sys.path.append('..')
from data.imagenet_zeroshot_data import (
    imagenet_classnames,
    openai_imagenet_template,
)
from rich.progress import track
from bpe_encoder import get_bpe_encoder
from fairseq.data import Dictionary
from utils import pad_text_sequence

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
            language_tokens, padding_mask = pad_text_sequence(tokens=texts[i], num_max_bpe_tokens=num_max_bpe_tokens,
                                                              pad_idx=dictionary.pad(), bos_idx=dictionary.bos())
            
            texts[i] = language_tokens
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
def _get_zero_shot_retrieval_embeddings(model:KDMMData2Vec, dataloader:DataLoader, device:str) -> Tuple[torch.Tensor, torch.Tensor]:
    embedding_table = []
    ground_truth = []
    for batch in dataloader:
        source = batch[batch['modes'][0].name.lower()].to(device)
        padding_mask = batch['padding_mask'].to(device) if 'padding_mask' in batch else None
        # encoding also normalizes the output
        emb = model.encode_modality(modes=batch['modes'], source=source, padding_mask=padding_mask, normalize=True)
        embedding_table.append(emb.detach().cpu())
        ground_truth.append(batch['target'])

    embedding_table = torch.cat(embedding_table, 0)
    ground_truth = torch.cat(ground_truth, 0)
    return embedding_table, ground_truth

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

@rank_zero_only
def unimodal_zero_shot_retrieval(model:KDMMData2Vec,
                                 train_loader:DataLoader,
                                 test_loader:DataLoader,
                                 device:str,
                                 name:str) -> Dict[str, float]:
    results = {}
    
    memory_bank, memory_bank_targets = _get_zero_shot_retrieval_embeddings(model=model, dataloader=train_loader, device=device)
    query, query_targets = _get_zero_shot_retrieval_embeddings(model=model, dataloader=test_loader, device=device)

    similarity_scores = query @ memory_bank.t()
    similarity_scores_t = similarity_scores.t()
    train_test_top1 = _get_all_match(similarity_scores, memory_bank_targets, query_targets, k=1)
    test_train_top1 = _get_all_match(similarity_scores_t, query_targets, memory_bank_targets, k=1)

    results[f"unimodal-{name}-retrieval--zeroshot-train-test-top1"] = train_test_top1.item()
    results[f"unimodal-{name}-retrieval--zeroshot-test-train-top1"] = test_train_top1.item()

    try:
        train_test_top3 = _get_all_match(similarity_scores, memory_bank_targets, query_targets, k=3)
        test_train_top3 = _get_all_match(similarity_scores_t, query_targets, memory_bank_targets, k=3)

        results[f"unimodal-{name}-retrieval--zeroshot-train-test-top3"] = train_test_top3.item()
        results[f"unimodal-{name}-retrieval--zeroshot-test-train-top3"] = test_train_top3.item()
    except:
        pass # if data has less than 5 classes, top5-accuracy will throw an error

    return results


def compute_recall(similarity_scores: torch.Tensor, k: int = 5) -> float:
    dataset_size = similarity_scores.size(0)
    targets = torch.arange(dataset_size).view(dataset_size, -1)
    _, topk_idx = torch.topk(similarity_scores, k)
    recall = targets.eq(topk_idx).sum()
    recall = recall / dataset_size
    return recall

@torch.no_grad()
def _get_zero_shot_pair_retrieval_embeddings(model:KDMMData2Vec, dataloader:DataLoader, device:str) -> Tuple[torch.Tensor, torch.Tensor]:
    embedding_tables = {i: [] for i in range(2)}
    for batch in dataloader:
        for i in range(2): # for each element in the pair
            source = batch[f"{batch['modes'][0].name.lower()}{i}"].to(device) # "text0", "text1"
            padding_mask = batch[f'padding_mask{i}'].to(device) if f'padding_mask{i}' in batch else None # "padding_mask0", "padding_mask1"
            # encoding also normalizes the output
            emb = model.encode_modality(modes=batch['modes'], source=source, padding_mask=padding_mask, normalize=True)
            embedding_tables[i].append(emb.detach().cpu())

    return torch.cat(embedding_tables[0], 0), torch.cat(embedding_tables[1], 0)

@rank_zero_only
def unimodal_zero_shot_pair_retrieval(model:KDMMData2Vec,
                                      datamodule:LightningDataModule,
                                      device:str,
                                      name:str) -> Dict[str, float]:
    results = {}
    
    for stage, dataloader in zip(['train', 'test'], [datamodule.train_dataloader(), datamodule.test_dataloader()]):
        memory_bank1, memory_bank2 = _get_zero_shot_pair_retrieval_embeddings(model=model, dataloader=dataloader, device=device)

        similarity_scores = memory_bank1 @ memory_bank2.t()
        similarity_scores_t = similarity_scores.t()

        pair0_to_1_r1 = compute_recall(similarity_scores, k=1)
        pair0_to_1_r5 = compute_recall(similarity_scores, k=5)
        pair1_to_0_r1 = compute_recall(similarity_scores_t, k=1)
        pair1_to_0_r5 = compute_recall(similarity_scores_t, k=5)

        results[f"unimodal-{name}-{stage}-pair1_2-retrieval--zeroshot-recall@1"] = pair0_to_1_r1
        results[f"unimodal-{name}-{stage}-pair1_2-retrieval--zeroshot-recall@5"] = pair0_to_1_r5

        results[f"unimodal-{name}-{stage}-pair2_1-retrieval--zeroshot-recall@1"] = pair1_to_0_r1
        results[f"unimodal-{name}-{stage}-pair2_1-retrieval--zeroshot-recall@5"] = pair1_to_0_r5

    return results

class ZeroShotCallback(Callback):
    """
    datamodules: Dict[str, LightningDataModule] -> Dict of LightningDataModule, keys are the names of the LightningDataModule.
    """
    def __init__(self, datamodules: Dict[str, LightningDataModule], val_every_n_batches:int, ):
        super().__init__()
        self.datamodules = datamodules
        self.val_every_n_batches = val_every_n_batches

    def on_train_batch_end(self, trainer:Trainer, pl_module:LightningModule, outputs, batch, batch_idx):
        # Check if the current batch count is a multiple of the specified frequency
        if trainer.global_step != 0 and trainer.global_step % self.val_every_n_batches == 0:
            pl_module.eval()
            self.validate(trainer, pl_module)
            pl_module.train()

    def validate(self, trainer, pl_module) -> None:
        for name_key in self.datamodules.keys():
            self.datamodules[name_key].prepare_data()
            self.datamodules[name_key].setup(stage='fit')
            self.datamodules[name_key].setup(stage='test')

    def cleanup(self) -> None:
        for name_key in self.datamodules.keys():
            self.datamodules[name_key].teardown(stage='fit')
            self.datamodules[name_key].teardown(stage='test')


class ZeroShotRetrievalCallback(ZeroShotCallback):
    def validate(self, trainer, pl_module) -> None:
        super().validate(trainer, pl_module)
        all_metrics = []
        for name_key in self.datamodules.keys():
            if name_key not in ['qqp', 'mrpc']:
                metrics = unimodal_zero_shot_retrieval(
                    model=pl_module.model,
                    train_loader=self.datamodules[name_key].train_dataloader(),
                    test_loader=self.datamodules[name_key].test_dataloader(),
                    device=pl_module.device,
                    name=name_key,
                )
            else:
                metrics = unimodal_zero_shot_pair_retrieval(
                    model=pl_module.model,
                    datamodule=self.datamodules[name_key],
                    device=pl_module.device,
                    name=name_key,
                )
            for metric_key in metrics:
                pl_module.log(
                    f"val/{metric_key}",
                    metrics[metric_key],
                    logger=True,
                    rank_zero_only=True,
                    on_step=True,
                    )
                all_metrics.append(metrics[metric_key])
        
        mean_score = np.mean(all_metrics)
        pl_module.log(
                "val/unimodal-mean-retrieval--zeroshot",
                mean_score,
                prog_bar=True,
                logger=True,
                rank_zero_only=True,
                on_step=True,
                )
        
        self.cleanup() # release memory
