from typing import Tuple
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import Callback, LightningDataModule
from pytorch_lightning.utilities import rank_zero_only
from typing import Dict
import os
import sys
sys.path.append('..')
from data.imagenet_zeroshot_data import (
    imagenet_classnames,
    openai_imagenet_template,
)
from data.filip_zero_shot_data import filip_prompt_templates
from modules.cmli import infer_cmli_logits
from modules import mask_eos
from rich.progress import track
from fairseq.data import Dictionary
from utils import pad_text_sequence # src/utils.py
from data2vec_fairseq.data.modality import Modality
from transformers import BertTokenizer


logger = logging.getLogger(__name__)

def filip_zero_shot(
    pl_module,
    device,
    num_max_bpe_tokens:int,):

    data_path = '/workspace'
    bpe_encoder:BPEEncoder = get_bpe_encoder(data_path)
    dictionary = Dictionary.load(os.path.join(data_path, "dict.txt"))
    
    texts = bpe_encoder.encode_lines(
        filip_prompt_templates,
        tokens_per_sample=num_max_bpe_tokens,
        to_tensor=False,
    )
    padding_masks = []
    for i in range(len(texts)):
        language_tokens, padding_mask = pad_text_sequence(tokens=texts[i], num_max_bpe_tokens=num_max_bpe_tokens,
                                                          pad_idx=dictionary.pad(), bos_idx=dictionary.bos(),
                                                          eos_idx=dictionary.eos(),)
        
        texts[i] = language_tokens
        padding_masks.append(padding_mask)
    
    texts = torch.tensor(texts, dtype=torch.long)
    padding_masks = torch.tensor(padding_masks, dtype=torch.long)
    assert texts.size(1) == num_max_bpe_tokens
    assert padding_masks.size(1) == num_max_bpe_tokens

    stacked_classifier = []
    for i in track(range(0, texts.shape[0], 256), description="Building FILIP classifier"):
        text_chunk = texts[i:i+256].to(device)
        padding_mask_chunk = padding_masks[i:i+256].to(device)
        result_chunk = pl_module.model.encode_text(text=text_chunk, padding_mask=padding_mask_chunk)['x_raw']
        # result_chunk ->  (256, num_max_bpe_tokens, embed_dim)
        result_chunk = result_chunk / result_chunk.norm(dim=-1, keepdim=True)
        stacked_classifier.append(result_chunk)
    stacked_classifier = torch.cat(stacked_classifier, dim=0) # (30000, num_max_bpe_tokens, embed_dim)
    assert stacked_classifier.shape[0] == 30_000

    padding_masks = mask_eos(padding_masks)

    return stacked_classifier, padding_masks.to(device)

@rank_zero_only
def run_filip_zero_shot(
    pl_module,
    dataloader:DataLoader,
    num_max_bpe_tokens:int,
    device,
    name,):

    logger.info(f"Starting multimodal {name} Zero-Shot Eval")
    logger.info("Building classifier")
    classifier, padding_mask = filip_zero_shot(pl_module, device, num_max_bpe_tokens)
    logger.info("Classifier built")
    top1, top5, n = 0.0, 0.0, 0.0
    n_images_for_cmli = 5
    for sample in track(dataloader, description=f"Zero-shot eval: {name}"):
        logits = []
        images = sample["image"]
        target = sample["target"]
        images = images.to(device)
        target = target.to(device)
        if pl_module.dtype == torch.float16: # when using deep speed
            images = images.half()
        image_features = pl_module.model.encode_image(image=images)['x_raw'] # (bsz, 197, embed_dim)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        for image_input in image_features.split(n_images_for_cmli):
            image_to_text = infer_cmli_logits(
                text_features=classifier,
                image_features=image_input,
                padding_mask=padding_mask,
                logit_scale=100.0,
            )['i2t'].t().view(-1, 1000, 30).mean(dim=-1) # (n_images_for_cmli, 1000)
            logits.append(image_to_text)
        
        logits = torch.cat(logits, dim=0)
        assert logits.shape[0] == images.size(0)
        assert logits.shape[1] == 1000

        # measure accuracy
        acc1, acc5 = _n_correct(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

    top1 = top1 / n
    top5 = top5 / n
    results = {}
    results[f"multimodal-{name}--zeroshot-val-top1"] = top1
    results[f"multimodal-{name}--zeroshot-val-top5"] = top5
    return results


def _zero_shot_classifier(pl_module, device, num_max_bpe_tokens):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    zeroshot_weights = []
    for classname in track(imagenet_classnames, description="Building classifier"):
        texts = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(template(classname)))
                 for template in openai_imagenet_template]
        padding_masks = []
        for i in range(len(texts)):
            language_tokens, padding_mask = pad_text_sequence(tokens=texts[i], num_max_bpe_tokens=num_max_bpe_tokens,
                                                              pad_idx=tokenizer.pad_token_id, bos_idx=tokenizer.cls_token_id,
                                                              eos_idx=tokenizer.sep_token_id,)
            
            texts[i] = language_tokens
            padding_masks.append(padding_mask)

        texts = torch.tensor(texts, dtype=torch.long)
        padding_masks = torch.tensor(padding_masks, dtype=torch.long)
        assert texts.size(1) == num_max_bpe_tokens
        assert padding_masks.size(1) == num_max_bpe_tokens

        texts = texts.to(device)
        padding_masks = padding_masks.to(device)
        # check for fp16 not needed here -> texts is long tensor and will be converted by embedding table to correct dtype
        class_embeddings = pl_module.model.encode_text(text=texts, padding_mask=padding_masks)['x']
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)

    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def _n_correct(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        correct[:k].reshape(-1).float().sum().cpu().item()
        for k in topk
    ]


@rank_zero_only
def run_multimodal_zero_shot(pl_module,
                             dataloader:DataLoader,
                             num_max_bpe_tokens:int,
                             device,
                             name,):
    logger.info(f"Starting multimodal {name} Zero-Shot Eval")
    logger.info("Building classifier")
    classifier = _zero_shot_classifier(pl_module, device, num_max_bpe_tokens)
    logger.info("Classifier built")
    top1, top5, n = 0.0, 0.0, 0.0
    for sample in track(dataloader, description=f"Zero-shot eval: {name}"):
        images = sample["image"]
        target = sample["target"]
        images = images.to(device)
        target = target.to(device)
        if pl_module.dtype == torch.float16: # when using deep speed
            images = images.half()
        image_features = pl_module.model.encode_image(image=images)['x']
        logits = 100.0 * image_features @ classifier

        # measure accuracy
        acc1, acc5 = _n_correct(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

    top1 = top1 / n
    top5 = top5 / n
    results = {}
    results[f"multimodal-{name}--zeroshot-val-top1"] = top1
    results[f"multimodal-{name}--zeroshot-val-top5"] = top5
    return results


def _get_zero_shot_retrieval_embeddings(model, dataloader:DataLoader, device:str) -> Tuple[torch.Tensor, torch.Tensor]:
    embedding_table = []
    ground_truth = []
    for batch in dataloader:
        x = batch['x'].to(device)
        padding_mask = batch['padding_mask'].to(device) if 'padding_mask' in batch else None
        # encoding also normalizes the output
        emb = model.encode_modality(x=x, modality=batch['modality'], padding_mask=padding_mask)
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
def unimodal_zero_shot_retrieval(model,
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

def _get_zero_shot_pair_retrieval_embeddings(model, dataloader:DataLoader, device:str) -> Tuple[torch.Tensor, torch.Tensor]:
    embedding_tables = {i: [] for i in range(2)}
    for batch in dataloader:
        for i in range(2): # for each element in the pair
            x = batch[f"x{i}"].to(device)
            padding_mask = batch[f'padding_mask{i}'].to(device) if f'padding_mask{i}' in batch else None # "padding_mask0", "padding_mask1"
            # encoding also normalizes the output
            emb = model.encode_modality(x=x, modality=batch['modality'], padding_mask=padding_mask)
            embedding_tables[i].append(emb.detach().cpu())

    return torch.cat(embedding_tables[0], 0), torch.cat(embedding_tables[1], 0)

@rank_zero_only
def unimodal_zero_shot_pair_retrieval(model,
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
    def __init__(
            self,
            datamodules: Dict[str, LightningDataModule],):
        super().__init__()
        self.datamodules = datamodules
        self.has_resumed = False

    def validate(self, trainer, pl_module) -> None:
        raise NotImplementedError

    @torch.no_grad()
    @rank_zero_only
    def on_validation_start(self, trainer, pl_module, **kwargs) -> None:
        if self.check_if_just_resumed():
            return

        for name_key in self.datamodules.keys(): # setup datamodules
            self.datamodules[name_key].prepare_data()
            self.datamodules[name_key].setup(stage='fit')
            self.datamodules[name_key].setup(stage='test')
        
        self.validate(trainer, pl_module)

        for name_key in self.datamodules.keys(): # cleanup datamodules
            self.datamodules[name_key].teardown(stage='fit')
            self.datamodules[name_key].teardown(stage='test')

    def check_if_just_resumed(self):
        if self.has_resumed:
            logger.info("Resumed from checkpoint. Skipping repeated zero-shot validation.")
            self.has_resumed = False
            return True
        return False # else

    def on_train_start(self, trainer, pl_module):
        self.has_resumed = trainer.ckpt_path is not None


class ZeroShotRetrievalCallback(ZeroShotCallback):
    @torch.no_grad()
    @rank_zero_only
    def validate(self, trainer, pl_module) -> None:
        all_metrics_for_modality = dict()
        for name_key in self.datamodules.keys():
            modality:Modality = self.datamodules[name_key].modality
            if modality.name.lower() not in all_metrics_for_modality.keys():
                all_metrics_for_modality[modality.name.lower()] = []
            
            if modality != Modality.VL:
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
                    on_epoch=True,
                    )
                all_metrics_for_modality[modality.name.lower()].append(metrics[metric_key])
        
        if len(all_metrics_for_modality.keys()) > 1:
            mean_scores = []
        else:
            mean_scores = None
        
        for modality in all_metrics_for_modality.keys():
            mean_score = np.mean(all_metrics_for_modality[modality])
            pl_module.log(
                f"val/unimodal-{modality}-mean-retrieval--zeroshot",
                mean_score,
                prog_bar=True,
                logger=True,
                rank_zero_only=True,
                on_epoch=True,
                )
            if mean_scores is not None:
                mean_scores.append(mean_score)
        
        if mean_scores is not None:
            mean_score = np.mean(mean_scores)
            pl_module.log(
                    "val/unimodal-mean-retrieval--zeroshot",
                    mean_score,
                    prog_bar=True,
                    logger=True,
                    rank_zero_only=True,
                    on_epoch=True,
                    )


class MultimodalZeroShotRetrievalCallback(ZeroShotCallback):
    def __init__(
            self,
            datamodules: Dict[str, LightningDataModule],
            num_max_bpe_tokens:int):
        super().__init__(
            datamodules=datamodules,
        )
        self.num_max_bpe_tokens = num_max_bpe_tokens


    @torch.no_grad()
    @rank_zero_only
    def validate(self, trainer, pl_module) -> None:
        for name_key in self.datamodules.keys():
            
            metrics = run_multimodal_zero_shot(
                pl_module=pl_module,
                dataloader=self.datamodules[name_key].val_dataloader(),
                num_max_bpe_tokens=self.num_max_bpe_tokens,
                device=pl_module.device,
                name=name_key,
            )

            for metric_key in metrics:
                pl_module.log(
                    f"val/{metric_key}",
                    metrics[metric_key],
                    logger=True,
                    rank_zero_only=True,
                    on_epoch=True,
                )
