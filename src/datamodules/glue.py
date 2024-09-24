from datasets_ import GLUE_DATASET_REGISTRY
from .unimodal_datamodules import BaseDataModule
from functools import partial
from data2vec_fairseq.data.modality import Modality
from torch.utils.data import DataLoader

class GLUEDataModule(BaseDataModule):
    def __init__(self,
                 data_path:str,
                 dataset:str,
                 num_max_bpe_tokens:int=512,
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.dataset = dataset
        self.val_split_name = 'dev'
        if self.dataset == 'mrpc_glue':
            self.val_split_name = 'test'
        self.num_max_bpe_tokens = num_max_bpe_tokens

    @property
    def modality(self) -> Modality:
        return Modality.TEXT
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()
            if self.dataset == 'mnli_glue':
                self.val_dataset_1.load()
                self.val_dataset_2.load()
            else:
                self.val_dataset.load()
        if stage == 'test' or stage is None:
            self.test_dataset.load()
    
    def set_train_dataset(self):
        self.train_dataset = GLUE_DATASET_REGISTRY[self.dataset](
            data_path=self.data_path, 
            split='train', 
            num_max_bpe_tokens=self.num_max_bpe_tokens
        )

    def set_val_dataset(self):
        if self.dataset == 'mnli_glue':
            self.val_dataset_1 = GLUE_DATASET_REGISTRY[self.dataset](
                data_path=self.data_path, 
                split='dev_matched', 
                num_max_bpe_tokens=self.num_max_bpe_tokens
            )
            self.val_dataset_2 = GLUE_DATASET_REGISTRY[self.dataset](
                data_path=self.data_path, 
                split='dev_mismatched', 
                num_max_bpe_tokens=self.num_max_bpe_tokens
            )
        else:
            self.val_dataset = GLUE_DATASET_REGISTRY[self.dataset](
                data_path=self.data_path, 
                split=self.val_split_name, 
                num_max_bpe_tokens=self.num_max_bpe_tokens
            )

    def val_dataloader(self):
        if hasattr(self, 'val_dataset'):
            return DataLoader(self.val_dataset,
                            collate_fn=self.val_dataset.collater,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            sampler=None,
                            shuffle=False,
                            drop_last=False,)
        else:
            val_dataloders = [
                DataLoader(self.val_dataset_1,
                    collate_fn=self.val_dataset_1.collater,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    sampler=None,
                    shuffle=False,
                    drop_last=False,),
                DataLoader(self.val_dataset_2,
                    collate_fn=self.val_dataset_2.collater,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    sampler=None,
                    shuffle=False,
                    drop_last=False,)
            ]
            return val_dataloders


GLUE_DATAMODULE_REGISTRY = {
    dataset: partial(GLUEDataModule, dataset=dataset) for dataset in GLUE_DATASET_REGISTRY.keys()
}
