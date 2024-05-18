from datasets_ import GLUE_DATASET_REGISTRY
from .unimodal_datamodules import BaseDataModule
from functools import partial

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
        if dataset == 'mnli':
            self.val_split_name = 'dev_matched'
        elif dataset == 'mrpc':
            self.val_split_name = 'test'
        self.num_max_bpe_tokens = num_max_bpe_tokens
    
    def set_train_dataset(self):
        self.train_dataset = GLUE_DATASET_REGISTRY[self.dataset](
            data_path=self.data_path, 
            split='train', 
            num_max_bpe_tokens=self.num_max_bpe_tokens
        )

    def set_val_dataset(self):
        self.val_dataset = GLUE_DATASET_REGISTRY[self.dataset](
            data_path=self.data_path, 
            split=self.val_split_name, 
            num_max_bpe_tokens=self.num_max_bpe_tokens
        )


GLUE_DATAMODULE_REGISTRY = {
    dataset: partial(GLUEDataModule, dataset=dataset) for dataset in GLUE_DATASET_REGISTRY.keys()
}