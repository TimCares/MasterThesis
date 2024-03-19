import numpy as np
from fairseq.data import FairseqDataset


class RoundRobinDataset(FairseqDataset):
    def __init__(
        self,
        datasets
    ):
        FairseqDataset.__init__(self)
        self.datasets = datasets
        self.total_length = sum([len(d) for d in self.datasets])
        self.dataset_index = 0
        self.batch_size = None
        

    def __getitem__(self, index):
        return self.datasets[self.dataset_index][index]
    
    def collater(self, samples):
        batch = self.datasets[self.dataset_index].collater(samples)
        self.dataset_index = (self.dataset_index + 1) % len(self.datasets)
        return batch

    def __len__(self) -> int:
        return self.total_length

    def __repr__(self) -> str:
        pass
    
    def num_tokens(self, index):
        return 1

    def size(self, index):
        return 1

    @property
    def sizes(self):
        return np.full((len(self),), 1)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        return order[0]