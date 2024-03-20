# adapted from: https://github.com/facebookresearch/multimodal/blob/main/examples/common/data/multidata.py
import warnings
from typing import Callable, List, Optional, Dict
import torch

class MultiDataLoader:
    def __init__(
        self,
        loaders: Dict[str, torch.utils.data.DataLoader],
        sampling_func: Optional[Callable] = None,
    ):
        """MultiDataLoader takes in a list of dataloaders and a sampling function
        and cycles between these dataloaders after each batch based on the index
        provided by the sampling function passed. Useful for doing multi-tasking
        over multiple datasets

        Args:
            loaders (Dict[str, torch.utils.data.DataLoader]): Dict of dataloaders on
                which the multitasking has to be done. Keys are the names of the dataloaders.

            sampling_func (Optional[Callable], optional): Function which will return
                the next index to be selected. Defaults to equally weight sampling.
        """
        if loaders is None or len(loaders) == 0:
            warnings.warn(
                "Empty loaders passed into MultiDataLoader. This can have "
                "unintended consequences."
            )

        if sampling_func is None:
            class CycleN:
                def __init__(self, n):
                    self.n = n
                    self.current = -1  # Start at -1 so the first call returns 0

                def __call__(self):
                    self.current = (self.current + 1) % self.n  # Increment and wrap around using modulo
                    return self.current
            sampling_func = CycleN(len(loaders))

        self.sampling_func = sampling_func
        self.loaders = list(loaders.values())
        self.loaders_names = list(loaders.keys())
        self.num_datasets = len(self.loaders)
        self.iterators = [None for _ in loaders]
        self.current_index = 0
        self.set_samplers()

    def set_samplers(self):
        self.samplers: List[torch.utils.data.Sampler] = []
        for loader in self.loaders:
            if hasattr(loader, "sampler"):
                self.samplers.append(loader.sampler)

    def __iter__(self):
        self.iterators = []

        for loader in self.loaders:
            self.iterators.append(iter(loader))

        self.change_dataloader()

        return self

    def __next__(self):
        """
        Calculation of next batch is performed using following logic.

        Current chosen iterator is set in the change_dataloader function
        based on the `sampling_func` function passed to `__init__` of the
        dataloader which is called to get the index of next selected dataloader.

        If we get the next batch from iterator without any StopIteration exception,
        we return it as it is.

        Epochs don't make sense in case of using `sampling_func` unless you add
        extra logic to support epoch-based sampling functions. MMF does this in
        a different way, so take a look at IterationStrategies there to understand
        how this can be possibly done.

        Think of a case of random (equal) proportional sampling for dataset x and y
        where x is half the size of y. When x will complete its 2 epochs, y will
        have only 1 epoch completed. **So please don't use max_epochs or epoch
        based training in this case as it won't be honored**. If an iterator is
        finished, we just reignite it in this case and finished iterators
        variable isn't used. This means that this case will never reach the
        __iter__ function ever again.


        Returns:
            Dict: Contains two keys, one "batch" containing the batch from current
                selected dataloader and "datamodule_index" which is index of
                currently selected dataloader.
        """
        self.change_dataloader()
        try:
            next_batch = next(self.current_iterator)
        except StopIteration:
            iterator = iter(self.loaders[self.current_index])
            self.iterators[self.current_index] = iterator
            self.current_iterator = iterator
            next_batch = next(self.current_iterator)

        return {"batch": next_batch, "datamodule_name": self.loaders_names[self.current_index]}

    def change_dataloader(self):
        choice = 0

        if self.num_datasets <= 1:
            self.current_index = choice
            self.current_iterator = self.iterators[self.current_index]
            return

        choice = [self.sampling_func()]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # This broadcast is probably unnecessary with lightning if everything
            # is already properly seeded. But,to be on safe side, we can still
            # do this.
            # There are also some smarter ways to do this to avoid any broadcasting
            # by basically having a fixed generator with a fixed seed which will
            # always work deterministically.
            # TODO: Check if not doing this provides any speed benefits.
            torch.distributed.broadcast_object_list(choice, 0)

        self.current_index = choice[0]
        self.current_iterator = self.iterators[self.current_index]

    def set_epoch(self, epoch: int):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for sampler in self.samplers:
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
