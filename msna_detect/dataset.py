import numpy as np
import random 

import torch
from torch.utils.data import Dataset

from typing import Callable
from typing import Optional
from typing import Tuple
from typing import List


class MsnaRandomSliceDataset(Dataset):
    def __init__(
        self, 
        signals: List[np.ndarray], 
        bursts: List[np.ndarray], 
        transforms: Optional[Callable] = None,
        batch_size: Optional[int] = None, 
        window_size: int = 2048 * 4
    ) -> None:
        """
        A fast dataset for stochastic training when you have a small number of
        independent long signals. Random selection of signals, and windows of
        each signal are done during indexing, hence the batch size can also be
        set dynamically. For these reasons, this dataset should only be used
        during training, and not inference.

        Args:
            signals (List[np.ndarray]): A list of the signal data, each of shape
                (channels, time). 
            bursts (List[np.ndarray]): A list of the ground truth labels, each
                of shape (channels, time).
            batch_size (Optional[int]): Because we assume that there are a few 
                number of signals, we set the length of this dataset to be the
                batch size and then randomly select from the signals during
                indexing. 
            window_size (int): The size of each window to train from during 
                training.
        """
        # Ensure that the signals are a list of float32 numpy arrays.
        self.signals = [np.atleast_2d(np.array(s, dtype = np.float32)) for s in signals]
        self.bursts = [np.atleast_2d(np.array(b, dtype = np.float32)) for b in bursts]

        self.transforms = transforms
        self.batch_size = batch_size if batch_size is not None else len(self.signals)
        self.window_size = window_size

        if len(self.signals) != len(self.bursts):
            raise ValueError("`signals` and `bursts` must have the same length.")

        self._n = len(signals)
        
    def __len__(self) -> int:
        return self.batch_size
    
    def __getitem__(self, _: int) -> Tuple[np.ndarray, np.ndarray]:
        i = random.randint(0, self._n-1)
        x = self.signals[i]
        y = self.bursts[i]
        
        # Get a random window of size `self.window_size`.
        start = random.randint(0, x.shape[1] - self.window_size - 1)
        end = start + self.window_size
        
        x = x[:, start:end]
        y = y[:, start:end]
        if self.transforms is not None:
            x, y = self.transforms(x, y)
        
        return x, y


