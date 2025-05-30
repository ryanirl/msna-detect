from __future__ import annotations

import numpy as np
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scipy.signal import find_peaks as _find_peaks

from msna_detect.models import get_model
from msna_detect.trainer import train as torch_train
from msna_detect.registry import get_registry
from msna_detect.dataset import MsnaRandomSliceDataset
from msna_detect.utils.ndtiler import dynamic_tile_nd
from msna_detect.filters import normalize_msna
from msna_detect.filters import transform_bursts

from typing import Callable
from typing import Optional
from typing import Union
from typing import Dict
from typing import List
from typing import Any

t_OPTIMIZER = torch.optim.Optimizer
t_CRITERION = torch.nn.modules.loss._Loss


class MsnaModel:
    def __init__(
        self, model: Union[str, nn.Module] = "unet1d", sampling_rate: int = 250, device: str = "cpu"
    ) -> None:
        """
        The main user-facing model with support for training and inference. 

        Args:   
            model (str): The backbone model to use.
            device (str): The device (ex: cuda, cpu) to use.
        """
        self.model = model if isinstance(model, nn.Module) else get_model(model)
        self.to(device)

        self.sampling_rate = sampling_rate
        self.model_name = model

        # The training results is a list to accommodate for pretraining, while
        # maintaining the previous runs history.
        self.training_results: List[Dict[str, Any]] = []

        self._alpha = 1.0
        self._is_fit = False

    @classmethod
    def from_pretrained(cls, path: str) -> MsnaModel:
        """Load a pretrained model from a file."""
        state_dict = torch.load(path)
        model = cls(state_dict["model_name"], state_dict["sampling_rate"])
        model.model.load_state_dict(state_dict["model_state_dict"])
        model.training_results = state_dict["training_results"]
        model._alpha = state_dict["alpha"]
        model._is_fit = True
        return model

    def state_dict(self) -> Dict[str, Any]:
        """Get the state dict of the model."""
        return {
            "model_state_dict": self.model.state_dict(),
            "training_results": self.training_results,
            "model_name": self.model_name,
            "sampling_rate": self.sampling_rate,
            "alpha": self._alpha
        }

    def save(self, path: str) -> None:
        """Write weights to an output file."""
        prev_device = self.device
        self.to("cpu")
        torch.save(self.state_dict(), path)
        self.to(prev_device)

    def to(self, device: str) -> MsnaModel:
        """Move the model to a different device."""
        self.model.to(device)
        self.device = device
        return self

    def fit(
        self,
        train_signal: List[np.ndarray],
        train_bursts: List[np.ndarray],
        valid_signal: Optional[List[np.ndarray]] = None,
        valid_bursts: Optional[List[np.ndarray]] = None,
        criterion: Union[str, t_CRITERION] = "MSELoss",
        optimizer: Union[str, t_OPTIMIZER] = "Adam",
        transforms: Optional[Callable] = None,
        sigma: float = 15.0,
        epochs: int = 500,
        lr: float = 0.01,
        batch_size: int = 32,
        drop_last: bool = True,
        num_workers: int = 0,
        check_val_every_n_epochs: int = 1,
        verbose: bool = True
    ) -> MsnaModel:
        """ 
        Train the model on the given data. 
        
        There are two parts to training a model like this. First, we need to
        train the base model to predict the burst probabilities. Second, we need
        to calibrate the base model to output probabilities in the proper range. 

        Args:
            train_signal (List[np.ndarray]): The training signal data.
            train_bursts (List[np.ndarray]): The training burst data.
            valid_signal (Optional[List[np.ndarray]]): The validation signal data.
            valid_bursts (Optional[List[np.ndarray]]): The validation burst data.
            criterion (str): The loss function to use.
            optimizer (str): The optimizer to use.
            transforms (Optional[Callable]): The transforms to apply to the data.
            sigma (float): The sigma value to use for the Gaussian filter.
            epochs (int): The number of epochs to train for.
            lr (float): The learning rate.
            batch_size (int): The batch size.
            drop_last (bool): Whether to drop the last batch.
            num_workers (int): The number of workers to use.
            check_val_every_n_epochs (int): The number of epochs to check the validation set.

        Returns:
            MsnaModel: The trained model.
        """
        # Resolve the criterion
        if isinstance(criterion, str):
            _criterion: t_CRITERION = get_registry("criterion")[criterion]()
        else:
            _criterion: t_CRITERION = criterion

        # Resolve the optimizer.
        if isinstance(optimizer, str):
            _optimizer: t_OPTIMIZER = get_registry("optimizer")[optimizer](
                params = self.model.parameters(), lr = lr
            )
        else:
            _optimizer: t_OPTIMIZER = optimizer

        # Normalize the training and validation signals and bursts.
        train_signal = [normalize_msna(ts, self.sampling_rate) for ts in train_signal]
        if valid_signal is not None:
            valid_signal = [normalize_msna(ts, self.sampling_rate) for ts in valid_signal]

        # Generate the soft distributions for the training and validation bursts.
        train_bursts = [transform_bursts(ts, sigma) for ts in train_bursts]
        if valid_bursts is not None:
            valid_bursts = [transform_bursts(ts, sigma) for ts in valid_bursts]

        # Create the dataloaders.
        train_dataloader = DataLoader(
            dataset = MsnaRandomSliceDataset(
                signals = train_signal, 
                bursts = train_bursts, 
                transforms = transforms,
                batch_size = batch_size
            ),
            batch_size = batch_size,
            drop_last = drop_last,
            num_workers = num_workers
        )
        valid_dataloader = None
        if valid_signal is not None and valid_bursts is not None:
            valid_dataloader = DataLoader(
                dataset = MsnaRandomSliceDataset(
                    signals = valid_signal, 
                    bursts = valid_bursts, 
                    transforms = transforms,
                    batch_size = batch_size
                ),
                batch_size = batch_size,
                drop_last = drop_last,
                num_workers = num_workers
            )

        # Train the model.
        results = torch_train(
            model = self.model,
            criterion = _criterion,
            optimizer = _optimizer,
            train_dataloader = train_dataloader,
            val_dataloader = valid_dataloader,
            #val_signals = valid_signal,
            #val_bursts = valid_bursts,
            device = self.device,
            min_epochs = 1,
            max_epochs = epochs,
            check_val_every_n_epochs = check_val_every_n_epochs,
            verbose = verbose
        )
        self._is_fit = True

        # Get the alpha value for the model
        self._alpha = np.mean([
            np.percentile(self._predict(ts), 99) for ts in train_signal])

        self.training_results.append(results)

        return self

    def re_calibrate(self, signal: np.ndarray) -> MsnaModel:
        """Re-calibrate the model on a new signal."""
        self._alpha = np.percentile(self.predict_proba(signal), 99)
        return self

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Perform inference on a new signal.

        Args:
            signal (np.ndarray): The signal to perform inference on of shape (b, c, t) or (c, t).
        
        Returns:
            np.ndarray: The predicted signal of shape (b, c, t) or (c, t).
        """
        return self.predict_proba(signal)

    @torch.no_grad()
    def _predict(self, signal: np.ndarray, window_size: Optional[int] = None, overlap: int = 128) -> np.ndarray:
        """
        Perform inference on a new signal.

        Args:
            signal (np.ndarray): The signal to perform inference on of shape (b, c, t) or (c, t).
            window_size (Optional[int]): The window size to use for inference. Defaults to the minimum signal length.
            overlap (int): The window overlap to use for inference.

        Returns:
            np.ndarray: The predicted signal of shape (b, c, t) or (c, t).
        """
        if not self._is_fit:
            warnings.warn("Calling `predict()` before training the `MsnaModel`.")

        # Normalize the signal.
        signal = normalize_msna(signal, self.sampling_rate)

        torch_signal = torch.tensor(
            signal, dtype = torch.float, device = self.device
        )
        if torch_signal.ndim == 1: torch_signal = torch_signal.unsqueeze(0)
        if torch_signal.ndim == 2: torch_signal = torch_signal.unsqueeze(0)

        self.model.eval()
        torch_pred = _windowed_inference(torch_signal, self.model, window_size, overlap)
        pred = torch_pred.cpu().numpy().reshape(signal.shape)

        return pred 

    def predict_proba(self, signal: np.ndarray, window_size: Optional[int] = None, overlap: int = 128) -> np.ndarray:
        """
        Perform inference on a new signal and return the predicted probabilities.

        Args:
            signal (np.ndarray): The signal to perform inference on of shape (b, c, t) or (c, t).
            window_size (Optional[int]): The window size to use for inference. Defaults to the minimum signal length.
            overlap (int): The window overlap to use for inference.

        Returns:
            np.ndarray: The predicted signal of shape (b, c, t) or (c, t).
        """
        # Perform inference on the signal.
        pred = self._predict(signal, window_size, overlap)

        # Calibration of the output probabilities.
        pred = np.clip(pred / (self._alpha + 1e-8), 0.0, 1.0)

        return pred

    def find_peaks(self, signal: np.ndarray, height: float = 0.4, distance: int = 50) -> np.ndarray:
        """
        Find the peaks in the signal.
        """
        return _find_peaks(signal, height = height, distance = distance)[0]

    def predict(self, signal: np.ndarray, window_size: Optional[int] = None, overlap: int = 128, height: float = 0.4, distance: int = 50) -> np.ndarray:
        """
        Perform inference on a new signal and return the predicted peaks.
        """
        peaks = self.find_peaks(
            self.predict_proba(signal, window_size, overlap), 
            height = height, distance = distance
        )
        return peaks 


def _windowed_inference(signal: torch.Tensor, model: nn.Module, window_size: Optional[int] = None, overlap: int = 128) -> torch.Tensor:
    """
    Perform inference on a signal using a sliding window.

    Args:
        signal (torch.Tensor): The signal to perform inference on of shape (b, c, t) or (c, t).
        model (nn.Module): The model to use for inference.
        window_size (Optional[int]): The window size to use for inference.
        overlap (int): The window overlap to use for inference.

    Returns:
        torch.Tensor: The predicted signal of shape (b, c, t) or (c, t).
    """
    b, c, l = signal.shape

    # If the signal is shorter than the minimum signal length given by the model, raise an error.
    if l < model.downsample_factor:
        raise ValueError(f"Signal is too short. Minimum signal length is {model.downsample_factor}.")

    # If the signal is shorter than the window size, set to base case of no windowing.
    if (window_size is None) or (l < window_size):
        return model(signal)

    # Else, perform windowed inference.
    div = torch.zeros((b, c, l), dtype = torch.float, device = signal.device)
    pred = torch.zeros((b, c, l), dtype = torch.float, device = signal.device)
    for ((x0, x1),) in dynamic_tile_nd((l,), (window_size,), (overlap,)):
        pred[:, :, x0:x1] += model(signal[:, :, x0:x1])
        div[:, :, x0:x1] += 1.0

    return pred / div


