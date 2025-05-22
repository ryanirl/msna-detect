import pandas as pd
import numpy as np
import argparse
import glob
import torch
import os

from typing import Optional
from typing import Tuple
from typing import List

from msna_detect import MsnaModel

# Data parameters
SAMPLING_RATE = 250

# Hyperparameters
BURST_HEIGHT_THRESHOLD = 0.5
BURST_DISTANCE = 100


def main(filepath: str, output_path: str, epochs: int = 50, lr: float = 0.01, batch_size: int = 16, device: Optional[str] = None) -> None:
    # Load the MSNA signal from the input file. This is a numpy array of shape (time,).
    train_signal, train_bursts = _load_msna(filepath)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pretrained MSNA burst detection model.
    model = MsnaModel(sampling_rate = SAMPLING_RATE, device = device)
    model.fit(
        train_signal = train_signal,
        train_bursts = train_bursts,
        epochs = epochs,
        lr = lr,
        batch_size = batch_size
    )

    # Save the model
    model.save(output_path)


def _load_msna(path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load the MSNA signals and annotator bursts from the input folder"""
    if not os.path.isdir(path):
        raise ValueError(f"The input path {path} is not a directory.")
    
    files = glob.glob(os.path.join(path, "*.csv"))
    if len(files) == 0:
        raise ValueError(f"No files found in the input path {path}.")
    
    signals = []
    bursts = []
    for file in files:
        df = pd.read_csv(file)

        signals.append(df["Integrated MSNA"].to_numpy())
        bursts.append(df["Burst"].to_numpy())

    return signals, bursts


def parse_args():
    parser = argparse.ArgumentParser(
        prog = "train", description = "Train a MSNA burst detection model.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input", type = str, required = True, metavar = "",
        help = "The path to the input folder containing the MSNA signals."
    )
    parser.add_argument(
        "-o", "--output", type = str, required = True, metavar = "",
        help = "The path to save the model to."
    )
    parser.add_argument(
        "--epochs", type = int, required = False, default = 1000, metavar = "",
        help = "The number of epochs to train for."
    )
    parser.add_argument(
        "--lr", type = float, required = False, default = 0.01, metavar = "",
        help = "The learning rate to use."
    )
    parser.add_argument(
        "--batch-size", type = int, required = False, default = 16, metavar = "",
        help = "The batch size to use."
    )
    parser.add_argument(
        "--device", type = str, required = False, default = None, metavar = "",
        help = "The device to use for training."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        filepath = args.input,
        output_path = args.output,
        epochs = args.epochs,
        lr = args.lr,
        batch_size = args.batch_size,
        device = args.device
    )

