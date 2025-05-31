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


def train(
    filepath: str, 
    output_path: str, 
    pretrained_path: Optional[str] = None, 
    epochs: int = 50, 
    lr: float = 0.01, 
    batch_size: int = 16, 
    sampling_rate: int = 250,
    device: Optional[str] = None
) -> None:
    # Load the MSNA signal from the input file. This is a numpy array of shape (time,).
    train_signal, train_bursts = _load_msna(filepath)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pretrained MSNA burst detection model.
    if pretrained_path is not None:
        model = MsnaModel.from_pretrained(pretrained_path)
    else:
        model = MsnaModel(sampling_rate = sampling_rate)

    print(f"Model:\n{model.model}")
    print(f"\nParmaters: {sum(p.numel() for p in model.model.parameters())}")

    model.to(device)
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
        prog = "train", 
        description = "Train a MSNA burst detection model.",
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
        "--sampling-rate", type = int, required = False, default = 250, metavar = "",
        help = "The sampling rate of the MSNA signal."
    )
    parser.add_argument(
        "--device", type = str, required = False, default = None, metavar = "",
        help = "The device to use for training."
    )
    parser.add_argument(
        "--pretrained", type = str, required = False, default = None, metavar = "",
        help = "The path to the pretrained model to use."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train(
        filepath = args.input,
        output_path = args.output,
        pretrained_path = args.pretrained,
        epochs = args.epochs,
        lr = args.lr,
        batch_size = args.batch_size,
        sampling_rate = args.sampling_rate,
        device = args.device
    )


if __name__ == "__main__":
    main()

