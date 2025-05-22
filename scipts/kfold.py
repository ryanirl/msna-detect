from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import argparse
import glob
import torch
import os

from sklearn.model_selection import KFold

from typing import Optional
from typing import Tuple
from typing import List

from msna_detect import MsnaModel
from msna_detect.metrics import msna_metric
from msna_detect.metrics import peaks_from_bool_1d

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

    # Create stratified k-fold splits
    k_fold = KFold(n_splits = 5, shuffle = True, random_state = 42).split(train_signal)

    models = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    for train_inds, test_inds in tqdm(k_fold, total = 5):
        x_train = [train_signal[i] for i in train_inds]
        y_train = [train_bursts[i] for i in train_inds]
        x_valid = [train_signal[i] for i in test_inds]
        y_valid = [train_bursts[i] for i in test_inds]
        
        model = MsnaModel(model = "unet1d", device = device)
        model.fit(
            train_signal = x_train,
            train_bursts = y_train,
            valid_signal = x_valid,
            valid_bursts = y_valid,
            lr = lr,
            batch_size = batch_size,
            epochs = epochs
        )
        models.append(model)

        for i in test_inds:
            x_test = train_signal[i].reshape(-1)
            y_true = train_bursts[i]
            
            y_pred = model(x_test)
            y_pred = model.find_peaks(y_pred, height = BURST_HEIGHT_THRESHOLD, distance = BURST_DISTANCE)
            
            f1, precision, recall = msna_metric(y_pred, peaks_from_bool_1d(y_true))
            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
            
            print(f"F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")

    print("\nF1 Score:")
    print("Mean", "|", np.mean(f1_scores))
    print(" Std", "|", np.std(f1_scores))
    print(" Max", "|", np.max(f1_scores))
    print(" Min", "|", np.min(f1_scores))

    print("\nPrecision Score:")
    print("Mean", "|", np.mean(precision_scores))
    print(" Std", "|", np.std(precision_scores))
    print(" Max", "|", np.max(precision_scores))
    print(" Min", "|", np.min(precision_scores))

    print("\nRecall Score:")
    print("Mean", "|", np.mean(recall_scores))
    print(" Std", "|", np.std(recall_scores))
    print(" Max", "|", np.max(recall_scores))
    print(" Min", "|", np.min(recall_scores))


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

