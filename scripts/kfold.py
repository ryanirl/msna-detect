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


def main(
    filepath: str, 
    epochs: int = 50, 
    lr: float = 0.01, 
    batch_size: int = 16, 
    sampling_rate: int = 250,
    device: Optional[str] = None, 
    height_threshold: float = 0.3, 
    distance: int = 50,
    n_folds: int = 5
) -> None:
    """
    Perform k-fold cross validation for MSNA burst detection.
    """
    # Load the MSNA signal from the input file. This is a numpy array of shape (time,).
    signals, bursts = _load_msna(filepath)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create stratified k-fold splits
    k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 42).split(signals)

    models = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    for train_inds, test_inds in tqdm(k_fold, total = n_folds):
        x_train = [signals[i] for i in train_inds]
        y_train = [bursts[i] for i in train_inds]
        
        model = MsnaModel(model = "unet1d", sampling_rate = sampling_rate, device = device)
        model.fit(
            train_signal = x_train,
            train_bursts = y_train,
            lr = lr,
            batch_size = batch_size,
            epochs = epochs
        )
        models.append(model)

        for i in test_inds:
            x_test = signals[i].reshape(-1)
            y_true = bursts[i]
            
            # Predict the peaks using the model.
            y_pred = model.predict(x_test, height = height_threshold, distance = distance)
            
            # Compute the metrics.
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
        prog = "kfold",
        description = "Perform k-fold cross validation for MSNA burst detection.",
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
        "--sampling-rate", type = int, required = False, default = 250, metavar = "",
        help = "The sampling rate of the MSNA signal."
    )
    parser.add_argument(
        "--device", type = str, required = False, default = None, metavar = "",
        help = "The device to use for training."
    )
    parser.add_argument(
        "--height", type = float, required = False, default = 0.3, metavar = "",
        help = "The height threshold for peak detection."
    )
    parser.add_argument(
        "--distance", type = int, required = False, default = 50, metavar = "",
        help = "The minimum distance between detected peaks."
    )
    parser.add_argument(
        "--folds", type = int, required = False, default = 5, metavar = "",
        help = "The number of folds for cross validation."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        filepath = args.input,
        epochs = args.epochs,
        lr = args.lr,
        batch_size = args.batch_size,
        sampling_rate = args.sampling_rate,
        device = args.device,
        height_threshold = args.height,
        distance = args.distance,
        n_folds = args.folds
    )

