import pandas as pd
import numpy as np
import argparse
import torch
import glob
import os

from typing import Optional
from typing import Tuple
from typing import List

from msna_detect import MsnaModel
from msna_detect.metrics import msna_metric
from msna_detect.metrics import peaks_from_bool_1d


def eval(filepath: str, model_path: str, height_threshold: float = 0.3, distance: int = 50, device: Optional[str] = None) -> None:
    # Load the MSNA signal from the input file. This is a numpy array of shape (time,).
    signals, bursts = _load_msna(filepath)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pretrained MSNA burst detection model.
    model = MsnaModel.from_pretrained(model_path)
    model.to(device)

    # Get the burst probabilities. This is also a numpy array of shape (time,).
    f1_scores = []
    precision_scores = []
    recall_scores = []
    for signal, burst in zip(signals, bursts):
        burst_probabilities = model.predict_proba(signal)

        # Now perform peak-finding to get the burst times
        burst_times = model.find_peaks(
            burst_probabilities, height = height_threshold, distance = distance)

        # Compute the F1 score
        f1_score, precision, recall = msna_metric(burst_times, peaks_from_bool_1d(burst))
        f1_scores.append(f1_score)
        precision_scores.append(precision)
        recall_scores.append(recall)
        print(f"F1 score: {f1_score:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")

    # Compute the mean F1 score
    mean_f1_score = np.mean(f1_scores)
    print(f"\nMean F1 score: {mean_f1_score}")

    mean_precision = np.mean(precision_scores)
    print(f"Mean precision: {mean_precision}")

    mean_recall = np.mean(recall_scores)
    print(f"Mean recall: {mean_recall}")


def _load_msna(path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load the MSNA signals and annotator bursts from the input folder"""
    if not os.path.isdir(path):
        raise ValueError(f"The input path {path} is not a directory.")
    
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
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
        prog = "eval",
        description = "Evaluate MSNA burst detection model performance.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input", type = str, required = True, metavar = "",
        help = "The path to the input folder containing .csv files."
    )
    parser.add_argument(
        "-m", "--model", type = str, required = True, metavar = "",
        help = "The path to the trained model file."
    )
    parser.add_argument(
        "--device", type = str, required = False, default = "cpu", metavar = "",
        help = "The device to run the model on."
    )
    parser.add_argument(
        "--height", type = float, required = False, default = 0.3, metavar = "",
        help = "The height threshold for peak detection."
    )
    parser.add_argument(
        "--distance", type = int, required = False, default = 50, metavar = "",
        help = "The minimum distance between detected peaks."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    eval(
        filepath = args.input,
        model_path = args.model,
        device = args.device,
        height_threshold = args.height,
        distance = args.distance
    )


if __name__ == "__main__":
    main()

