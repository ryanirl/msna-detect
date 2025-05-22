import pandas as pd
import numpy as np
import argparse

from msna_detect import MsnaModel

PRETRAINED_MODEL_PATH = "model.pt"

# Data parameters
SAMPLING_RATE = 250

# Hyperparameters
BURST_HEIGHT_THRESHOLD = 0.5
BURST_DISTANCE = 100


def main(filepath: str, output_path: str, device: str) -> None:
    # Load the MSNA signal from the input file. This is a numpy array of shape (time,).
    signal = _load_msna(filepath)

    # Load the pretrained MSNA burst detection model.
    model = MsnaModel.from_pretrained(PRETRAINED_MODEL_PATH)
    model.to(device)

    # Get the burst probabilities. This is also a numpy array of shape (time,).
    burst_probabilities = model.predict(signal)

    # Now perform peak-finding to get the burst times
    burst_times = model.find_peaks(
        burst_probabilities, height = BURST_HEIGHT_THRESHOLD, distance = BURST_DISTANCE)

    _write_predicted_bursts(burst_times, output_path)


def _load_msna(filepath: str) -> np.ndarray:
    return pd.read_csv(filepath)["Integrated MSNA"].to_numpy()


def _write_predicted_bursts(burst_times: np.ndarray, output_path: str) -> None:
    pd.DataFrame(burst_times, columns = ["Burst"]).to_csv(output_path, index = False)


def parse_args():
    parser = argparse.ArgumentParser(
        prog = "predict",
        description = "Predict the burst times of a MSNA signal."
    )
    parser.add_argument(
        "-i", "--input", type = str, required = True, metavar = "",
        help = "The path to the input MSNA signal."
    )
    parser.add_argument(
        "-o", "--output", type = str, required = True, metavar = "",
        help = "The path to the output file."
    )
    parser.add_argument(
        "--device", type = str, required = False, default = "cpu", metavar = "", 
        help = "The device to run the model on."
    ) 
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.input, args.output, args.device)

