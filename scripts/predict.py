import pandas as pd
import numpy as np
import argparse
import torch

from typing import Optional

from msna_detect import MsnaModel


def predict(
    filepath: str, 
    output_path: str, 
    model_path: str, 
    height_threshold: float = 0.3, 
    distance: int = 50,
    device: Optional[str] = None,
    force_download: bool = False,
    quiet: bool = False
) -> None:
    # Load the MSNA signal from the input file. This is a numpy array of shape (time,).
    df = _load_msna(filepath)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pretrained MSNA burst detection model.
    model = MsnaModel.from_pretrained(model_path, force_download = force_download, quiet = quiet)
    model.to(device)

    # Get the burst probabilities. This is also a numpy array of shape (time,).
    # Alternatively, you can use the `predict` method to directly get the burst times.
    burst_probabilities = model.predict_proba(df["Integrated MSNA"].to_numpy())

    # Now perform peak-finding to get the burst times
    burst_times = model.find_peaks(
        burst_probabilities, height = height_threshold, distance = distance)

    # Add the predicted bursts to the dataframe.
    burst_bool = np.zeros(len(df), dtype = bool)
    burst_bool[burst_times] = True
    df["Predicted Burst"] = burst_bool
    df["Predicted Probability"] = burst_probabilities

    # Write the dataframe to a csv file.
    _write_predicted_bursts(df, output_path)


def _load_msna(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def _write_predicted_bursts(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index = False)


def parse_args():
    parser = argparse.ArgumentParser(
        prog = "predict",
        description = "Predict burst times in MSNA signals.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input", type = str, metavar = "",
        help = "The path to the input MSNA signal."
    )
    parser.add_argument(
        "-o", "--output", type = str, metavar = "",
        help = "The path to save predictions."
    )
    parser.add_argument(
        "-m", "--model", type = str, metavar = "",
        help = "The path to the trained model file or name of pretrained model (e.g., 'msna-v1')."
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
    parser.add_argument(
        "--force-download", action = "store_true",
        help = "Force re-download of pretrained models even if they exist locally."
    )
    parser.add_argument(
        "--quiet", action = "store_true",
        help = "Suppress download messages."
    )
    parser.add_argument(
        "--list-models", action = "store_true",
        help = "List available pretrained models and exit."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Handle special commands
    if args.list_models:
        MsnaModel.list_pretrained()
        return
    
    # Check required arguments when not listing models
    if not args.input:
        print("Error: -i/--input is required")
        return
    if not args.output:
        print("Error: -o/--output is required")
        return
    if not args.model:
        print("Error: -m/--model is required")
        return
    
    predict(
        filepath = args.input,
        output_path = args.output,
        model_path = args.model,
        device = args.device,
        height_threshold = args.height,
        distance = args.distance,
        force_download = args.force_download,
        quiet = args.quiet
    )


if __name__ == "__main__":
    main()