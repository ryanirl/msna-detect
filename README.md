# MSNA Burst Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/msna-detect.svg)](https://badge.fury.io/py/msna-detect)

A deep learning framework for automated detection of bursts in Muscle Sympathetic Nerve Activity (MSNA) signals.


## Overview

Muscle Sympathetic Nerve Activity (MSNA) provides direct measurement of
sympathetic outflow to skeletal muscle vasculature, offering insight into neural
mechanisms governing cardiovascular control. This repository presents a robust
and efficient deep learning framework for automated detection of bursts in MSNA
signals, addressing a challenging task that traditionally requires
time-consuming manual annotation by experts. Our approach utilizes a 1D
convolutional neural network based on U-Net architecture that reformulates burst
detection by modeling probabilistic distributions around burst points. The
framework exploits CNN translation invariance with random window sampling for
efficient training, but then can process recordings of arbitrary length (without 
windowing) during inference to exploit the full receptive field of our models and
avoid unnecessary boundary artifacts that are typically introduced when windowing. 
This enables models to be trained (from scratch) on most laptop CPUs in under 4 minutes.
A diagram of the model can be found in `img/` folder.

This library provides a simple API for both training and inference, along with
pre-trained models for immediate application.


## Installation

You can install the package from PyPI:

```bash
pip install msna-detect
```

See `requirements.txt` for the full list of dependencies.



## Quick Start

We provide one pretrained model, that was trained on synthetic dataset generated
from our other `msna-sim` library. A sample dataset can be download [here](https://drive.google.com/file/d/1nuayGCYgfn1Ke0xytR7krwXOLnAPiVjn/view?usp=sharing). The fasted way to get started
with using the tools in this library, would be to download the sample data and run inference on it with our pretrained model.

```bash
pip install msna-detect

# Make sure you download the sample data first. 
msna-detect-predict -i sample-data.csv -o predictions.csv -m msna-v1

# Visualize the predictions against the ground-truth using the dashboard. 
msna-detect-dashboard -i predictions.csv
```

Below, we provide more details about how to use the command-line interface, and
the codebase itself.


## CLI 

This directory contains command-line scripts for training, evaluating, and using
the MSNA burst detection model. These scripts provide a convenient way to work
with the MSNA detection library without writing Python code.

Before using these scripts, you need to install the MSNA detection library. Furthermore,
the scripts expect CSV files with the following columns:
- `Integrated MSNA`: The MSNA signal values
- `Burst`: Binary annotations of true bursts (1 for burst, 0 for no burst)


### Training Script

Trains a new MSNA burst detection model on your data.

```bash
msna-detect-train -i /path/to/training/data -o /path/to/save/model.pt [options]
```

### Prediction

Uses a trained model to detect bursts in MSNA signals.

```bash
msna-detect-predict -i /path/to/input.csv -o /path/to/output.csv -m /path/to/model.pt [options]
```

### Evaluation

Evaluates model performance on a test dataset.

```bash
msna-detect-eval -i /path/to/test/data -m /path/to/model.pt [options]
```

All scripts can be run with the `-h` of `--help` tag to get the additional options.


## Codebase Examples

### Inference with Pre-trained Model

```python
from msna_detect import MsnaModel

# Load your MSNA signal (should be shape (time,))
signal = ...

# Load the pre-trained model
model = MsnaModel.from_pretrained("msna-v1")

# Get burst probabilities
burst_probs = model.predict_proba(signal)

# Find burst peaks
burst_locations = model.find_peaks(signal, height = 0.3, distance = 50)

print(f"Found {len(burst_locations)} bursts")
```

### Training a New Model

```python
from msna_detect import MsnaModel

# Prepare your training data
signals = [signal1, signal2, ...]  # List of numpy arrays, each of shape [channels, time]
bursts = [burst1, burst2, ...]     # List of numpy arrays with binary burst annotations

# Create and train a model
model = MsnaModel(sampling_rate = 250, device = "cuda")
model.fit(
    train_signal = signals,
    train_bursts = bursts,
    epochs = 50,
    lr = 0.01,
    batch_size = 32
)

# Save the trained model
model.save("my_trained_model.pt")
```

For more advanced usage examples, see the `examples/` directory.


### Dataset Format

The model expects MSNA data as NumPy arrays. For training, both signals and burst annotations should be provided:

```python
# Example format for training data
signals = [signal1, signal2, ...]  # List of numpy arrays, each of shape (time,)
bursts = [burst1, burst2, ...]     # List of numpy arrays with binary burst annotations
```

## Visualization Dashboard


<p align="center">
 <img src="./img/dashboard.png" width="98%">
</p>


The package includes an interactive visualization dashboard built with Bokeh
that allows you to explore MSNA signals and their detected bursts. The dashboard
provides:

- Interactive visualization of the integrated MSNA signal
- Overlay of true and predicted burst locations
- Probability distribution of burst predictions
- Navigation tools for exploring long recordings
- Hover tools for detailed signal inspection

To use the dashboard you can run the dashboard on a CSV file containing MSNA data

```bash
msna-detect-dashboard -i path/to/your/data.csv
```

You can also export the dashboard as an HTML file. This can be sent to people without them having to install anything. 

```bash
msna-detect-dashboard -i path/to/your/data.csv --save
```

For the dashboard, the input CSV file should contain the following columns:
- `Integrated MSNA`: The normalized MSNA signal
- `Burst`: Binary annotations of true bursts
- `Predicted Burst`: Binary annotations of predicted bursts
- `Predicted Probability`: Probability scores for burst predictions

A file like this can be generated from the `msna-detect-predict` command-line tool.


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{msna2025,
  title={Muscle Sympathetic Nerve Activity Burst Detection},
  author={Peters, Ryan},
  year={2025}
}
```


