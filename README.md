# MSNA Burst Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/msna-detect.svg)](https://badge.fury.io/py/msna-detect)

A deep learning framework for automated detection of bursts in Muscle Sympathetic Nerve Activity (MSNA) signals.


## Overview

Muscle Sympathetic Nerve Activity (MSNA) provides direct measurement of sympathetic outflow to skeletal muscle vasculature, offering insight into neural mechanisms governing cardiovascular control. This repository presents a robust and efficient deep learning framework for automated detection of bursts in MSNA signals, addressing a challenging task that traditionally requires time-consuming manual annotation by experts. Our approach utilizes a 1D convolutional neural network based on U-Net architecture that reformulates burst detection by modeling probabilistic distributions around burst points. The framework exploits CNN translation invariance with random window sampling for efficient training, and can process recordings of arbitrary length without boundary artifacts during inference. 

This library provides a simple API for both training and inference, along with pre-trained models for immediate application.


## Installation

You can install the package from PyPI:

```bash
pip install msna-detect
```

Or you can install from source:

```bash
# Clone the repository
git clone https://github.com/ryanirl/msna-detect.git
cd msna-detect

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

See `requirements.txt` for the full list of dependencies.


## Quick Start

### Inference with Pre-trained Model

```python
from msna_detect import MsnaModel

# Load your MSNA signal (should be shape [channels, time] or [time])
signal = ...

# Load the pre-trained model
model = MsnaModel.from_pretrained("pretrained/model.pth")

# Get burst probabilities
burst_probs = model.predict(signal)

# Find burst peaks
burst_locations = model.find_peaks(signal, height = 0.4, distance = 100)

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
model.save("my_trained_model.pth")
```

For more advanced usage examples, see the `examples/` directory.


## Architecture

Our approach utilizes a 1D convolutional neural network based on the U-Net architecture:

1. **Input**: Normalized MSNA integrated signal
2. **Processing**: Seven downsampling stages with ResNet-like encoder blocks followed by corresponding upsampling stages with skip connections
3. **Output**: Probability map where peaks correspond to detected bursts

The model is trained by:
1. Transforming sparse binary annotations into soft distributions using Gaussian convolution
2. Random window sampling for computational efficiency
3. Optimizing with mean squared error loss
4. Post-processing with calibration to ensure consistent probability scaling


## Dataset Format

The model expects MSNA data as NumPy arrays. For training, both signals and burst annotations should be provided:

```python
# Example format for training data
signals = [signal1, signal2, ...]  # List of numpy arrays, each of shape [channels, time]
bursts = [burst1, burst2, ...]     # List of numpy arrays with binary burst annotations
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{msna2025,
  title={Muscle Sympathetic Nerve Activity Burst Detection},
  author={Peters, Ryan and [Other Authors]},
  journal={[Journal Name]},
  year={2025},
  volume={},
  pages={}
}
```


## Contact

For questions or comments, please contact ryanirl@icloud.com.



