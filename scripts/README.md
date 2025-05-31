# MSNA Detection Scripts

This directory contains command-line scripts for training, evaluating, and using the MSNA burst detection model. These scripts provide a convenient way to work with the MSNA detection library without writing Python code.

Before using these scripts, you need to install the MSNA detection library. Instructions for this can be found in the main `README.md` file for this library.


## Available Scripts

### Training Script (`train.py`)

Trains a new MSNA burst detection model on your data.

```bash
python train.py -i /path/to/training/data -o /path/to/save/model.pt [options]
```

Options:
- `-i, --input`: Path to directory containing training CSV files
- `-o, --output`: Path to save the trained model
- `--epochs`: Number of training epochs (default: 1000)
- `--lr`: Learning rate (default: 0.01)
- `--batch-size`: Batch size (default: 16)
- `--sampling-rate`: Sampling rate of MSNA signal in Hz (default: 250)
- `--device`: Device to use (default: auto-detect CUDA)
- `--pretrained`: Path to pretrained model for fine-tuning


### Prediction Script (`predict.py`)

Uses a trained model to detect bursts in MSNA signals.

```bash
python predict.py -i /path/to/input.csv -o /path/to/output.csv -m /path/to/model.pt [options]
```

Options:
- `-i, --input`: Path to input CSV file containing MSNA signal
- `-o, --output`: Path to save predictions
- `-m, --model`: Path to trained model file
- `--device`: Device to use (default: auto-detect CUDA)
- `--height`: Height threshold for peak detection (default: 0.3)
- `--distance`: Minimum distance between peaks (default: 50)


### Evaluation Script (`eval.py`)

Evaluates model performance on a test dataset.

```bash
python eval.py -i /path/to/test/data -m /path/to/model.pt [options]
```

Options:
- `-i, --input`: Path to test data directory containing CSV files
- `-m, --model`: Path to trained model file
- `--device`: Device to use (default: auto-detect CUDA)
- `--height`: Height threshold for peak detection (default: 0.3)
- `--distance`: Minimum distance between peaks (default: 50)


## Input Data Format

The scripts expect CSV files with the following columns:
- `Integrated MSNA`: The MSNA signal values
- `Burst`: Binary annotations of true bursts (1 for burst, 0 for no burst)

For prediction output, additional columns are added:
- `Predicted Burst`: Binary predictions of bursts
- `Predicted Probability`: Probability scores for each prediction


## Example Usage

Train a new model:

```bash
python train.py -i data/training -o models/my_model.pt --epochs 500 --sampling-rate 250
```

<br>

Evaluate model performance:

```bash
python eval.py -i data/test -m models/my_model.pt
```

<br>

Make predictions:

```bash
python predict.py -i data/test_signal.csv -o predictions.csv -m models/my_model.pt
```

