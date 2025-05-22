import numpy as np 

from typing import Tuple


def msna_metric(
    pred_peaks: np.ndarray, true_peaks: np.ndarray, window_size: int = 40
) -> Tuple[float, float, float]:
    """
    An F1 metric based on a custom ground-truth binning scheme for this task. 
    
    Args:
        pred_peaks (np.ndarray): An array of integers representing where each 
            MSNA peak was found.
        true_peaks (np.ndarray): An array of integers representing where each 
            ground truth MSNA peak is. This can be computed as `peaks_from_bool_1d(df["Burst"])`
        window_size (int): The size of the window around each true peak.
    
    Returns:
        float: The F1 score. 
    """
    pred, true = bin_predictions(pred_peaks, true_peaks, window_size)
    return scores(pred, true)


def bin_predictions(
    pred_peaks: np.ndarray, true_peaks: np.ndarray, window_size: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom binning scheme for the predicted and ground truth data. For each
    ground truth peak, we take some window around it and consider that as the
    true range. Any time-point that is not apart of this window is considered a
    false section. In this sense, this metric is NOT commutative and would be
    very sensitive to swapping the pred and true peaks. 

    Args:
        pred_peaks (np.ndarray): An array of integers representing where each 
            MSNA peak was found.
        true_peaks (np.ndarray): An array of integers representing where each 
            ground truth MSNA peak is. This can be computed as `peaks_from_bool_1d(df["Burst"])`
        window_size (int): The size of the window around each true peak.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: The pred and true binned data (in this order).
    """
    l = true_peaks - window_size
    r = true_peaks + window_size

    pred = []
    true = []
    for i in range(len(l)):
        # Check if there's at least one prediction in this window
        if np.sum((pred_peaks >= l[i]) & (pred_peaks < r[i])) >= 1: 
            pred.append(True)
            true.append(True)
        else:
            pred.append(False)
            true.append(True)
            
        # Check gap to next peak (only if not the last peak)
        if i < len(l) - 1:
            if np.sum((pred_peaks >= r[i]) & (pred_peaks < l[i+1])) >= 1: 
                pred.append(True)
                true.append(False)
            else:
                pred.append(False)
                true.append(False)
    
    return np.array(pred), np.array(true)


def scores(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float, float]:
    """An F1 metric based on boolean `pred` and `true` aligned data."""
    if np.array(pred.shape) != np.array(true.shape):
        raise ValueError("`pred` and `true` arrays must have the same shape.")

    tp, fp, tn, fn = confusion_matrix_values(pred, true)
    
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1, precision, recall
 

def confusion_matrix_values(
    pred: np.ndarray, true: np.ndarray
) -> Tuple[float, float, float, float]:
    """Get the true/false positive/negatives of the boolean data."""
    tp = np.sum((pred == 1) & (true == 1))
    fp = np.sum((pred == 1) & (true == 0))
    tn = np.sum((pred == 0) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))
    
    return tp, fp, tn, fn


def peaks_from_bool_1d(bool_array: np.ndarray) -> np.ndarray: 
    """Converts a boolean array to an array of integer indices."""
    return np.where(bool_array)[0]
    

    