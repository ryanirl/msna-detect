import numpy as np
import math

import scipy.ndimage as ndi
from scipy.signal import butter
from scipy.signal import filtfilt

from typing import Iterable
from typing import Union
from typing import Tuple
from typing import List


def normalize_msna(msna: np.ndarray, fs: int = 250, axis: int = -1) -> np.ndarray:
    """
    Given an MSNA signal (integrated), we filter the signal first with the 
    butterworth bandpass filter to remove frequencies outside the range we 
    care about, and then normalize the signal into a common scale to aid in 
    peakfinding.

    Args:
        msna (np.ndarray): The MSNA signal to normalize.
        fs (int): The sampling rate of the signal.
        axis (int): The axis to apply the filter along.

    Returns:
        np.ndarray: The normalized MSNA signal.
    """
    msna = np.asarray(msna)
    msna = butter_filter(msna, fs = fs, cutoff_freq = [40.0], btype = "lowpass", axis = axis)
    msna = baseline_filter(msna, fs, axis = axis)
    msna = standardize_percentile(msna, 5, 50)
    msna = msna.clip(-1, 9.0) - 1
    return msna


def transform_bursts(bursts: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    """
    Generate the soft distributions for the bursts by convolving the sparse
    binary annotations with a symmetric Gaussian kernel.

    Args:
        bursts (np.ndarray): The sparse binary annotations of the bursts.
        sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
        np.ndarray: The soft distributions for the bursts.
    """
    bursts = np.asarray(bursts).astype(np.float32)
    bursts = ndi.gaussian_filter1d(bursts, sigma)

    max_val = bursts.max()
    if max_val > 0:
        bursts = bursts / max_val

    return bursts


def butter_filter(
    signal: np.ndarray, 
    fs: int = 200, 
    cutoff_freq: Iterable[float] = [0.10, 50], 
    order: int = 4, 
    axis: int = -1,
    btype: str = "bandpass"
) -> np.ndarray:
    """Applies a butterworth bandpass filter to an input signal.

    Args:
        signal (np.ndarray): The input signal to filter.
        fs (int): The sampling rate of the signal.
        cutoff_freq (Iterable[float]): The cutoff frequencies.
        order (int): The order of the filter.
        btype (str): The type of filter to apply.

    Returns:
        np.ndarray: The filtered signal.
    """
    cutoff = np.asarray(cutoff_freq)
    b, a = butter(
        N = order, 
        Wn = cutoff / (0.5 * fs), 
        btype = btype,
        analog = False
    )
    return filtfilt(b, a, signal, axis = axis)


def standardize_percentile(y: np.ndarray, l: int, r: int) -> np.ndarray:
    """Standardizes a signal (y) using the percentile method.

    Args:
        y (np.ndarray): The signal to standardize.
        l (int): The lower percentile.
        r (int): The upper percentile.

    Returns:
        np.ndarray: The standardized signal.
    """
    lp = np.percentile(y, l)
    rp = np.percentile(y, r)
    y = (y - lp) / (rp - lp)
    return y


def baseline_filter(signal: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Subtracts the baseline from the signal.

    Args:
        signal (np.ndarray): The signal to subtract the baseline from.
        *args: Additional arguments to pass to `_estimate_baseline`.
        **kwargs: Additional keyword arguments to pass to `_estimate_baseline`.

    Returns:
        np.ndarray: The signal with the baseline subtracted.
    """
    return signal - _estimate_baseline(signal, *args, **kwargs)


def _estimate_baseline(
    signal: np.ndarray, 
    size: int = 450,
    percentile: int = 25,
    *,
    mode: str = "reflect",
    order: int = 2,
    axis: int = -1,
    **padding_kwargs
) -> np.ndarray:
    """
    Estimates the baseline of the signal by binning the signal and then
    taking the percentile of the binned signal.

    Args:
        signal (np.ndarray): The signal to estimate the baseline of.
        size (int): The size of the bins.
        percentile (int): The percentile to take of the binned signal.
        mode (str): The mode to pad the signal with.
        axis (int): The axis to bin the signal along.
        **padding_kwargs: Additional keyword arguments to pass to `bin_array`.

    Returns:
        np.ndarray: The baseline of the signal.
    """
    ndim = signal.ndim
    axis = axis % ndim
    signal = signal.astype(np.float32)
    signal, padding = bin_array(
        signal, 
        bin_size = size, 
        axis = axis, 
        mode = mode, 
        return_padding = True, 
        **padding_kwargs
    )
    pad_l, pad_r = padding[axis]
    pad_r = -pad_r if (pad_r != 0) else None

    unpad_idx_ = [slice(None)] * ndim
    unpad_idx_[axis] = slice(pad_l, pad_r)
    unpad_idx = tuple(unpad_idx_)

    axes_ = [1] * ndim
    axes_[axis] = size
    axes = tuple(axes_)

    baseline = np.percentile(signal, percentile, axis = axis + 1)
    baseline = ndi.zoom(baseline, axes, mode = "nearest", order = order)
    baseline = baseline[unpad_idx]

    return baseline


def bin_array(
    array, 
    bin_size, 
    axis = -1, 
    pad_dir = "symmetric", 
    mode = "edge", 
    return_padding = False,
    **padding_kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    """Given an array and bin size, bins the array along an arbitrary axis into
    bins of size `bin_size`. It will perform padding if the array does not split
    up into equal bin sizes. 

    Args:
        array (np.ndarray): The input array.
        bin_size (int): The size of each bin.
        axis (int): The axis to bin the array along. Default is -1.
        pad_dir (str): The padding direction. One of `left`, `right`, or
            `symmetric` (default).
        return_padding (bool): Option to return the padding width used. 
        mode (str): The padding mode. See the NumPy documentation for options. 

    Returns:
        np.ndarray: The binned array where the number of bins is first. That is,
            the shape will be `(..., n_bins, bin_size, ...)`.
        padding (Tuple[np.ndarray, List]): The padding width used. Only returned
            if `return_padding` is `True`.
    """
    if axis == -1:
        axis = array.ndim - 1

    curr_len = array.shape[axis]
    n_bins = math.ceil(curr_len / bin_size)
    new_len = n_bins * bin_size

    new_shape = list(array.shape)
    new_shape[axis] = n_bins
    new_shape.insert(axis + 1, bin_size)

    # Perform padding if curr_len does not equal new_len.
    padding = [(0, 0)] * array.ndim
    if curr_len != new_len:
        if pad_dir == "left":
            pad_l = new_len - curr_len
            pad_r = 0
        elif pad_dir == "right":
            pad_l = 0
            pad_r = new_len - curr_len
        else:
            pad_l = (new_len - curr_len) // 2
            pad_r = (new_len - curr_len) - pad_l

        padding[axis] = (pad_l, pad_r)
        array = np.pad(array, padding, mode = mode, **padding_kwargs)

    array = array.reshape(new_shape)

    if return_padding:
        return array, padding

    return array


