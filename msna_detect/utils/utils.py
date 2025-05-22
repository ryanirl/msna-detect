import numpy as np


def remap_burst(signal: np.ndarray, burst: np.ndarray, window: int = 40) -> np.ndarray:
    """
    A utility function to remap the burst time to the peak of the burst. This is useful
    for the human-annotated data, where the burst times are often not aligned perfectly 
    with the peak of the burst.

    Args:
        signal (np.ndarray): The signal to remap the burst on.
        burst (np.ndarray): The burst to remap.
        window (int): The window to use to find the peak.

    Returns:
        np.ndarray: The remapped burst.
    """
    inds = np.where(burst)[0]
    inds = inds[
        (inds >= window) & 
        (inds <= len(signal) - window)
    ]
    
    new_inds = []
    for ind in inds:
        bw = signal[ind-window:ind+window+1]
        guess = np.argmax(bw)
        
        # In case there are more than one of the same peaks, just ignore it.
        if np.sum(bw == bw[guess]) > 1:
            new_inds.append(ind)
            continue
            
        new_ind = ind - window + guess
        new_inds.append(new_ind)
        
    new_inds = np.array(new_inds)
    new_burst = np.zeros_like(burst)
    new_burst[new_inds] = 1
    
    return new_burst


