import numpy as np
from typing import List

def segment_signal(
    data: np.ndarray,
    sfreq: int = 256,
    epoch_duration: float = 2.0,
    overlap: float = 0.5
) -> np.ndarray:
    """
    Split continuous multi-channel EEG into overlapping fixed-duration epochs.
    Returns: 3D array of shape (n_epochs, n_channels, epoch_samples)
    """
    n_channels, total_samples = data.shape
    epoch_samples = int(epoch_duration * sfreq)
    step_samples = int(epoch_samples * (1.0 - overlap))

    if total_samples < epoch_samples:
        # Zero-pad if signal is shorter than epoch
        padded = np.zeros((n_channels, epoch_samples))
        padded[:, :total_samples] = data
        return np.expand_dims(padded, axis=0)

    epochs = []
    start = 0
    while start + epoch_samples <= total_samples:
        epoch = data[:, start:start + epoch_samples]
        epochs.append(epoch)
        start += step_samples

    return np.array(epochs, dtype=np.float32)
