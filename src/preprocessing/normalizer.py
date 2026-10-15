import numpy as np

def zscore_normalize(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Apply zero-mean unit-variance normalization per channel.
    Handles 2D (channels, samples) or 3D (epochs, channels, samples).
    """
    if data.ndim == 2:
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True) + eps
        return (data - mean) / std
    elif data.ndim == 3:
        mean = np.mean(data, axis=2, keepdims=True)
        std = np.std(data, axis=2, keepdims=True) + eps
        return (data - mean) / std
    else:
        return data
