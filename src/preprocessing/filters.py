import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Clip between 0 and 1 for stability
    low = max(0.001, min(low, 0.99))
    high = max(0.001, min(high, 0.99))
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(
    data: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    fs: float = 256.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter (0.5 - 45 Hz) across channels.
    data shape: (channels, samples)
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered_data[ch, :] = filtfilt(b, a, data[ch, :])
    return filtered_data

def apply_notch_filter(
    data: np.ndarray,
    freq: float = 50.0,
    fs: float = 256.0,
    q: float = 30.0
) -> np.ndarray:
    """
    Apply notch filter to eliminate 50Hz or 60Hz power-line AC noise.
    """
    nyq = 0.5 * fs
    w0 = freq / nyq
    if w0 >= 1.0:
        return data
    b, a = iirnotch(w0, q)
    filtered_data = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered_data[ch, :] = filtfilt(b, a, data[ch, :])
    return filtered_data
