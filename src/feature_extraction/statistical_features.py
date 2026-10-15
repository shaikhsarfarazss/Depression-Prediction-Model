import numpy as np
from scipy.stats import skew, kurtosis

def compute_statistical_features(epoch_channel: np.ndarray) -> np.ndarray:
    """
    Computes 8 unique statistical and Hjorth features for a single channel epoch:
    1. Mean
    2. Standard Deviation
    3. Skewness
    4. Kurtosis
    5. Zero-Crossing Rate (ZCR)
    6. Hjorth Activity (Signal variance)
    7. Hjorth Mobility (Mean frequency estimate)
    8. Hjorth Complexity (Frequency bandwidth change)
    """
    mean_val = float(np.mean(epoch_channel))
    var_val = float(np.var(epoch_channel))
    std_val = float(np.std(epoch_channel))

    skew_val = float(skew(epoch_channel)) if std_val > 1e-8 else 0.0
    kurt_val = float(kurtosis(epoch_channel)) if std_val > 1e-8 else 0.0

    # Zero-Crossing Rate around channel mean
    centered = epoch_channel - mean_val
    zcr_val = float(np.mean(np.diff(np.signbit(centered))))

    # Hjorth Parameters
    activity = var_val
    diff1 = np.diff(epoch_channel)
    std_diff1 = float(np.std(diff1))
    mobility = std_diff1 / (std_val + 1e-8)

    diff2 = np.diff(diff1)
    std_diff2 = float(np.std(diff2))
    mobility_diff = std_diff2 / (std_diff1 + 1e-8)
    complexity = mobility_diff / (mobility + 1e-8)

    return np.array([
        mean_val, std_val, skew_val, kurt_val,
        zcr_val, activity, mobility, complexity
    ], dtype=np.float32)
