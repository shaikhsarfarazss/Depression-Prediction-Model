import numpy as np
from sklearn.decomposition import FastICA
from ..utils.logger import get_logger

logger = get_logger("ArtifactRemoval")

def remove_ica_artifacts(
    data: np.ndarray,
    n_components: int = 15,
    max_kurtosis_threshold: float = 5.0
) -> np.ndarray:
    """
    Apply FastICA to decompose multi-channel EEG, identify high-kurtosis
    artifact components (e.g. eye blinks, muscle spikes), and reconstruct clean signal.
    data shape: (channels, samples)
    """
    n_channels, n_samples = data.shape
    n_comp = min(n_components, n_channels, n_samples // 50)
    if n_comp < 2:
        return data

    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ica = FastICA(n_components=n_comp, random_state=42, max_iter=250, tol=1e-2)
            # FastICA expects shape (n_samples, n_features)
            components = ica.fit_transform(data.T) # shape: (n_samples, n_comp)

        # Calculate kurtosis of each independent component
        clean_components = components.copy()
        for c in range(n_comp):
            comp_series = components[:, c]
            mean = np.mean(comp_series)
            std = np.std(comp_series) + 1e-8
            kurtosis = np.mean(((comp_series - mean) / std) ** 4) - 3.0
            
            # If kurtosis exceeds threshold (typical of sharp blink/muscle spikes), suppress it
            if abs(kurtosis) > max_kurtosis_threshold:
                clean_components[:, c] *= 0.1 # Suppress artifact component

        # Reconstruct signal
        reconstructed = ica.inverse_transform(clean_components).T
        return reconstructed
    except Exception as e:
        logger.warning(f"ICA artifact removal encountered warning/error ({e}), using raw filtered data.")
        return data

def clamp_extreme_artifacts(data: np.ndarray, std_factor: float = 5.0) -> np.ndarray:
    """
    Clip extreme voltage outliers beyond +/- 5 standard deviations.
    """
    clamped = np.copy(data)
    for ch in range(data.shape[0]):
        mean = np.mean(clamped[ch, :])
        std = np.std(clamped[ch, :])
        lower_bound = mean - std_factor * std
        upper_bound = mean + std_factor * std
        clamped[ch, :] = np.clip(clamped[ch, :], lower_bound, upper_bound)
    return clamped
