import numpy as np
from scipy.signal import welch

FREQUENCY_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0)
}

def compute_band_powers_and_psd(
    epoch_channel: np.ndarray,
    sfreq: int = 256
) -> np.ndarray:
    """
    Computes Power Spectral Density using Welch's method, absolute band powers,
    relative band powers, and clinical rhythm ratios:
    1. Delta power (0.5 - 4 Hz)
    2. Theta power (4 - 8 Hz)
    3. Alpha power (8 - 13 Hz)
    4. Beta power (13 - 30 Hz)
    5. Gamma power (30 - 45 Hz)
    6. Total Spectral Power
    7. Theta / Alpha Ratio (TAR - clinical biomarker for cognitive decline)
    8. Slowing Ratio: (Delta + Theta) / (Alpha + Beta)
    Returns: 1D array of 8 frequency/PSD features.
    """
    nperseg = min(len(epoch_channel), sfreq)
    freqs, psd = welch(epoch_channel, fs=sfreq, nperseg=nperseg)
    
    band_powers = {}
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    for band_name, (low, high) in FREQUENCY_BANDS.items():
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        if np.any(idx_band):
            # Integrate PSD over band
            power = trapz_fn(psd[idx_band], freqs[idx_band])
        else:
            power = 1e-8
        band_powers[band_name] = max(1e-8, float(power))

    total_power = sum(band_powers.values())
    p_delta = band_powers["delta"] / total_power
    p_theta = band_powers["theta"] / total_power
    p_alpha = band_powers["alpha"] / total_power
    p_beta  = band_powers["beta"] / total_power
    p_gamma = band_powers["gamma"] / total_power

    # Clinical ratios
    tar = band_powers["theta"] / (band_powers["alpha"] + 1e-8)
    slowing_ratio = (band_powers["delta"] + band_powers["theta"]) / (band_powers["alpha"] + band_powers["beta"] + 1e-8)

    return np.array([
        p_delta, p_theta, p_alpha, p_beta, p_gamma,
        np.log1p(total_power), tar, slowing_ratio
    ], dtype=np.float32)
