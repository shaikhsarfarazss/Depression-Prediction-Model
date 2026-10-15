import numpy as np
import pywt

def compute_wavelet_features(
    epoch_channel: np.ndarray,
    wavelet_name: str = "db4",
    level: int = 5
) -> np.ndarray:
    """
    Computes Discrete Wavelet Transform (DWT) multi-level decomposition features:
    - Subband Energy for A5, D5, D4, D3, D2, D1 (6 features)
    - Subband Shannon Entropy for A5, D5, D4, D3, D2, D1 (6 features)
    Total: 12 wavelet features per channel epoch.
    """
    try:
        max_level = pywt.dwt_max_level(len(epoch_channel), pywt.Wavelet(wavelet_name).dec_len)
        actual_level = min(level, max_level)
        if actual_level < 1:
            return np.zeros(12, dtype=np.float32)

        coeffs = pywt.wavedec(epoch_channel, wavelet_name, level=actual_level)
        # coeffs = [cA_n, cD_n, cD_n-1, ..., cD_1]

        energies = []
        entropies = []

        for c in coeffs:
            # Energy = sum of squares
            energy = np.sum(c ** 2) / (len(c) + 1e-8)
            energies.append(energy)

            # Shannon entropy of normalized energy
            p = (c ** 2) / (np.sum(c ** 2) + 1e-8)
            p = p[p > 0]
            shannon_ent = -np.sum(p * np.log2(p + 1e-12))
            entropies.append(shannon_ent)

        # Pad or trim to exactly 6 energies and 6 entropies (12 total)
        while len(energies) < 6:
            energies.append(0.0)
            entropies.append(0.0)

        energies = energies[:6]
        entropies = entropies[:6]

        wavelet_feats = np.array(energies + entropies, dtype=np.float32)
        # Log scale energies for numerical stability
        wavelet_feats[:6] = np.log1p(np.maximum(0, wavelet_feats[:6]))
        return wavelet_feats

    except Exception:
        return np.zeros(12, dtype=np.float32)
