import numpy as np
from typing import Dict, Any, List
from .statistical_features import compute_statistical_features
from .frequency_features import compute_band_powers_and_psd
from .wavelet_features import compute_wavelet_features
from ..utils.logger import get_logger

logger = get_logger("FeaturePipeline")

class FeatureExtractionPipeline:
    """
    Unified Feature Extraction Pipeline extracting 28 features per channel:
    - 8 Statistical / Hjorth features
    - 8 Frequency-domain / PSD / Clinical ratio features
    - 12 DWT Wavelet Subband Energy & Entropy features
    Total: 28 features per channel (e.g. 19 x 28 = 532 features per epoch).
    """
    def __init__(self, sfreq: int = 256, wavelet_name: str = "db4", wavelet_level: int = 5):
        self.sfreq = sfreq
        self.wavelet_name = wavelet_name
        self.wavelet_level = wavelet_level

    def extract_epoch_features(self, epoch_data: np.ndarray) -> np.ndarray:
        """
        Extract features for a single epoch of shape (n_channels, n_samples).
        Returns: 2D array of shape (n_channels, 28)
        """
        n_channels = epoch_data.shape[0]
        channel_feats = []

        for ch in range(n_channels):
            ch_sig = epoch_data[ch, :]
            stat_f = compute_statistical_features(ch_sig) # 8 features
            freq_f = compute_band_powers_and_psd(ch_sig, sfreq=self.sfreq) # 8 features
            wav_f  = compute_wavelet_features(ch_sig, wavelet_name=self.wavelet_name, level=self.wavelet_level) # 12 features

            combined = np.concatenate([stat_f, freq_f, wav_f]) # 28 features
            channel_feats.append(combined)

        return np.array(channel_feats, dtype=np.float32)

    def extract_batch(self, epochs: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract features for multiple epochs of shape (n_epochs, n_channels, n_samples).
        Returns:
        - 'features_3d': shape (n_epochs, n_channels, 28) [for CNN]
        - 'features_2d': shape (n_epochs, n_channels * 28) [for classical ML]
        """
        n_epochs = epochs.shape[0]
        all_epochs_3d = []

        for e in range(n_epochs):
            feat_matrix = self.extract_epoch_features(epochs[e])
            all_epochs_3d.append(feat_matrix)

        features_3d = np.array(all_epochs_3d, dtype=np.float32)
        features_2d = features_3d.reshape(n_epochs, -1)

        return {
            "features_3d": features_3d,
            "features_2d": features_2d
        }
