import numpy as np
from typing import Dict, Any
from .filters import apply_bandpass_filter, apply_notch_filter
from .artifact_removal import remove_ica_artifacts, clamp_extreme_artifacts
from .normalizer import zscore_normalize
from .segmenter import segment_signal
from ..utils.logger import get_logger

logger = get_logger("PreprocessingPipeline")

class PreprocessingPipeline:
    """
    End-to-End Preprocessing Pipeline:
    1. 50/60Hz Notch Filtering (power-line noise)
    2. Butterworth Bandpass Filtering (0.5 - 45 Hz)
    3. Artifact Suppression (Outlier Clamping & FastICA)
    4. Fixed-Epoch Segmentation (Preserves microvolt amplitudes for biomarker extraction)
    """
    def __init__(
        self,
        lowcut: float = 0.5,
        highcut: float = 45.0,
        notch_freq: float = 50.0,
        sfreq: int = 256,
        epoch_duration: float = 2.0,
        overlap: float = 0.5,
        enable_ica: bool = True
    ):
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.sfreq = sfreq
        self.epoch_duration = epoch_duration
        self.overlap = overlap
        self.enable_ica = enable_ica

    def process(self, raw_eeg: np.ndarray) -> Dict[str, Any]:
        # Step 1: Notch filter (50 Hz)
        s1_notch = apply_notch_filter(raw_eeg, freq=self.notch_freq, fs=self.sfreq)

        # Step 2: Bandpass filter (0.5 - 45 Hz)
        s2_bandpass = apply_bandpass_filter(
            s1_notch, lowcut=self.lowcut, highcut=self.highcut, fs=self.sfreq
        )

        # Step 3: Extreme artifact clamp & optional FastICA
        s3_clamped = clamp_extreme_artifacts(s2_bandpass, std_factor=6.0)
        if self.enable_ica and raw_eeg.shape[0] >= 4 and raw_eeg.shape[1] >= 256:
            s3_clean = remove_ica_artifacts(s3_clamped)
        else:
            s3_clean = s3_clamped

        # Step 4: Segment unnormalized clean signal to preserve inter-hemispheric & amplitude biomarkers
        epochs = segment_signal(
            s3_clean, sfreq=self.sfreq, epoch_duration=self.epoch_duration, overlap=self.overlap
        )

        return {
            "cleaned_continuous": s3_clean,
            "normalized_continuous": zscore_normalize(s3_clean),
            "raw": raw_eeg,
            "epochs": epochs,
            "sfreq": self.sfreq
        }
