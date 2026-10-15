import os
import numpy as np
import pandas as pd
from scipy.signal import resample
from typing import Tuple, List, Optional
from ..utils.logger import get_logger
from .synthetic_generator import STANDARD_19_CHANNELS

logger = get_logger("EEGLoader")

class EEGLoader:
    """
    Unified multi-format EEG loader supporting CSV, EDF, and NPY files
    with automatic channel alignment and sampling rate resampling.
    """
    def __init__(self, target_channels: Optional[List[str]] = None, sfreq: int = 256):
        self.target_channels = target_channels or STANDARD_19_CHANNELS
        self.sfreq = sfreq

    def _align_and_resample(
        self, data: np.ndarray, ch_names: List[str], file_sfreq: int
    ) -> Tuple[np.ndarray, List[str], int]:
        # 1. Align or pad/trim to target 19 channels
        clean_map = {
            c.replace("EEG ", "").replace("-REF", "").replace("-ref", "").strip().upper(): idx
            for idx, c in enumerate(ch_names)
        }
        aligned_data = np.zeros((len(self.target_channels), data.shape[1]), dtype=np.float32)
        for t_idx, target_ch in enumerate(self.target_channels):
            key = target_ch.upper()
            if key in clean_map:
                aligned_data[t_idx] = data[clean_map[key]]
            elif t_idx < data.shape[0]:
                aligned_data[t_idx] = data[t_idx]

        # 2. Resample if file sampling frequency differs from target sfreq
        if file_sfreq != self.sfreq and file_sfreq > 0:
            target_samples = int(round(aligned_data.shape[1] * (self.sfreq / float(file_sfreq))))
            if target_samples > 0:
                aligned_data = resample(aligned_data, target_samples, axis=1).astype(np.float32)

        return aligned_data, list(self.target_channels), self.sfreq

    def load_csv(self, file_path: str) -> Tuple[np.ndarray, List[str], int]:
        df = pd.read_csv(file_path)
        col_names = [c for c in df.columns if c.lower() not in ["time", "timestamp", "sample", "index"]]

        file_sfreq = self.sfreq
        time_cols = [c for c in df.columns if c.lower() in ["time", "timestamp"]]
        if time_cols and len(df) > 1:
            t = df[time_cols[0]].values
            dt = np.median(np.diff(t))
            if dt > 0:
                file_sfreq = int(round(1.0 / dt))

        data = df[col_names].values.T.astype(np.float32)
        return self._align_and_resample(data, col_names, file_sfreq)

    def load_edf(self, file_path: str) -> Tuple[np.ndarray, List[str], int]:
        try:
            import mne
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            data = raw.get_data().astype(np.float32)
            channel_names = raw.ch_names
            file_sfreq = int(raw.info["sfreq"])
            return self._align_and_resample(data, channel_names, file_sfreq)
        except Exception as e:
            logger.error(f"Error loading EDF with MNE: {e}")
            raise e

    def load(self, file_path: str) -> Tuple[np.ndarray, List[str], int]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"EEG file not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            return self.load_csv(file_path)
        elif ext in [".edf", ".bdf"]:
            return self.load_edf(file_path)
        elif ext == ".npy":
            data = np.load(file_path).astype(np.float32)
            if data.ndim == 2 and data.shape[0] > data.shape[1]:
                data = data.T
            channels = [f"Ch_{i+1}" for i in range(data.shape[0])]
            return self._align_and_resample(data, channels, self.sfreq)
        else:
            raise ValueError(f"Unsupported EEG file extension: {ext}")
