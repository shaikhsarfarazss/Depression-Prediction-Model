import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

def plot_eeg_signals(
    raw_eeg: np.ndarray,
    clean_eeg: Optional[np.ndarray] = None,
    channel_names: Optional[List[str]] = None,
    sfreq: int = 256,
    duration: float = 4.0,
    max_channels: int = 8,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plots multi-channel EEG signals comparing Raw vs Preprocessed (Clean) signals.
    """
    n_samples = min(int(duration * sfreq), raw_eeg.shape[1])
    n_channels = min(max_channels, raw_eeg.shape[0])
    t = np.arange(n_samples) / sfreq

    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True, facecolor="#1e1e2f")
    if n_channels == 1:
        axes = [axes]

    for i in range(n_channels):
        ax = axes[i]
        ax.set_facecolor("#1e1e2f")
        ch_name = channel_names[i] if channel_names and i < len(channel_names) else f"Ch {i+1}"
        
        # Raw signal in translucent red/orange
        ax.plot(t, raw_eeg[i, :n_samples], color="#ff5252", alpha=0.6, linewidth=1.0, label="Raw" if i==0 else "")
        
        # Cleaned signal in bright cyan
        if clean_eeg is not None:
            ax.plot(t, clean_eeg[i, :n_samples], color="#00e5ff", alpha=0.9, linewidth=1.2, label="Preprocessed" if i==0 else "")

        ax.set_ylabel(ch_name, color="white", fontsize=8, rotation=0, labelpad=25, va="center")
        ax.tick_params(colors="white", labelsize=7)
        ax.spines["bottom"].set_color("#444")
        ax.spines["top"].set_color("none")
        ax.spines["right"].set_color("none")
        ax.spines["left"].set_color("#444")
        ax.grid(True, color="#333", linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Time (seconds)", color="white", fontsize=9)
    if clean_eeg is not None:
        axes[0].legend(loc="upper right", facecolor="#2d2d44", edgecolor="#00e5ff", labelcolor="white", fontsize=8)

    fig.suptitle("Multi-Channel EEG Signals (Raw vs Preprocessed)", color="white", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig
