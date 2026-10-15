import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

STANDARD_19_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4", "T5", "P3",
    "Pz", "P4", "T6", "O1", "O2"
]

def generate_eeg_epoch(
    condition: str,
    duration: float = 10.0,
    sfreq: int = 256,
    channels: List[str] = None,
    add_artifacts: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    if channels is None:
        channels = STANDARD_19_CHANNELS

    n_samples = int(duration * sfreq)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    eeg_data = np.zeros((len(channels), n_samples), dtype=np.float32)

    # Subject-specific biological variability
    subj_scale = np.random.uniform(0.75, 1.30)
    subj_noise_level = np.random.uniform(4.5, 8.5)
    cond = condition.lower()

    for idx, ch in enumerate(channels):
        noise = np.random.normal(0, subj_noise_level, n_samples)

        delta = np.sin(2 * np.pi * np.random.uniform(1.0, 3.8) * t + np.random.uniform(0, 2 * np.pi))
        theta = np.sin(2 * np.pi * np.random.uniform(4.2, 7.8) * t + np.random.uniform(0, 2 * np.pi))
        alpha = np.sin(2 * np.pi * np.random.uniform(8.5, 12.2) * t + np.random.uniform(0, 2 * np.pi))
        beta  = np.sin(2 * np.pi * np.random.uniform(14.0, 26.0) * t + np.random.uniform(0, 2 * np.pi))
        gamma = np.sin(2 * np.pi * np.random.uniform(31.0, 43.0) * t + np.random.uniform(0, 2 * np.pi))

        if cond == "normal":
            base_alpha = 11.5 if ch in ["O1", "O2", "P3", "P4", "Pz"] else 5.5
            alpha_gain = max(1.0, np.random.normal(base_alpha, 2.2))
            beta_gain  = max(1.0, np.random.normal(6.0, 1.5))
            theta_gain = max(0.5, np.random.normal(3.8, 1.4))
            delta_gain = max(0.5, np.random.normal(4.5, 1.5))
            gamma_gain = max(0.2, np.random.normal(1.6, 0.5))

        elif cond in ["alzheimer", "alzheimer's disease", "ad"]:
            alpha_gain = max(0.5, np.random.normal(3.2, 1.4))
            beta_gain  = max(0.5, np.random.normal(2.2, 1.0))
            theta_gain = max(2.0, np.random.normal(11.5, 2.8))
            delta_gain = max(2.0, np.random.normal(13.5, 3.0))
            gamma_gain = max(0.2, np.random.normal(0.8, 0.4))

        elif cond in ["depression", "mdd"]:
            if ch in ["F3", "Fp1", "F7"]:
                alpha_gain = max(1.5, np.random.normal(10.5, 2.2))
            elif ch in ["F4", "Fp2", "F8"]:
                alpha_gain = max(1.0, np.random.normal(4.2, 1.5))
            else:
                alpha_gain = max(1.0, np.random.normal(6.2, 1.8))
            beta_gain  = max(0.8, np.random.normal(4.2, 1.3))
            theta_gain = max(1.5, np.random.normal(8.8 if "F" in ch else 4.5, 2.0))
            delta_gain = max(1.0, np.random.normal(5.5, 1.6))
            gamma_gain = max(0.2, np.random.normal(1.2, 0.4))
        else:
            alpha_gain, beta_gain, theta_gain, delta_gain, gamma_gain = 6.0, 4.0, 4.0, 4.0, 1.0

        signal = subj_scale * (
            delta_gain * delta +
            theta_gain * theta +
            alpha_gain * alpha +
            beta_gain * beta +
            gamma_gain * gamma
        ) + noise

        if add_artifacts:
            signal += 3.5 * np.sin(2 * np.pi * 50.0 * t)
            if ch in ["Fp1", "Fp2"]:
                blink_times = np.arange(1.5, duration, 2.8)
                for bt in blink_times:
                    center_sample = int(bt * sfreq)
                    blink_width = int(0.25 * sfreq)
                    start = max(0, center_sample - blink_width // 2)
                    end = min(n_samples, center_sample + blink_width // 2)
                    window = np.hanning(end - start)
                    signal[start:end] += 55.0 * window

        eeg_data[idx, :] = signal

    return eeg_data, t

def create_sample_dataset(
    output_dir: str,
    n_samples_per_class: int = 40,
    duration: float = 8.0,
    sfreq: int = 256
) -> Dict[str, List[str]]:
    os.makedirs(output_dir, exist_ok=True)
    generated_files = {"Normal": [], "Alzheimer's Disease": [], "Depression": []}

    classes_map = [
        ("Normal", "normal"),
        ("Alzheimer's Disease", "alzheimer"),
        ("Depression", "depression")
    ]

    for class_name, condition_key in classes_map:
        class_folder = os.path.join(output_dir, condition_key)
        os.makedirs(class_folder, exist_ok=True)
        for i in range(1, n_samples_per_class + 1):
            eeg_data, t = generate_eeg_epoch(
                condition=condition_key,
                duration=duration,
                sfreq=sfreq,
                add_artifacts=True
            )
            df = pd.DataFrame(eeg_data.T, columns=STANDARD_19_CHANNELS)
            df.insert(0, "time", t)

            filepath = os.path.join(class_folder, f"{condition_key}_sub{i:03d}.csv")
            df.to_csv(filepath, index=False)
            generated_files[class_name].append(filepath)

    return generated_files
