import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from typing import Dict, List, Optional

# Standard 10-20 2D Topographic Coordinates (Normalized X, Y on unit scalp circle)
ELECTRODE_2D_COORDS = {
    "Fp1": (-0.30, 0.85),
    "Fp2": ( 0.30, 0.85),
    "F7":  (-0.75, 0.50),
    "F3":  (-0.38, 0.50),
    "Fz":  ( 0.00, 0.50),
    "F4":  ( 0.38, 0.50),
    "F8":  ( 0.75, 0.50),
    "T3":  (-0.90, 0.00),
    "C3":  (-0.45, 0.00),
    "Cz":  ( 0.00, 0.00),
    "C4":  ( 0.45, 0.00),
    "T4":  ( 0.90, 0.00),
    "T5":  (-0.75, -0.50),
    "P3":  (-0.38, -0.50),
    "Pz":  ( 0.00, -0.50),
    "P4":  ( 0.38, -0.50),
    "T6":  ( 0.75, -0.50),
    "O1":  (-0.30, -0.85),
    "O2":  ( 0.30, -0.85)
}

def plot_brain_topomap(
    channel_values: Dict[str, float],
    title: str = "Cortical Activity Topographic Map",
    cmap: str = "plasma",
    figsize: tuple = (5, 5)
) -> plt.Figure:
    """
    Plots a 2D EEG Topographic Brain Map with head outline, nose, ears,
    and electrode activity heat map interpolated across scalp coordinates.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#1e1e2f")
    ax.set_facecolor("#1e1e2f")

    # Filter matching electrodes
    xs, ys, vals, labels = [], [], [], []
    for ch, val in channel_values.items():
        # Match standard names
        clean_ch = ch.replace("EEG ", "").replace("-REF", "").replace("-ref", "").strip()
        for k in ELECTRODE_2D_COORDS:
            if clean_ch.upper() == k.upper():
                x, y = ELECTRODE_2D_COORDS[k]
                xs.append(x)
                ys.append(y)
                vals.append(val)
                labels.append(k)
                break

    if len(xs) < 4:
        ax.text(0, 0, "Insufficient channels for Topomap", color="white", ha="center")
        ax.axis("off")
        return fig

    xs = np.array(xs)
    ys = np.array(ys)
    vals = np.array(vals)

    # Grid for interpolation
    grid_x, grid_y = np.mgrid[-1.05:1.05:150j, -1.05:1.05:150j]
    grid_z = griddata((xs, ys), vals, (grid_x, grid_y), method="cubic", fill_value=np.mean(vals))

    # Mask outside the unit circle head radius (r = 1.0)
    radius = np.sqrt(grid_x**2 + grid_y**2)
    grid_z[radius > 0.98] = np.nan

    # Plot interpolated heatmap
    im = ax.imshow(
        grid_z.T,
        extent=[-1.05, 1.05, -1.05, 1.05],
        origin="lower",
        cmap=cmap,
        aspect="equal"
    )

    # Draw head circle
    head_circle = plt.Circle((0, 0), 0.98, color="#00e5ff", fill=False, linewidth=2.5)
    ax.add_artist(head_circle)

    # Draw nose
    nose_x = [ -0.12, 0.0, 0.12 ]
    nose_y = [ 0.98, 1.12, 0.98 ]
    ax.plot(nose_x, nose_y, color="#00e5ff", linewidth=2.5)

    # Draw ears
    ear_theta = np.linspace(-np.pi/2, np.pi/2, 30)
    left_ear_x = -0.98 - 0.06 * np.cos(ear_theta)
    left_ear_y = 0.16 * np.sin(ear_theta)
    ax.plot(left_ear_x, left_ear_y, color="#00e5ff", linewidth=2.0)

    right_ear_x = 0.98 + 0.06 * np.cos(ear_theta)
    right_ear_y = 0.16 * np.sin(ear_theta)
    ax.plot(right_ear_x, right_ear_y, color="#00e5ff", linewidth=2.0)

    # Plot electrode dots and labels
    ax.scatter(xs, ys, color="white", edgecolors="#00e5ff", s=45, zorder=5)
    for x, y, lab in zip(xs, ys, labels):
        ax.text(x, y + 0.04, lab, color="white", fontsize=7, fontweight="bold", ha="center", zorder=6)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.06)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    cbar.outline.set_edgecolor("#00e5ff")

    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.axis("off")
    plt.tight_layout()
    return fig
