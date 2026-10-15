import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

def plot_confusion_matrix_heatmap(
    cm: List[List[int]],
    class_names: List[str] = ["Normal", "Alzheimer's Disease", "Depression"],
    figsize: tuple = (6, 5)
) -> plt.Figure:
    """
    Renders high-contrast confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#1e1e2f")
    ax.set_facecolor("#1e1e2f")

    cm_array = np.array(cm)
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        annot_kws={"size": 12, "weight": "bold", "color": "white"}
    )

    ax.set_title("Confusion Matrix — Proposed Hybrid Framework", color="white", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("True Condition", color="white", fontsize=10)
    ax.set_xlabel("Predicted Condition", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=9)

    plt.tight_layout()
    return fig

def plot_training_curves(history: Dict[str, list], figsize: tuple = (10, 4)) -> plt.Figure:
    """
    Plots training loss and validation accuracy trajectories over epochs.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor="#1e1e2f")
    ax1.set_facecolor("#1e1e2f")
    ax2.set_facecolor("#1e1e2f")

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    ax1.plot(epochs, history["train_loss"], color="#00e5ff", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], color="#ff5252", linestyle="--", label="Val Loss", linewidth=2)
    ax1.set_title("CNN Cross-Entropy Loss", color="white", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Epoch", color="white")
    ax1.set_ylabel("Loss", color="white")
    ax1.grid(True, color="#333", linestyle="--", alpha=0.5)
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#2d2d44", labelcolor="white")

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], color="#00e676", label="Train Acc", linewidth=2)
    ax2.plot(epochs, history["val_acc"], color="#ffd600", linestyle="--", label="Val Acc", linewidth=2)
    ax2.set_title("Accuracy Trajectory (%)", color="white", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Epoch", color="white")
    ax2.set_ylabel("Accuracy (%)", color="white")
    ax2.grid(True, color="#333", linestyle="--", alpha=0.5)
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#2d2d44", labelcolor="white")

    plt.tight_layout()
    return fig
