import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ElectrodeSEAttention(nn.Module):
    """
    Squeeze-and-Excitation (SE) Spatial Attention across EEG channels.
    Learns to dynamically weight the importance of specific 10-20 electrodes
    (e.g., F3/F4 frontal asymmetry in Depression, O1/O2 posterior alpha in AD).
    """
    def __init__(self, n_channels: int, reduction: int = 4):
        super(ElectrodeSEAttention, self).__init__()
        hidden = max(4, n_channels // reduction)
        self.fc1 = nn.Linear(n_channels, hidden)
        self.fc2 = nn.Linear(hidden, n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, L)
        squeeze = x.mean(dim=-1) # (B, C)
        attn = torch.sigmoid(self.fc2(F.gelu(self.fc1(squeeze)))).unsqueeze(-1) # (B, C, 1)
        return x * attn

class EEGDeepCNN(nn.Module):
    """
    Attention-Enhanced 1D-CNN for Deep EEG Feature Extraction & Classification.
    Input: (batch_size, n_channels=19, n_features=28)
    Outputs:
    - 64-dimensional Deep Feature Vector (for SVM & KNN)
    - 3-class classification logits (for end-to-end CNN prediction)
    """
    def __init__(
        self,
        n_channels: int = 19,
        n_features: int = 28,
        n_classes: int = 3,
        dropout_rate: float = 0.3
    ):
        super(EEGDeepCNN, self).__init__()
        self.n_channels = n_channels
        self.n_features = n_features
        self.n_classes = n_classes

        # Spatial Electrode Attention on input channels
        self.electrode_attn = ElectrodeSEAttention(n_channels, reduction=3)

        # Block 1: Pointwise + Local Feature Mixing (B, 19, 28) -> (B, 64, 14)
        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(dropout_rate)

        # Block 2: (B, 64, 14) -> (B, 128, 7)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.se2 = ElectrodeSEAttention(128, reduction=8)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.drop2 = nn.Dropout(dropout_rate)

        # Block 3: (B, 128, 7) -> (B, 256, 1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Deep Feature Representation Layer (64-dimensional bottleneck)
        self.fc_deep = nn.Linear(256, 64)
        self.bn_deep = nn.BatchNorm1d(64)
        self.drop_deep = nn.Dropout(0.4)

        # CNN Classifier Head
        self.classifier = nn.Linear(64, n_classes)

    def extract_deep_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.electrode_attn(x)

        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.drop1(self.pool1(out))

        out = F.gelu(self.bn2(self.conv2(out)))
        out = self.se2(out)
        out = self.drop2(self.pool2(out))

        out = F.gelu(self.bn3(self.conv3(out)))
        out = self.global_pool(out).squeeze(-1) # (B, 256)

        deep_features = F.relu(self.bn_deep(self.fc_deep(out)))
        return deep_features

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        deep_features = self.extract_deep_features(x)
        dropped = self.drop_deep(deep_features)
        logits = self.classifier(dropped)
        return logits, deep_features

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            return F.softmax(logits, dim=1)
