import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Tuple
from .cnn_model import EEGDeepCNN
from .hybrid_ensemble import HybridCNNSVMKNNEnsemble
from ..utils.logger import get_logger

logger = get_logger("ModelTrainer")

class ModelTrainer:
    def __init__(
        self,
        n_channels: int = 19,
        n_features: int = 28,
        n_classes: int = 3,
        epochs: int = 35,
        batch_size: int = 32,
        lr: float = 0.001,
        device: str = "cpu"
    ):
        self.n_channels = n_channels
        self.n_features = n_features
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

    def train_cnn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[EEGDeepCNN, Dict[str, list]]:
        cnn = EEGDeepCNN(
            n_channels=self.n_channels,
            n_features=self.n_features,
            n_classes=self.n_classes
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(cnn.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_acc = -1.0
        best_state = None

        logger.info(f"Starting CNN training ({self.epochs} epochs)...")

        for epoch in range(1, self.epochs + 1):
            cnn.train()
            total_train_loss, correct_train, total_train_samples = 0.0, 0, 0

            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits, _ = cnn(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * bx.size(0)
                preds = torch.argmax(logits, dim=1)
                correct_train += (preds == by).sum().item()
                total_train_samples += bx.size(0)

            epoch_train_loss = total_train_loss / max(1, total_train_samples)
            epoch_train_acc = (correct_train / max(1, total_train_samples)) * 100.0

            cnn.eval()
            total_val_loss, correct_val, total_val_samples = 0.0, 0, 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    logits, _ = cnn(bx)
                    loss = criterion(logits, by)
                    total_val_loss += loss.item() * bx.size(0)
                    preds = torch.argmax(logits, dim=1)
                    correct_val += (preds == by).sum().item()
                    total_val_samples += bx.size(0)

            epoch_val_loss = total_val_loss / max(1, total_val_samples)
            epoch_val_acc = (correct_val / max(1, total_val_samples)) * 100.0

            scheduler.step(epoch_val_acc)

            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(epoch_val_loss)
            history["train_acc"].append(epoch_train_acc)
            history["val_acc"].append(epoch_val_acc)

            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                # Deep copy prevents in-place tensor mutation on subsequent epochs
                best_state = copy.deepcopy(cnn.state_dict())

            if epoch % 5 == 0 or epoch == self.epochs:
                logger.info(
                    f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                    f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.1f}% | "
                    f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.1f}%"
                )

        if best_state is not None:
            cnn.load_state_dict(best_state)

        logger.info(f"CNN Training Complete. Best Val Accuracy: {best_val_acc:.2f}%")
        return cnn, history

    def train_hybrid_framework(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        fusion_strategy: str = "weighted_voting"
    ) -> Tuple[HybridCNNSVMKNNEnsemble, Dict[str, Any]]:
        cnn, history = self.train_cnn(X_train, y_train, X_val, y_val)

        cnn.eval()
        with torch.no_grad():
            t_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
            t_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            logits_val, deep_val_t = cnn(t_val)
            p_cnn_val = torch.softmax(logits_val, dim=1).cpu().numpy()
            deep_train = cnn.extract_deep_features(t_train).cpu().numpy()
            deep_val = deep_val_t.cpu().numpy()

        ensemble = HybridCNNSVMKNNEnsemble(
            cnn_model=cnn,
            fusion_strategy=fusion_strategy,
            device=self.device
        )
        ensemble.fit_shallow(
            X_deep=deep_train,
            y=y_train,
            X_val_deep=deep_val,
            y_val=y_val,
            p_cnn_val=p_cnn_val
        )

        return ensemble, history
