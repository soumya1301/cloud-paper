"""
Autoencoder Module (PyTorch)
Implements the 16-12-5-12-16 Autoencoder architecture from Section 4 & 6.3 of the paper.
Compresses 16-dimensional VM features into 5 latent features for SVM classification.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List

class AutoencoderNN(nn.Module):
    """
    PyTorch Autoencoder Architecture: 16 -> 12 -> 5 -> 12 -> 16
    Encoder: Dense(16, 12) + ReLU -> Dense(12, 5) + ReLU
    Decoder: Dense(5, 12) + ReLU -> Dense(12, 16) + Sigmoid
    """
    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim1: int = 12,
        latent_dim: int = 5,
        hidden_dim2: int = 12,
        output_dim: int = 16
    ):
        super(AutoencoderNN, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim),
            nn.Sigmoid() # Features are normalized in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class VMAutoencoder:
    """
    Wrapper for Autoencoder training, latent feature extraction, and evaluation.
    """
    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim1: int = 12,
        latent_dim: int = 5,
        hidden_dim2: int = 12,
        output_dim: int = 16,
        learning_rate: float = 0.001,
        epochs: int = 500,
        batch_size: int = 64,
        weight_decay: float = 1e-5,
        device: str = None
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoencoderNN(
            input_dim=input_dim,
            hidden_dim1=hidden_dim1,
            latent_dim=latent_dim,
            hidden_dim2=hidden_dim2,
            output_dim=output_dim
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.history = {"train_loss": [], "val_loss": []}

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray = None,
        verbose: bool = True
    ) -> "VMAutoencoder":
        """
        Trains the Autoencoder using MSE loss and Adam optimizer.
        """
        train_tensor = torch.tensor(X_train, dtype=torch.float32)
        train_dataset = TensorDataset(train_tensor, train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_tensor = None
        if X_val is not None:
            val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        print(f"[*] Training Autoencoder ({self.input_dim}->12->{self.latent_dim}->12->{self.input_dim}) for {self.epochs} epochs on {self.device}...")
        
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_train_loss = 0.0
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device)
                
                self.optimizer.zero_grad()
                recon, _ = self.model(batch_x)
                loss = self.criterion(recon, batch_x)
                loss.backward()
                self.optimizer.step()
                
                total_train_loss += loss.item() * batch_x.size(0)

            epoch_train_loss = total_train_loss / len(train_tensor)
            self.history["train_loss"].append(epoch_train_loss)

            # Validation loss
            if val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_recon, _ = self.model(val_tensor)
                    val_loss = self.criterion(val_recon, val_tensor).item()
                    self.history["val_loss"].append(val_loss)
                self.model.train()
            else:
                self.history["val_loss"].append(epoch_train_loss)

            if verbose and (epoch % 50 == 0 or epoch == 1 or epoch == self.epochs):
                val_str = f" - Val MSE: {self.history['val_loss'][-1]:.6f}" if val_tensor is not None else ""
                print(f"    Epoch [{epoch:03d}/{self.epochs}] - Train MSE: {epoch_train_loss:.6f}{val_str}")

        return self

    def extract_latent(self, X: np.ndarray) -> np.ndarray:
        """
        Compresses 16-dimensional feature matrix X into 5-dimensional latent representation.
        """
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            latent = self.model.encode(x_tensor)
            return latent.cpu().numpy()

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstructs input features through Encoder-Decoder."""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            recon, _ = self.model(x_tensor)
            return recon.cpu().numpy()

    def evaluate_reconstruction(self, X_test: np.ndarray) -> Dict[str, float]:
        """Calculates test reconstruction MSE and accuracy metric."""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            recon, _ = self.model(x_tensor)
            mse = self.criterion(recon, x_tensor).item()
            
            # Reconstruction accuracy: 1.0 - normalized mean absolute error
            mae = torch.mean(torch.abs(recon - x_tensor)).item()
            accuracy = max(0.0, 1.0 - mae) * 100.0

        return {
            "test_mse": float(mse),
            "test_mae": float(mae),
            "reconstruction_accuracy": float(accuracy)
        }

    def save(self, filepath: str = "models/autoencoder.pth"):
        """Saves model weights and configuration."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "history": self.history
        }, filepath)
        print(f"[+] Saved Autoencoder model to '{filepath}'")

    @classmethod
    def load(cls, filepath: str = "models/autoencoder.pth", device: str = None) -> "VMAutoencoder":
        """Loads model weights and configuration."""
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(filepath, map_location=dev)
        ae = cls(
            input_dim=checkpoint.get("input_dim", 16),
            latent_dim=checkpoint.get("latent_dim", 5),
            device=dev
        )
        ae.model.load_state_dict(checkpoint["model_state_dict"])
        ae.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})
        ae.model.eval()
        return ae

    def plot_loss(self, save_path: str = "results/figures/fig5_autoencoder_loss.png"):
        """Generates train/val reconstruction loss curve matching Paper Fig. 5a."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        epochs = range(1, len(self.history["train_loss"]) + 1)
        
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history["train_loss"], label="Training Loss (MSE)", color="#025e8d", linewidth=2)
        if self.history["val_loss"]:
            plt.plot(epochs, self.history["val_loss"], label="Validation Loss (MSE)", color="#d95f02", linestyle="--", linewidth=2)
        
        plt.title("Autoencoder Reconstruction Loss Curve (Paper Fig. 5a)", fontsize=13, fontweight="bold")
        plt.xlabel("Epochs", fontsize=11)
        plt.ylabel("Loss (MSE)", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[+] Saved Autoencoder loss graph to '{save_path}'")
