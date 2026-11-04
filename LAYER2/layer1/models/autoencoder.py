"""
BenignAutoencoder — copy nguyen kien truc tu
UTS_Layer1/AE_XGBoost/autoencoder_xgboost_ids_v2.ipynb (Cell 4), KHONG doi logic.

Encoder: input_dim -> 64 -> 32 -> bottleneck_dim
Decoder: bottleneck_dim -> 32 -> 64 -> input_dim

Train tren benign-only. Reconstruction error = anomaly score.
Dung LayerNorm thay BatchNorm de on dinh voi moi batch size.

Vai tro trong pipeline AE+XGBoost: AE CHI dong vai tro feature encoder —
latent_z (bottleneck_dim) + recon_error (1) duoc noi vao 78 feature goc lam
input cho XGBoost. XGBoost la diem quyet dinh DUY NHAT (threshold tren
xgb_proba) — khong co nguong AE rieng, khong OR-Fusion.
"""
import torch
import torch.nn as nn


class BenignAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 78, bottleneck_dim: int = 16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE error — dung lam anomaly score."""
        x_hat, _ = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=1)
