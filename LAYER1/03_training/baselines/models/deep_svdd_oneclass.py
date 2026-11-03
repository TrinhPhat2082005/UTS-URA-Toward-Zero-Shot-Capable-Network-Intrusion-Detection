"""One-Class Deep SVDD model following the core objective from Ruff et al. (2018)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


class DeepSVDDEncoder(nn.Module):
    """Bias-free MLP encoder for tabular features (paper CNN adapted to CICIDS)."""

    def __init__(self, input_dim: int, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(256, 128, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(128, embed_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OneClassDeepSVDD:
    """Single fixed-center hypersphere trained using the One-Class SVDD loss."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        epochs: int = 100,
        batch_size: int = 2048,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.encoder = DeepSVDDEncoder(input_dim, embed_dim).to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.center: torch.Tensor | None = None

    def _loader(self, X: np.ndarray, shuffle: bool) -> DataLoader:
        tensor = torch.as_tensor(np.asarray(X, dtype=np.float32))
        return DataLoader(
            TensorDataset(tensor),
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            pin_memory=self.device.type == "cuda",
        )

    @torch.no_grad()
    def initialize_center(self, X: np.ndarray) -> torch.Tensor:
        """Set c to the mean initial embedding; keep it fixed during training."""
        self.encoder.eval()
        embeddings = []
        for (xb,) in self._loader(X, shuffle=False):
            embeddings.append(self.encoder(xb.to(self.device, non_blocking=True)))
        center = torch.cat(embeddings, dim=0).mean(dim=0)
        # Avoid zero coordinates, which can permit degenerate constant mappings.
        eps = 1e-6
        center = torch.where(center.abs() < eps, torch.where(center < 0, -eps, eps), center)
        self.center = center.detach()
        return self.center

    def fit(
        self,
        X: np.ndarray,
        checkpoint_path: str,
        resume: bool = True,
    ) -> None:
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        start_epoch = 0
        ckpt_path = Path(checkpoint_path)

        if resume and ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            self.center = checkpoint["center"].to(self.device)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = int(checkpoint["epoch"])
            if start_epoch >= self.epochs:
                print(f"[Checkpoint] Training already complete ({start_epoch} epochs); reusing model.")
                return
            print(f"[Checkpoint] Resuming at epoch {start_epoch + 1}/{self.epochs}.")
        else:
            self.initialize_center(X)

        loader = self._loader(X, shuffle=True)
        self.encoder.train()
        for epoch in range(start_epoch, self.epochs):
            total_loss = 0.0
            total_rows = 0
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)
                z = self.encoder(xb)
                distances = ((z - self.center) ** 2).sum(dim=1)
                l2_penalty = sum(parameter.square().sum() for parameter in self.encoder.parameters())
                loss = distances.mean() + 0.5 * self.weight_decay * l2_penalty

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(xb)
                total_rows += len(xb)

            mean_loss = total_loss / max(total_rows, 1)
            done = epoch + 1
            print(f"  Epoch {done:3d}/{self.epochs}  one_class_loss={mean_loss:.7f}", flush=True)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": done,
                    "encoder_state_dict": self.encoder.state_dict(),
                    "center": self.center.detach().cpu(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "input_dim": self.encoder.net[0].in_features,
                    "embed_dim": self.encoder.net[-1].out_features,
                },
                ckpt_path,
            )

    @torch.no_grad()
    def decision_scores(self, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        if self.center is None:
            raise RuntimeError("Deep SVDD has no initialized center; train/load it first.")
        loader = DataLoader(
            TensorDataset(torch.as_tensor(np.asarray(X, dtype=np.float32))),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=self.device.type == "cuda",
        )
        self.encoder.eval()
        scores = []
        for (xb,) in loader:
            z = self.encoder(xb.to(self.device, non_blocking=True))
            scores.append(((z - self.center) ** 2).sum(dim=1).cpu().numpy())
        return np.concatenate(scores).astype(np.float64, copy=False)

    def save(self, path: str) -> None:
        if self.center is None:
            raise RuntimeError("Cannot save Deep SVDD before center initialization.")
        torch.save(
            {
                "encoder_state_dict": self.encoder.state_dict(),
                "center": self.center.detach().cpu(),
                "input_dim": self.encoder.net[0].in_features,
                "embed_dim": self.encoder.net[-1].out_features,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.center = checkpoint["center"].to(self.device)
