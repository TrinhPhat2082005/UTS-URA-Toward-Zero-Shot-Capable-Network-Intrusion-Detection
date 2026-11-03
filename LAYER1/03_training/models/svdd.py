"""
SVDDEncoder + MixtureSVDD — copy nguyen kien truc tu
UTS_Layer1/SVDD_XGBoost/svdd_xgboost_ids.ipynb (Cell 6), KHONG doi logic.

MixtureSVDD: K sphere doc lap, moi sphere bao quanh 1 benign sub-cluster
(sub-cluster duoc dung K-Means chia truoc, xem 04_training/train_svdd_xgboost.py).
Inference: d_min = khoang cach nho nhat den K center — dung lam anomaly score.

Vai tro trong pipeline SVDD+XGBoost: SVDD CHI dong vai tro feature encoder —
K sphere distance + d_min duoc noi vao 78 feature goc lam input cho XGBoost.
XGBoost la diem quyet dinh DUY NHAT — khong co nguong SVDD rieng, khong OR-Fusion.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SVDDEncoder(nn.Module):
    """
    MLP Encoder: input_dim -> 256 -> 128 -> embed_dim
    Output L2-normalized embedding tren unit sphere.
    Dung LayerNorm thay BatchNorm de tranh loi khi batch_size=1.
    Khong dung bias o layer cuoi (chuan Deep SVDD).
    """

    def __init__(self, input_dim: int = 78, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embed_dim, bias=False),  # No bias — Deep SVDD rule
        )

    def forward(self, x):
        z = self.net(x)
        return nn.functional.normalize(z, dim=1)  # L2 normalize -> unit sphere


class MixtureSVDD:
    """
    K doc lap SVDD sphere, moi sphere bao quanh 1 benign sub-cluster.
    Inference: d_min = min distance den K centers.
    """

    def __init__(self, K: int = 5, input_dim: int = 78, embed_dim: int = 64,
                 nu: float = 0.1, lr: float = 1e-3, epochs: int = 30,
                 batch_size: int = 2048, device: str = "cpu"):
        self.K = K
        self.nu = nu  # Fraction of outliers tolerated (0.05-0.15)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # K encoders + K centers + K radii
        self.encoders = [SVDDEncoder(input_dim, embed_dim).to(device) for _ in range(K)]
        self.centers = [None] * K  # Se duoc khoi tao luc train
        self.radii = [None] * K

    def _init_center(self, encoder, loader):
        """Khoi tao center = trung binh embedding cua toan bo data."""
        encoder.eval()
        zs = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                zs.append(encoder(xb))
        c = torch.cat(zs).mean(0)
        # Tranh center bang 0 (collapse)
        c[(c.abs() < 1e-6)] = 1e-6
        return c.detach()

    def fit(self, X_clusters: list[np.ndarray]):
        """
        X_clusters: list gom K arrays, moi array la samples cua 1 sub-cluster.
        """
        for k in range(self.K):
            print(f"\n-- Training SVDD sphere {k + 1}/{self.K}  (n={len(X_clusters[k]):,}) --")
            X_k = torch.tensor(X_clusters[k], dtype=torch.float32)
            loader = DataLoader(
                TensorDataset(X_k), batch_size=self.batch_size, shuffle=True, drop_last=False
            )

            encoder = self.encoders[k]
            optimizer = optim.Adam(encoder.parameters(), lr=self.lr, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

            # Init center dung loader rieng KHONG drop_last
            init_loader = DataLoader(
                TensorDataset(X_k), batch_size=self.batch_size, shuffle=False, drop_last=False
            )
            c = self._init_center(encoder, init_loader)
            self.centers[k] = c

            best_loss = float("inf")
            for epoch in range(self.epochs):
                encoder.train()
                epoch_loss = 0.0
                for (xb,) in loader:
                    xb = xb.to(self.device)
                    z = encoder(xb)
                    # Soft-boundary Deep SVDD loss
                    dist = ((z - c) ** 2).sum(dim=1)
                    loss = dist.mean() + (1 / self.nu) * torch.relu(
                        dist - dist.quantile(1 - self.nu)
                    ).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                scheduler.step()
                avg = epoch_loss / len(loader)
                if avg < best_loss:
                    best_loss = avg
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch + 1:3d}/{self.epochs}  loss={avg:.6f}")

            # Radius = quantile (1-nu) cua distances tren training data
            radius_loader = DataLoader(
                TensorDataset(X_k), batch_size=self.batch_size, shuffle=False, drop_last=False
            )
            encoder.eval()
            dists = []
            with torch.no_grad():
                for (xb,) in radius_loader:
                    xb = xb.to(self.device)
                    z = encoder(xb)
                    dists.append(((z - c) ** 2).sum(dim=1))
            all_dists = torch.cat(dists)
            self.radii[k] = torch.quantile(all_dists, 1 - self.nu).item()
            print(f"  Radius R_{k} = {self.radii[k]:.6f}")

    def all_sphere_distances(self, X_np: np.ndarray) -> np.ndarray:
        """
        Tra ve ma tran [n_samples, K]: khoang cach binh phuong den TUNG center.
        Dung lam K feature bo sung cho XGBoost (thay vi chi 1 d_min tong hop).
        """
        X_t = torch.tensor(X_np, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t), batch_size=4096, shuffle=False)
        all_dists = []
        for k in range(self.K):
            self.encoders[k].eval()
            c = self.centers[k]
            dists_k = []
            with torch.no_grad():
                for (xb,) in loader:
                    xb = xb.to(self.device)
                    z = self.encoders[k](xb)
                    dists_k.append(((z - c) ** 2).sum(dim=1).cpu().numpy())
            all_dists.append(np.concatenate(dists_k))
        return np.stack(all_dists, axis=1)  # [n_samples, K]

    def decision_scores(self, X_np: np.ndarray) -> np.ndarray:
        """
        Tra ve d_min: khoang cach nho nhat den K centers.
        Score cao -> anomaly, score thap -> benign.
        """
        return self.all_sphere_distances(X_np).min(axis=1)

    def predict(self, X_np: np.ndarray) -> np.ndarray:
        """Binary prediction dua tren R (radius) cua sphere gan nhat."""
        X_t = torch.tensor(X_np, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t), batch_size=4096, shuffle=False)
        pred_flags = np.ones(len(X_np), dtype=int)  # Default = anomaly
        for k in range(self.K):
            self.encoders[k].eval()
            c = self.centers[k]
            dists_k = []
            with torch.no_grad():
                for (xb,) in loader:
                    xb = xb.to(self.device)
                    z = self.encoders[k](xb)
                    dists_k.append(((z - c) ** 2).sum(dim=1).cpu().numpy())
            d = np.concatenate(dists_k)
            # Neu nam trong sphere k -> benign
            pred_flags[d <= self.radii[k]] = 0
        return pred_flags

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        for k, enc in enumerate(self.encoders):
            torch.save(enc.state_dict(), f"{path}/encoder_{k}.pt")
        np.save(f"{path}/centers.npy", [c.cpu().numpy() for c in self.centers])
        np.save(f"{path}/radii.npy", self.radii)
        print(f"SVDD saved to '{path}/'")

    def load(self, path: str):
        centers_np = np.load(f"{path}/centers.npy", allow_pickle=True)
        self.radii = list(np.load(f"{path}/radii.npy", allow_pickle=True))
        for k, enc in enumerate(self.encoders):
            enc.load_state_dict(torch.load(f"{path}/encoder_{k}.pt", map_location=self.device))
            self.centers[k] = torch.tensor(centers_np[k], dtype=torch.float32).to(self.device)
        print(f"SVDD loaded from '{path}/'")
