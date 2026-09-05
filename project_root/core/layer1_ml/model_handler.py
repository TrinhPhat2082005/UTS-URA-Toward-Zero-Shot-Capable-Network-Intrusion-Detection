import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

BASE_MODEL_FEATURES = [
    'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts',
    'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
    'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max',
    'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s',
    'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s',
    'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var',
    'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt',
    'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt',
    'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
    'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
    'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts',
    'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts',
    'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

RENAME_MAP = {
    # Metadata columns
    'src_ip': 'Src IP', 'dst_ip': 'Dst IP',
    'src_port': 'Src Port', 'dst_port': 'Dst Port',
    'protocol': 'Protocol', 'timestamp': 'Timestamp',
    # Core flow features
    'flow_duration': 'Flow Duration',
    'tot_fwd_pkts': 'Tot Fwd Pkts', 'tot_bwd_pkts': 'Tot Bwd Pkts',
    'totlen_fwd_pkts': 'TotLen Fwd Pkts', 'totlen_bwd_pkts': 'TotLen Bwd Pkts',
    # Packet length stats
    'fwd_pkt_len_max': 'Fwd Pkt Len Max', 'fwd_pkt_len_min': 'Fwd Pkt Len Min',
    'fwd_pkt_len_mean': 'Fwd Pkt Len Mean', 'fwd_pkt_len_std': 'Fwd Pkt Len Std',
    'bwd_pkt_len_max': 'Bwd Pkt Len Max', 'bwd_pkt_len_min': 'Bwd Pkt Len Min',
    'bwd_pkt_len_mean': 'Bwd Pkt Len Mean', 'bwd_pkt_len_std': 'Bwd Pkt Len Std',
    'pkt_len_max': 'Pkt Len Max', 'pkt_len_min': 'Pkt Len Min',
    'pkt_len_mean': 'Pkt Len Mean', 'pkt_len_std': 'Pkt Len Std',
    'pkt_len_var': 'Pkt Len Var', 'pkt_size_avg': 'Pkt Size Avg',
    # Flow rate
    'flow_byts_s': 'Flow Byts/s', 'flow_pkts_s': 'Flow Pkts/s',
    'fwd_pkts_s': 'Fwd Pkts/s', 'bwd_pkts_s': 'Bwd Pkts/s',
    # IAT (Inter-Arrival Time)
    'flow_iat_mean': 'Flow IAT Mean', 'flow_iat_std': 'Flow IAT Std',
    'flow_iat_max': 'Flow IAT Max', 'flow_iat_min': 'Flow IAT Min',
    'fwd_iat_tot': 'Fwd IAT Tot', 'fwd_iat_mean': 'Fwd IAT Mean',
    'fwd_iat_std': 'Fwd IAT Std', 'fwd_iat_max': 'Fwd IAT Max',
    'fwd_iat_min': 'Fwd IAT Min',
    'bwd_iat_tot': 'Bwd IAT Tot', 'bwd_iat_mean': 'Bwd IAT Mean',
    'bwd_iat_std': 'Bwd IAT Std', 'bwd_iat_max': 'Bwd IAT Max',
    'bwd_iat_min': 'Bwd IAT Min',
    # Flags
    'fwd_psh_flags': 'Fwd PSH Flags', 'bwd_psh_flags': 'Bwd PSH Flags',
    'fwd_urg_flags': 'Fwd URG Flags', 'bwd_urg_flags': 'Bwd URG Flags',
    'fin_flag_cnt': 'FIN Flag Cnt', 'syn_flag_cnt': 'SYN Flag Cnt',
    'rst_flag_cnt': 'RST Flag Cnt', 'psh_flag_cnt': 'PSH Flag Cnt',
    'ack_flag_cnt': 'ACK Flag Cnt', 'urg_flag_cnt': 'URG Flag Cnt',
    'ece_flag_cnt': 'ECE Flag Cnt', 'cwr_flag_count': 'CWE Flag Count',
    # Header / Segment
    'fwd_header_len': 'Fwd Header Len', 'bwd_header_len': 'Bwd Header Len',
    'fwd_seg_size_avg': 'Fwd Seg Size Avg', 'bwd_seg_size_avg': 'Bwd Seg Size Avg',
    'fwd_seg_size_min': 'Fwd Seg Size Min',
    # Bulk
    'fwd_byts_b_avg': 'Fwd Byts/b Avg', 'fwd_pkts_b_avg': 'Fwd Pkts/b Avg',
    'fwd_blk_rate_avg': 'Fwd Blk Rate Avg',
    'bwd_byts_b_avg': 'Bwd Byts/b Avg', 'bwd_pkts_b_avg': 'Bwd Pkts/b Avg',
    'bwd_blk_rate_avg': 'Bwd Blk Rate Avg',
    # Subflow
    'subflow_fwd_pkts': 'Subflow Fwd Pkts', 'subflow_fwd_byts': 'Subflow Fwd Byts',
    'subflow_bwd_pkts': 'Subflow Bwd Pkts', 'subflow_bwd_byts': 'Subflow Bwd Byts',
    # Window / Active / Idle
    'init_fwd_win_byts': 'Init Fwd Win Byts', 'init_bwd_win_byts': 'Init Bwd Win Byts',
    'fwd_act_data_pkts': 'Fwd Act Data Pkts',
    'active_mean': 'Active Mean', 'active_std': 'Active Std',
    'active_max': 'Active Max', 'active_min': 'Active Min',
    'idle_mean': 'Idle Mean', 'idle_std': 'Idle Std',
    'idle_max': 'Idle Max', 'idle_min': 'Idle Min',
    'down_up_ratio': 'Down/Up Ratio',
}


class _BenignAutoencoder:
    """Lazy Torch wrapper — module import stays inside __init__ (see CLAUDE.md:
    lazy imports for heavy components to avoid Windows DLL crash)."""

    def __init__(self, checkpoint_path, input_dim, bottleneck_dim=16, device="cpu"):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(32, bottleneck_dim), nn.LayerNorm(bottleneck_dim), nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(bottleneck_dim, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(32, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(64, input_dim),
                )

            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z), z

        self._torch = torch
        self.device = device
        self.model = Net().to(device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.eval()

    def encode_and_score(self, X_np):
        """Returns (latent[n, bottleneck_dim], recon_error[n])."""
        torch = self._torch
        with torch.no_grad():
            x = torch.tensor(X_np, dtype=torch.float32, device=self.device)
            x_hat, z = self.model(x)
            err = ((x - x_hat) ** 2).mean(dim=1)
        return z.cpu().numpy(), err.cpu().numpy()


class _MixtureSVDDInference:
    """Lazy Torch wrapper for K trained SVDD spheres — inference-only (no fit)."""

    def __init__(self, artifacts_dir, input_dim, embed_dim=64, device="cpu"):
        import torch
        import torch.nn as nn

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(128, embed_dim, bias=False),
                )

            def forward(self, x):
                return nn.functional.normalize(self.net(x), dim=1)

        self._torch = torch
        self.device = device

        centers_np = np.load(os.path.join(artifacts_dir, "centers.npy"), allow_pickle=True)
        self.radii = list(np.load(os.path.join(artifacts_dir, "radii.npy"), allow_pickle=True))
        self.K = len(centers_np)

        self.encoders = []
        self.centers = []
        for k in range(self.K):
            enc = Encoder().to(device)
            enc.load_state_dict(torch.load(
                os.path.join(artifacts_dir, f"encoder_{k}.pt"), map_location=device))
            enc.eval()
            self.encoders.append(enc)
            self.centers.append(torch.tensor(centers_np[k], dtype=torch.float32, device=device))

    def all_sphere_distances(self, X_np):
        """Returns [n_samples, K]: squared distance to EACH benign sphere.

        Used as K feature columns for XGBoost (not just a single d_min summary).
        """
        torch = self._torch
        with torch.no_grad():
            x = torch.tensor(X_np, dtype=torch.float32, device=self.device)
            dists = []
            for k in range(self.K):
                z = self.encoders[k](x)
                dists.append(((z - self.centers[k]) ** 2).sum(dim=1).cpu().numpy())
        return np.stack(dists, axis=1)  # [n_samples, K]

    def decision_scores(self, X_np):
        """Returns d_min: distance to the nearest of K benign spheres. High = anomaly."""
        return self.all_sphere_distances(X_np).min(axis=1)


class Layer1Inference:
    """Layer 1 anomaly gate.

    variant='xgboost': plain XGBoost binary gate.
    variant='ae':      AE-augmented XGBoost — reconstruction error + latent
                       code from a benign-only autoencoder concatenated onto
                       the 78 raw features before XGBoost predicts.
    variant='svdd':    SVDD-augmented XGBoost — squared distance to each of
                       the K benign spheres (+ d_min) concatenated onto the
                       78 raw features before XGBoost predicts.

    In every variant, `self.bst` is the underlying XGBoost model, so
    `IDSExplainer(l1_handler.bst, l1_handler.model_features)` keeps working
    unchanged — TreeExplainer only ever sees XGBoost.
    """

    def __init__(self, my_ip, variant="xgboost", device=None):
        self.my_ip = my_ip
        self.variant = variant
        self.device = device or "cpu"

        self.metadata_columns = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Timestamp', 'Peer_IP']

        if variant == "xgboost":
            self._init_xgboost_only()
        elif variant == "ae":
            self._init_ae()
        elif variant == "svdd":
            self._init_svdd()
        else:
            raise ValueError(f"Unknown Layer1Inference variant: {variant!r} (expected 'xgboost', 'ae', or 'svdd')")

        self.all_columns = self.model_features + self.metadata_columns

    def _init_xgboost_only(self):
        """Loads the XGBoost_only checkpoint + the scaler it was trained on
        (splits were fit on scaled input, so it must be re-applied at inference)."""
        artifacts_dir = os.path.join(ARTIFACTS_DIR, "xgboost_only")
        self.scaler = joblib.load(os.path.join(artifacts_dir, "scaler_xgb_only_v1_infilteration.pkl"))
        self.bst = joblib.load(os.path.join(artifacts_dir, "xgb_only_model_v1_infilteration.pkl"))  # XGBClassifier
        print(f"[Layer 1] Loaded XGBoost-only model: {artifacts_dir}")

        self.model_features = list(BASE_MODEL_FEATURES)
        self.threshold = 0.7

    def _init_ae(self, bottleneck_dim=16, clip_val=20.0, threshold=0.7):
        """AE is a feature encoder only: latent code + reconstruction error are
        concatenated onto the 78 raw features, and XGBoost alone decides
        (single threshold, no separate AE branch/OR-fusion)."""
        artifacts_dir = os.path.join(ARTIFACTS_DIR, "ae")
        self.scaler = joblib.load(os.path.join(artifacts_dir, "scaler_ae.pkl"))
        self.bst = joblib.load(os.path.join(artifacts_dir, "xgb_aug_model.pkl"))  # XGBClassifier

        self.ae = _BenignAutoencoder(
            os.path.join(artifacts_dir, "ae_best.pt"),
            input_dim=len(BASE_MODEL_FEATURES),
            bottleneck_dim=bottleneck_dim,
            device=self.device,
        )
        self.clip_val = clip_val
        self.threshold = threshold

        latent_features = [f"ae_latent_{i}" for i in range(bottleneck_dim)]
        self.model_features = list(BASE_MODEL_FEATURES) + latent_features + ["ae_recon_error"]
        print(f"[Layer 1] Loaded AE-augmented XGBoost (threshold={self.threshold:.2f})")

    def _init_svdd(self, clip_val=20.0, threshold=0.7):
        """SVDD is a feature encoder only: squared distance to each of the K
        benign spheres + d_min are concatenated onto the 78 raw features, and
        XGBoost alone decides (same philosophy as the 'ae' variant)."""
        artifacts_dir = os.path.join(ARTIFACTS_DIR, "svdd")
        self.scaler = joblib.load(os.path.join(artifacts_dir, "scaler_svdd.pkl"))
        self.bst = joblib.load(os.path.join(artifacts_dir, "xgb_svdd_model.pkl"))  # XGBClassifier

        self.svdd = _MixtureSVDDInference(
            artifacts_dir, input_dim=len(BASE_MODEL_FEATURES), device=self.device)
        self.clip_val = clip_val
        self.threshold = threshold

        expected_features = len(BASE_MODEL_FEATURES) + self.svdd.K + 1
        trained_features = getattr(self.bst, "n_features_in_", expected_features)
        if trained_features != expected_features:
            raise ValueError(
                f"[Layer 1] xgb_svdd_model.pkl was trained on {trained_features} features, "
                f"but this SVDD checkpoint has K={self.svdd.K} spheres, expecting "
                f"{expected_features} (78 raw + {self.svdd.K} sphere distances + d_min). "
                "Retrain XGBoost on the SVDD-augmented features (see "
                "UTS_Layer1/SVDD_XGBoost/svdd_xgboost_ids.ipynb) and re-copy the artifact."
            )

        sphere_features = [f"svdd_dist_{k}" for k in range(self.svdd.K)]
        self.model_features = list(BASE_MODEL_FEATURES) + sphere_features + ["svdd_dmin"]
        print(f"[Layer 1] Loaded SVDD-augmented XGBoost (threshold={self.threshold:.2f})")



    def _optimize_data_types(self, df):
        for col in BASE_MODEL_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df

    def process_cicflowmeter_csv(self, csv_path):
        """Processes CSV from the Python cicflowmeter package (snake_case
        columns) into the Title Case names the XGBoost model expects."""
        try:
            if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
                return pd.DataFrame()

            df = pd.read_csv(csv_path)
            if df.empty:
                return pd.DataFrame()

            # CICFlowMeter columns often have leading/trailing spaces
            df.columns = df.columns.str.strip()
            df.rename(columns=RENAME_MAP, inplace=True)

            # Ensure Peer_IP exists for metadata
            if 'Peer_IP' not in df.columns:
                if 'Src IP' in df.columns and 'Dst IP' in df.columns:
                    df['Peer_IP'] = df.apply(
                        lambda row: row['Dst IP'] if str(row.get('Src IP', '')) == self.my_ip else row['Src IP'], axis=1
                    )
                else:
                    df['Peer_IP'] = 'unknown'

            if 'Flow ID' not in df.columns:
                if all(c in df.columns for c in ['Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol']):
                    df['Flow ID'] = df.apply(
                        lambda r: f"{r['Src IP']}-{r['Dst IP']}-{r['Src Port']}-{r['Dst Port']}-{r['Protocol']}", axis=1
                    )
                else:
                    df['Flow ID'] = range(len(df))

            # Fill missing base features with 0 (augmented ae_*/recon columns are
            # computed later in run_inference, not expected from CICFlowMeter)
            for col in BASE_MODEL_FEATURES:
                if col not in df.columns:
                    df[col] = 0

            print(f"[ML] Loaded {len(df)} flows from CICFlowMeter CSV.")
            return self._optimize_data_types(df)
        except Exception as e:
            print(f"[Error] Failed to process CICFlowMeter CSV: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()



    def run_inference(self, ids_df):
        """Runs the configured Layer 1 variant. Returns (preds, preds_prob).

        `ids_df` is mutated in place: BASE_MODEL_FEATURES are overwritten with
        their scaled/clipped values, and any augmented columns are added, so
        IDSExplainer sees the exact same feature values XGBoost used —
        keeping SHAP TreeExplainer exact.
        """
        if ids_df.empty:
            return None, None

        if self.variant == "xgboost":
            return self._run_xgboost_only(ids_df)
        elif self.variant == "ae":
            return self._run_ae_augmented(ids_df)
        elif self.variant == "svdd":
            return self._run_svdd_augmented(ids_df)
        raise ValueError(f"Unknown variant: {self.variant!r}")

    def _run_xgboost_only(self, ids_df):
        X_raw = ids_df[BASE_MODEL_FEATURES].astype('float32').values
        X_raw = np.where(np.isfinite(X_raw), X_raw, 0.0)
        X_s = self.scaler.transform(X_raw)

        preds_prob = self.bst.predict_proba(X_s)[:, 1]
        preds = (preds_prob > self.threshold).astype(int)

        ids_df[BASE_MODEL_FEATURES] = X_s

        return preds, preds_prob

    def _prep_scaled(self, ids_df):
        X_raw = ids_df[BASE_MODEL_FEATURES].astype('float32').values
        X_raw = np.where(np.isfinite(X_raw), X_raw, 0.0)
        X_s = self.scaler.transform(X_raw)
        return np.clip(X_s, -self.clip_val, self.clip_val)

    def _run_ae_augmented(self, ids_df):
        """AE only encodes (latent + recon_error); XGBoost alone decides."""
        X_s = self._prep_scaled(ids_df)

        latent, recon_error = self.ae.encode_and_score(X_s)
        bottleneck_dim = latent.shape[1]
        for i in range(bottleneck_dim):
            ids_df[f"ae_latent_{i}"] = latent[:, i]
        ids_df["ae_recon_error"] = recon_error
        ids_df[BASE_MODEL_FEATURES] = X_s

        X_aug = np.concatenate([X_s, latent, recon_error.reshape(-1, 1)], axis=1)
        xgb_proba = self.bst.predict_proba(X_aug)[:, 1]
        preds = (xgb_proba > self.threshold).astype(int)

        return preds, xgb_proba

    def _run_svdd_augmented(self, ids_df):
        """SVDD only encodes (K sphere distances + d_min); XGBoost alone decides."""
        X_s = self._prep_scaled(ids_df)

        sphere_dists = self.svdd.all_sphere_distances(X_s)  # [n, K]
        d_min = sphere_dists.min(axis=1)
        for k in range(sphere_dists.shape[1]):
            ids_df[f"svdd_dist_{k}"] = sphere_dists[:, k]
        ids_df["svdd_dmin"] = d_min
        ids_df[BASE_MODEL_FEATURES] = X_s

        X_aug = np.concatenate([X_s, sphere_dists, d_min.reshape(-1, 1)], axis=1)
        xgb_proba = self.bst.predict_proba(X_aug)[:, 1]
        preds = (xgb_proba > self.threshold).astype(int)

        return preds, xgb_proba
