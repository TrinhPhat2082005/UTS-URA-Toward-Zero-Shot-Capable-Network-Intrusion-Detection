"""
CSV cua CICFlowMeter (ban Python, ten cot snake_case) -> DataFrame co ten cot giong
CSE-CIC-IDS2018 (Title Case) ma Layer 1 cua LAYER2 da hoc.

Chi doi ten va bo sung cot thieu; scale/clip/augment do gate.engine.Layer1Engine (LAYER2) lam.
NaN/Inf de nguyen: Layer1Engine.preprocess() thay bang 0 truoc khi scale.
"""
import os

import numpy as np
import pandas as pd

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


def load_cicflowmeter_csv(csv_path, base_features):
    """Doc CSV cua CICFlowMeter, doi ten cot, them cot feature con thieu (= 0).

    Args:
        csv_path: file CSV ma LiveSniffer vua sinh.
        base_features: 78 ten feature goc theo dung thu tu Layer 1 da hoc
            (Layer1Engine.feature_names[:Layer1Engine.n_base]).

    Returns:
        DataFrame (rong neu file khong co flow nao).
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    df.columns = df.columns.str.strip()
    df = df.rename(columns=RENAME_MAP)
    missing = [c for c in base_features if c not in df.columns]
    for c in missing:
        df[c] = 0.0
    if missing:
        print(f"[Flow] {len(missing)} feature khong co trong CSV, dat = 0: {missing[:5]}"
              + (" ..." if len(missing) > 5 else ""))
    df[base_features] = df[base_features].apply(pd.to_numeric, errors="coerce").astype("float64")
    df[base_features] = df[base_features].replace([np.inf, -np.inf], np.nan)
    return df
