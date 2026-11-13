"""
Bang doi chieu C (chi dung khi alert BI BAT CO "khong co tien le gan").

Y tuong (da chot): so alert voi TRUNG BINH cac tien le cung nhan thang trong lan tra cuu do
(cac lang gieng trong 5-NN mang nhan l*), dat SHAP len truoc:

    | ID | Feature | phi alert | phi trung binh tien le | delta phi | x alert | x tien le |

  - Dong duoc chon: hop cua top-5 |phi_alert| va top-5 |phi_alert - phi_ref|, sap theo |delta phi|,
    toi da 8 dong. 16 chieu ae_latent gop thanh 1 dong (SHAP cong tinh nen phi nhom = tong phi).
  - x tien le: cot lien tuc -> trung binh; cot roi rac (cong, giao thuc, co, Init Win co -1) -> cac gia
    tri kem so lan, vd "80 (3/4), 443 (1/4)". Danh sach cot roi rac lay tu KB (index.json).
  - x lay tu R (retrieval/store.py), khong suy nguoc tu SHAP.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

LATENT = "ae_latent"


def _groups(names):
    """[(ten dong, [chi so chieu])]: moi feature mot dong, rieng ae_latent_* gop lam mot."""
    out, latent = [], []
    for i, n in enumerate(names):
        if n.startswith(LATENT + "_"):
            latent.append(i)
        else:
            out.append((n, [i]))
    if latent:
        out.append((f"{LATENT} ({len(latent)} dims, summed)", latent))
    return out


def _fmt_counts(values):
    c = Counter(values.tolist())
    n = len(values)
    return ", ".join(f"{v:g} ({k}/{n})" for v, k in c.most_common())


def contrast_table(phi_q, x_q, store, winners, label, discrete, n_top=5, max_rows=8):
    """Bang C cho mot alert. x_q: {ten cot: gia tri tho} cua alert."""
    if store.x is None:
        raise ValueError("tap R chua co gia tri tho (x__): chay reference_set.py --add-raw")
    phi_q = np.asarray(phi_q, dtype="float64").ravel()
    winners = np.asarray(winners)
    phi_ref = store.phi[winners].mean(axis=0)
    base = store.base_feature_names
    col_of = {n: j for j, n in enumerate(base)}
    discrete = set(discrete)

    groups = _groups(store.feature_names)
    gq = np.array([phi_q[d].sum() for _, d in groups])
    gr = np.array([phi_ref[d].sum() for _, d in groups])
    delta = gq - gr
    pick = set(np.argsort(-np.abs(gq))[:n_top]) | set(np.argsort(-np.abs(delta))[:n_top])
    order = sorted(pick, key=lambda g: -abs(delta[g]))[:max_rows]

    rows = []
    for k, g in enumerate(order, 1):
        name = groups[g][0]
        # feature_key: ten dung de tra KB (retrieve.select contrast_features). Dong latent gop co ten hien thi
        # "ae_latent (16 dims, summed)" nhung khoa la "ae_latent" -> doan H1-ae_latent (checklist 2.3).
        key = LATENT if name.startswith(LATENT + " (") else name
        row = {"id": f"C{k}", "feature": name, "feature_key": key, "phi_alert": float(gq[g]),
               "phi_ref": float(gr[g]), "delta_phi": float(delta[g]), "x_alert": None, "x_ref": None}
        if name in col_of:
            vals = store.x[winners, col_of[name]]
            row["x_alert"] = x_q.get(name)
            row["x_ref"] = _fmt_counts(vals) if name in discrete else float(vals.mean())
        rows.append(row)
    return {"label": label, "n_precedents": int(len(winners)), "rows": rows}
