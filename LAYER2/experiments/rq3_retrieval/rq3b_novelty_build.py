"""R3 voi pool DAY DU (khop pipeline): moi round, pool = TOAN BO attack trong train_benign_attack.csv
(13 loai giu lai, khong benign, khong cap). Query = TOAN BO alert (known: attack cua val_binary;
unknown: test_zeroshot) o nguong thap nhat trong BUDGETS; loc lai theo nguong cao hon o buoc cham.
Khong luu vector (qua lon) - luu diem novelty kNN-10 L2 cho 3 khoa (SHAP day du / SHAP 78 / feature tho),
nhan lang gieng top-1 (cosine, SHAP day du va feature tho) va khoang cach toi flow gan nhat (feature tho).
Chi doc checkpoint + CSV; ghi r3_full/<variant>__<label>.npz. Khong sua gi trong project."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402  (experiments/paths.py)
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, joblib, torch, xgboost as xgb

paths.use_layer1_code()
from models.autoencoder import BenignAutoencoder
from models.svdd import MixtureSVDD

SPL = paths.SPLITS
ART = paths.L1_ART
HERE = paths.ensure(paths.OUT_RQ3B)
OUT = HERE / "r3_full"
OUT.mkdir(exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = sorted(p.name for p in SPL.iterdir() if p.is_dir())
VARS = ["xgboost_only", "ae_xgboost_mono", "svdd_xgboost_mono"]
BUDGETS = [0.004, 0.015]
CLIP, K = 20.0, 10
TB = np.load(paths.OUT_L1 / "sweep_train_scores.npz")  # tu rq1_rq2_layer1/02

ae_sc = joblib.load(ART / "ae_only/checkpoint/scaler.pkl")
sv_sc = joblib.load(ART / "svdd_only/checkpoint/scaler.pkl")
k_in = len(list((ART / "svdd_only/checkpoint/svdd_model").glob("encoder_*.pt")))
ae = svdd = None
T = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=DEV)


def ae_aug(Xs):
    zs, es = [], []
    with torch.no_grad():
        for i in range(0, len(Xs), 8192):
            xb = torch.tensor(Xs[i:i+8192], device=DEV)
            xh, z = ae(xb)
            zs.append(z.cpu().numpy()); es.append(((xb - xh) ** 2).mean(1).cpu().numpy())
    return np.concatenate([Xs, np.concatenate(zs), np.concatenate(es)[:, None]], 1)


def build_input(var, lab, X):
    if var == "xgboost_only":
        return joblib.load(ART / var / lab / "checkpoint" / "scaler.pkl").transform(X)
    if var == "ae_xgboost_mono":
        return ae_aug(np.clip(ae_sc.transform(X), -CLIP, CLIP).astype("float32"))
    Xs = np.clip(sv_sc.transform(X), -CLIP, CLIP)
    d = svdd.all_sphere_distances(Xs)
    return np.concatenate([Xs, d, d.min(1, keepdims=True)], 1)


def knn_mean(P, Q, k=K, chunk=1024):
    if len(Q) == 0:
        return np.zeros(0, dtype="float32")
    Pt = T(P)
    return np.concatenate([torch.cdist(T(Q[i:i+chunk]), Pt).topk(k, largest=False).values.mean(1).cpu().numpy()
                           for i in range(0, len(Q), chunk)])


def nn_min(P, Q, chunk=1024):
    if len(Q) == 0:
        return np.zeros(0, dtype="float32")
    Pt = T(P)
    return np.concatenate([torch.cdist(T(Q[i:i+chunk]), Pt).min(1).values.cpu().numpy() for i in range(0, len(Q), chunk)])


def cos_top1(P, yp, Q, chunk=2048):
    if len(Q) == 0:
        return np.array([], dtype=object)
    Pn = torch.nn.functional.normalize(T(P), dim=1)
    idx = [torch.nn.functional.normalize(T(Q[i:i+chunk]), dim=1).matmul(Pn.T).argmax(1).cpu().numpy()
           for i in range(0, len(Q), chunk)]
    return yp[np.concatenate(idx)]


for lab in LABELS:
    t0 = time.time()
    if all((OUT / f"{v}__{lab}.npz").exists() for v in VARS):
        continue                                                    # da xong o lan chay truoc
    tr = pd.read_csv(SPL / lab / "train_benign_attack.csv")
    feats = [c for c in tr.columns if c not in ("Label", "Label_Text")]
    tr = tr[tr["Label"] == 1].reset_index(drop=True)                # pool = toan bo attack cua train
    va = pd.read_csv(SPL / lab / "val_binary.csv")
    va = va[va["Label"] == 1].reset_index(drop=True)
    zs = pd.read_csv(SPL / lab / "test_zeroshot.csv")
    assert lab not in set(tr["Label_Text"]) and lab not in set(va["Label_Text"]), "nhan bi giu lai lot vao pool/known"
    if ae is None:
        ae = BenignAutoencoder(input_dim=len(feats), bottleneck_dim=16).to(DEV)
        ae.load_state_dict(torch.load(ART / "ae_only/checkpoint/ae_best.pt", map_location=DEV)); ae.eval()
        svdd = MixtureSVDD(K=k_in, input_dim=len(feats), embed_dim=64, device=DEV)
        svdd.load(str(ART / "svdd_only/checkpoint/svdd_model"))

    Xtr, Xva, Xzs = (d[feats].values.astype("float32") for d in (tr, va, zs))
    ytr, yva = tr["Label_Text"].values, va["Label_Text"].values
    raw = lambda X: np.clip(ae_sc.transform(X), -CLIP, CLIP).astype("float32")
    R_pool = raw(Xtr)

    for var in VARS:
        if (OUT / f"{var}__{lab}.npz").exists():
            continue
        model = joblib.load(ART / var / lab / "checkpoint" / "model.pkl")
        booster = model.get_booster(); booster.set_param({"device": DEV.type})
        thr_low = min(np.quantile(TB[f"{var}|{lab}|tb"], 1 - b, method="higher") for b in BUDGETS)
        A_va, A_zs = build_input(var, lab, Xva), build_input(var, lab, Xzs)
        p_va, p_zs = model.predict_proba(A_va)[:, 1], model.predict_proba(A_zs)[:, 1]
        k_idx, u_idx = np.where(p_va >= thr_low)[0], np.where(p_zs >= thr_low)[0]
        contrib = lambda A: booster.predict(xgb.DMatrix(A), pred_contribs=True)[:, :-1].astype("float32")
        S_pool = contrib(build_input(var, lab, Xtr))
        S_k = contrib(A_va[k_idx]) if len(k_idx) else np.zeros((0, S_pool.shape[1]), "float32")
        S_u = contrib(A_zs[u_idx]) if len(u_idx) else np.zeros((0, S_pool.shape[1]), "float32")
        R_k = raw(Xva[k_idx]) if len(k_idx) else np.zeros((0, len(feats)), "float32")
        R_u = raw(Xzs[u_idx]) if len(u_idx) else np.zeros((0, len(feats)), "float32")
        nb = len(feats)
        np.savez(OUT / f"{var}__{lab}.npz",
                 p_known=p_va[k_idx], p_unk=p_zs[u_idx], y_known=yva[k_idx],
                 s_full_known=knn_mean(S_pool, S_k), s_full_unk=knn_mean(S_pool, S_u),
                 s_b78_known=knn_mean(S_pool[:, :nb], S_k[:, :nb]), s_b78_unk=knn_mean(S_pool[:, :nb], S_u[:, :nb]),
                 s_raw_known=knn_mean(R_pool, R_k), s_raw_unk=knn_mean(R_pool, R_u),
                 nb_full_unk=cos_top1(S_pool, ytr, S_u), nb_raw_unk=cos_top1(R_pool, ytr, R_u),
                 dmin_known=nn_min(R_pool, R_k), dmin_unk=nn_min(R_pool, R_u),
                 thr_budgets=np.array(BUDGETS),
                 thr_values=np.array([np.quantile(TB[f"{var}|{lab}|tb"], 1 - b, method="higher") for b in BUDGETS]),
                 n_pool=len(S_pool), n_pool_per_type=np.array([f"{t}:{(ytr == t).sum()}" for t in np.unique(ytr)]))
        print(f"{lab:<26} {var:<18} pool={len(S_pool):,} known_alerts={len(k_idx):,} unk_alerts={len(u_idx):,}", flush=True)
        torch.cuda.empty_cache()
    print(f"  ({time.time() - t0:.0f}s)", flush=True)
print("DONE")
