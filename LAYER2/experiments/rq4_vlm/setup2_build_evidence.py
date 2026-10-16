"""
Setup 2 (paper Section 4.5, Table tab:res-vlm-zeroday) — dung evidence cho VLM tren
loai tan cong CHUA TUNG THAY, theo tung fold leave-one-attack-out (Protocol A).

Moi fold giu lai loai h, moi thanh phan la cua RIENG fold do (nhu khi trien khai):
  Layer 1   : checkpoint ae_xgboost_mono cua fold, nguong tau_alpha (alpha = 0.4%)
  SHAP      : TreeSHAP chinh xac (xgboost pred_contribs) — cung cach build pool
  Retriever : R = loao_pools/<fold>/ (13 loai, khong co h), tau_zd cua fold
              (02_calibrate_novelty.py --protocol A)
Alert     : flow cua h trong test_zeroshot.csv co p >= tau_alpha.
Lay mau   : fold co >= MIN_ALERTS alert, bo 2 loai song sinh; moi fold PER_FOLD alert, chia
            deu nhom co co / khong co (thieu nhom nao thi bu tu nhom kia), seed co dinh.

Chi chay CPU, khong goi VLM. Output (bundle de upload len Kaggle, VLM chi doc bundle):
  output/rq4_vlm/setup2_bundle/
    manifest.jsonl      mot dong / alert: nhan that, Layer 1, novelty, co, 3 ung vien, anh
    fold_stats.json     moi fold (ke ca fold bi loai): n test, n alert, n co, ly do loai
    bundle_meta.json    cau hinh, nguong, phien ban code, thoi diem tao
    images/<alert_id>/  5 anh (SHAP waterfall + 4 bieu do luong)

CHAY:  python LAYER2/experiments/rq4_vlm/setup2_build_evidence.py
"""
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

paths.use_layer2_code()
import config     # noqa: E402
import pipeline   # noqa: E402
from shap_knn_retriever import SHAPKNNRetriever, LABEL_TO_PROFILE_ID  # noqa: E402

# Ham dung input / nap encoder dung chung voi luc build pool (cung khong gian SHAP)
_spec = importlib.util.spec_from_file_location(
    "build_loao_pools", Path(__file__).resolve().parents[1] / "rq3_retrieval" / "01_build_loao_pools.py")
blp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(blp)

VARIANT = "ae_xgboost_mono"
ALPHA = "0.004"
PER_FOLD = 10
MIN_ALERTS = 20
SEED = 42
OUT = paths.ensure(paths.OUT / "rq4_vlm" / "setup2_bundle")
PROFILES = {json.loads(p.read_text(encoding="utf-8"))["id"]: json.loads(p.read_text(encoding="utf-8"))
            for p in paths.PROFILES.glob("*.json")}


def tactics_of(label):
    """Tactic (va technique) cua mot loai, ke ca cac giai doan cua Bot/Infiltration."""
    doc = PROFILES[LABEL_TO_PROFILE_ID[label]]
    stages = doc.get("mitre_stages") or [{"tactic": doc["mitre_tactic"], "mitre_id": doc["mitre_id"]}]
    return sorted({s["tactic"] for s in stages}), sorted({s["mitre_id"] for s in stages})


def sample_alerts(flagged, rng):
    """Chon PER_FOLD chi so, chia deu co / khong co; thieu nhom nao bu tu nhom kia."""
    f_idx, u_idx = np.where(flagged)[0], np.where(~flagged)[0]
    half = PER_FOLD // 2
    nf, nu = min(half, len(f_idx)), min(half, len(u_idx))
    nf += min(PER_FOLD - nf - nu, len(f_idx) - nf)
    nu += min(PER_FOLD - nf - nu, len(u_idx) - nu)
    pick = list(rng.choice(f_idx, nf, replace=False)) + list(rng.choice(u_idx, nu, replace=False))
    return sorted(int(i) for i in pick)


def git_commit():
    """Commit cua repo code (LAYER2 trong repo, hoac UTS_URA canh workspace/LAYER2)."""
    for d in (paths.LAYER2, paths.WS / "UTS_URA"):
        try:
            c = subprocess.run(["git", "-C", str(d), "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True).stdout.strip()
            if c:
                return c
        except Exception:
            pass
    return None


def main():
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    ae_sc = joblib.load(paths.L1_ART / "ae_only/checkpoint/scaler.pkl")
    sv_sc = joblib.load(paths.L1_ART / "svdd_only/checkpoint/scaler.pkl")
    folds = sorted(p.name for p in paths.SPLITS.iterdir() if p.is_dir())
    ae = svdd = k_in = None
    manifest, stats = [], {}

    for fold in folds:
        meta = json.loads((paths.LOAO_POOLS / fold / f"shap_pool_{VARIANT}.meta.json").read_text(encoding="utf-8"))
        thr = float(meta["layer1_thresholds_by_alpha"][ALPHA])
        tau = (meta.get("novelty") or {}).get("tau_zd")
        assert tau is not None, f"{fold}: chua hieu chinh tau_zd (02_calibrate_novelty.py --protocol A)"

        zs = pd.read_csv(paths.SPLITS / fold / "test_zeroshot.csv")
        base = [c for c in zs.columns if c not in ("Label", "Label_Text")]
        assert set(zs["Label_Text"]) == {fold}, f"{fold}: test_zeroshot co nhan khac"
        if ae is None:
            ae, svdd, k_in = blp.load_encoders(len(base))
        X = zs[base].to_numpy("float32")
        model = joblib.load(paths.L1_ART / VARIANT / fold / "checkpoint" / "model.pkl")
        booster = model.get_booster()
        booster.set_param({"device": "cpu"})
        A = blp.build_input(VARIANT, fold, X, ae, svdd, ae_sc, sv_sc)
        p = model.predict_proba(A)[:, 1]
        alert_idx = np.where(p >= thr)[0]
        st = {"n_test": int(len(zs)), "n_alerts": int(len(alert_idx)), "tau_alpha": thr, "tau_zd": tau,
              "zs_acc": float(len(alert_idx) / len(zs)), "twin": fold in paths.TWINS}
        if fold in paths.TWINS or len(alert_idx) < MIN_ALERTS:
            st["excluded"] = "twin (failed attack, duplicate flows)" if fold in paths.TWINS \
                else f"fewer than {MIN_ALERTS} alerts"
            stats[fold] = st
            print(f"{fold:<26} bo qua: {st['excluded']} (alert = {len(alert_idx)})", flush=True)
            continue

        names = blp.feature_names(VARIANT, base, k_in)
        phi = booster.predict(xgb.DMatrix(A[alert_idx]), pred_contribs=True)[:, :-1].astype("float32")
        npz = paths.LOAO_POOLS / fold / f"shap_pool_{VARIANT}.npz"
        ret = SHAPKNNRetriever(str(npz), str(paths.PROFILES), tau_zd=tau, k=config.RAG_NOVELTY_K)
        assert fold not in set(ret.labels), f"{fold}: loai bi giu lai nam trong R"
        cand_idx, cand_dist, s = ret.score(phi)
        flagged = s > tau
        st.update(n_flagged=int(flagged.sum()), flag_rate=float(flagged.mean()))

        pick = sample_alerts(flagged, rng)
        ev = pipeline.EvidenceBuilder(ret)
        true_tac, true_tech = tactics_of(fold)
        for j in pick:
            i = int(alert_idx[j])
            aid = f"{fold}__{i}"
            img_dir = OUT / "images" / aid
            imgs, _ = ev.make_images(zs.iloc[i], phi[j], names, str(img_dir), float(p[i]), model_input=A[i])
            cands = [{"label": str(ret.labels[c]), "distance": float(d),
                      "profile_id": LABEL_TO_PROFILE_ID[str(ret.labels[c])]}
                     for c, d in zip(cand_idx[j], cand_dist[j])]
            top = np.argsort(-np.abs(phi[j]))[:5]
            manifest.append({
                "alert_id": aid, "fold": fold, "row_idx_test_zeroshot": i,
                "true_label": fold, "true_tactics": true_tac, "true_techniques": true_tech,
                "layer1_proba": float(p[i]), "layer1_threshold": thr,
                "novelty_score": float(s[j]), "tau_zd": tau, "flagged": bool(flagged[j]),
                "candidates": cands, "retriever_top1": cands[0]["label"],
                "retriever_top1_tactics": tactics_of(cands[0]["label"])[0],
                "shap_top_features": [{"name": names[k], "value": float(phi[j][k])} for k in top],
                "images": [str(Path(x).relative_to(OUT)).replace("\\", "/") for x in imgs],
            })
        st.update(n_sampled=len(pick), n_sampled_flagged=int(sum(flagged[j] for j in pick)))
        stats[fold] = st
        print(f"{fold:<26} alert {len(alert_idx):>6,} | co {flagged.mean():.3f} | lay {len(pick)} "
              f"({st['n_sampled_flagged']} co)", flush=True)

    with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "fold_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    (OUT / "bundle_meta.json").write_text(json.dumps({
        "setup": "Setup 2 (Protocol A, leave-one-attack-out)", "variant": VARIANT, "alpha": float(ALPHA),
        "per_fold": PER_FOLD, "min_alerts": MIN_ALERTS, "seed": SEED, "k": config.RAG_NOVELTY_K,
        "excluded_twins": sorted(paths.TWINS), "n_alerts": len(manifest),
        "code_commit": git_commit(), "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "build_seconds": round(time.time() - t_start, 1),
    }, indent=1), encoding="utf-8")
    print(f"\n{len(manifest)} alert -> {OUT}")


if __name__ == "__main__":
    main()
