"""
Setup 2 (paper Section 4.5, Table tab:res-vlm-zeroday) — dung evidence cho VLM tren
loai tan cong CHUA TUNG THAY, theo tung fold leave-one-attack-out (Protocol A).

Moi fold giu lai loai h, moi thanh phan la cua RIENG fold do (nhu khi trien khai):
  Layer 1   : checkpoint ae_xgboost_mono cua fold (Layer1Engine.from_fold), nguong tau_alpha (alpha = 0.4%)
  SHAP      : TreeSHAP chinh xac — cung cach build tap R (gate/engine.py)
  Retriever : R = reference_sets/protocol_a/<fold>/ (13 loai, khong co h), tau_zd cua fold
              (02_calibrate_novelty.py --protocol A)
Alert     : flow cua h trong test_zeroshot.csv co p >= tau_alpha.
Lay mau   : fold co >= MIN_ALERTS alert, bo 2 loai song sinh; moi fold PER_FOLD alert, chia
            deu nhom co co / khong co (thieu nhom nao thi bu tu nhom kia), seed co dinh.

Chi chay CPU, khong goi VLM (chang A; dinh dang bundle o evidence/bundle.py). Output (bundle de upload
len Kaggle, VLM chi doc bundle):
  output/rq4_vlm/setup2_bundle/
    manifest.jsonl      mot dong / alert: nhan that, Layer 1, novelty, co, 3 ung vien, anh
    phi.npy             vector SHAP cua cac alert, cung thu tu voi manifest
    fold_stats.json     moi fold (ke ca fold bi loai): n test, n alert, n co, ly do loai
    bundle_meta.json    cau hinh, nguong, phien ban code, thoi diem tao
    images/<alert_id>/  5 anh (SHAP waterfall + 4 bieu do luong)

CHAY:  python LAYER2/experiments/rq4_vlm/setup2_build_evidence.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

paths.use_layer2_code()
import config  # noqa: E402
from evidence import bundle  # noqa: E402
from evidence.build import EvidenceBuilder, candidates, top_features  # noqa: E402
from knowledge_base.profiles import LABEL_TO_PROFILE_ID, load_profiles  # noqa: E402
from gate.engine import Layer1Engine  # noqa: E402
from retrieval import novelty, reference_set  # noqa: E402
from retrieval.retriever import SHAPKNNRetriever  # noqa: E402
from utils.io import git_commit  # noqa: E402

VARIANT = "ae_xgboost_mono"
ALPHA = "0.004"
PER_FOLD = 10
MIN_ALERTS = 20
SEED = 42
OUT = paths.ensure(paths.OUT / "rq4_vlm" / "setup2_bundle")
PROFILES = load_profiles(paths.PROFILES)


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


def main():
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    folds = sorted(p.name for p in paths.SPLITS.iterdir() if p.is_dir())
    manifest, stats, phis = [], {}, []

    for fold in folds:
        npz = reference_set.protocol_a_path(fold, VARIANT)
        meta = json.loads(novelty.meta_path(npz).read_text(encoding="utf-8"))
        thr = float(meta["layer1_thresholds_by_alpha"][ALPHA])
        tau = (meta.get("novelty") or {}).get("tau_zd")
        assert tau is not None, f"{fold}: chua hieu chinh tau_zd (02_calibrate_novelty.py --protocol A)"

        zs = pd.read_csv(paths.SPLITS / fold / "test_zeroshot.csv")
        base = [c for c in zs.columns if c not in ("Label", "Label_Text")]
        assert set(zs["Label_Text"]) == {fold}, f"{fold}: test_zeroshot co nhan khac"
        X = zs[base].to_numpy("float32")
        l1 = Layer1Engine.from_fold(VARIANT, fold, paths.L1_ART, base, threshold=thr, xgb_device="cpu")
        A = l1.build_input(X)
        p = l1.model.predict_proba(A)[:, 1]
        alert_idx = np.where(p >= thr)[0]
        st = {"n_test": int(len(zs)), "n_alerts": int(len(alert_idx)), "tau_alpha": thr, "tau_zd": tau,
              "zs_acc": float(len(alert_idx) / len(zs)), "twin": fold in paths.TWINS}
        # Hai loai song sinh duoc GIU lai: chung van du alert, va chi hoan vi lan
        # nhau — khong tung lam ung vien hang dau o bat ky fold nao khac (0/220
        # alert do duoc — nen khong lam nhieu cac fold con lai. Co "twin" o tren
        # danh dau chung de bao cao rieng trong Discussion.
        if len(alert_idx) < MIN_ALERTS:
            st["excluded"] = f"fewer than {MIN_ALERTS} alerts"
            stats[fold] = st
            print(f"{fold:<26} bo qua: {st['excluded']} (alert = {len(alert_idx)})", flush=True)
            continue

        names = l1.feature_names
        phi = l1.shap_values(A[alert_idx]).astype("float32")
        ret = SHAPKNNRetriever(str(npz), str(paths.PROFILES), tau_zd=tau, k=config.RAG_NOVELTY_K)
        assert fold not in set(ret.labels), f"{fold}: loai bi giu lai nam trong R"
        cand_idx, cand_dist, s = ret.score(phi)
        flagged = np.array([novelty.is_flagged(x, tau) for x in s], dtype=bool)
        st.update(n_flagged=int(flagged.sum()), flag_rate=float(flagged.mean()))

        pick = sample_alerts(flagged, rng)
        ev = EvidenceBuilder(ret)
        true_tac, true_tech = tactics_of(fold)
        for j in pick:
            i = int(alert_idx[j])
            aid = f"{fold}__{i}"
            img_dir = OUT / "images" / aid
            imgs, _ = ev.make_images(zs.iloc[i], phi[j], names, str(img_dir), float(p[i]), model_input=A[i])
            cands = candidates([ret.labels[c] for c in cand_idx[j]], cand_dist[j])
            manifest.append({
                "alert_id": aid, "fold": fold, "row_idx_test_zeroshot": i,
                "true_label": fold, "true_tactics": true_tac, "true_techniques": true_tech,
                "layer1_proba": float(p[i]), "layer1_threshold": thr,
                "novelty_score": float(s[j]), "tau_zd": tau, "flagged": bool(flagged[j]),
                "candidates": cands, "retriever_top1": cands[0]["label"],
                "retriever_top1_tactics": tactics_of(cands[0]["label"])[0],
                "shap_top_features": top_features(phi[j], names),
                "images": [bundle.rel(x, OUT) for x in imgs],
            })
            phis.append(phi[j])
        st.update(n_sampled=len(pick), n_sampled_flagged=int(sum(flagged[j] for j in pick)))
        stats[fold] = st
        print(f"{fold:<26} alert {len(alert_idx):>6,} | co {flagged.mean():.3f} | lay {len(pick)} "
              f"({st['n_sampled_flagged']} co)", flush=True)

    bundle.write(OUT, manifest, {
        "setup": "Setup 2 (Protocol A, leave-one-attack-out)", "variant": VARIANT, "alpha": float(ALPHA),
        "per_fold": PER_FOLD, "min_alerts": MIN_ALERTS, "seed": SEED, "k": config.RAG_NOVELTY_K,
        "twins_kept": sorted(paths.TWINS), "n_alerts": len(manifest),
        "code_commit": git_commit(paths.LAYER2, paths.WS / "UTS_URA"),
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "build_seconds": round(time.time() - t_start, 1),
    }, phi=np.array(phis) if phis else None, extra={"fold_stats.json": stats})
    print(f"\n{len(manifest)} alert -> {OUT}")


if __name__ == "__main__":
    main()
