"""Setup 1 (Protocol B) — dung evidence cho VLM tren cac loai tan cong DA BIET.

Khac Setup 2 o cho moi alert deu thuoc mot trong 14 loai co trong tap tham chieu R,
nen nhan dung LUON nam trong danh sach lua chon. Cau hoi do o day la: khi bi ep chon
trong ba ung vien cua retriever, VLM co chon dung hon chinh ung vien gan nhat khong.

Lay dung 140 alert ma cac lan chay truoc da dung (10 moi loai, deu la true positive),
doc tu mot file ket qua cu, de con so moi so sanh truc tiep duoc voi chung.

Moi variant Layer 1 dung mot bundle rieng: AE va SVDD co khong gian attribution khac
nhau (95 va 90 chieu), nen bieu do SHAP va ten feature khong dung chung duoc.

Chi chay CPU, khong goi VLM (chang A; dinh dang bundle o evidence/bundle.py). Output moi variant:
  output/rq4_vlm/setup1_bundle/<variant>/
    manifest.jsonl      mot dong / alert: nhan that, Layer 1, 3 ung vien, anh
    phi.npy             vector SHAP cua cac alert, cung thu tu voi manifest
    bundle_meta.json    cau hinh, phien ban code, thoi diem tao
    images/<alert_id>/  5 anh (SHAP waterfall + 4 bieu do luong)

CHAY:  python LAYER2/experiments/rq4_vlm/setup1_build_evidence.py
       python LAYER2/experiments/rq4_vlm/setup1_build_evidence.py ae_xgboost
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

paths.use_layer2_code()
import config  # noqa: E402
from evidence import bundle  # noqa: E402
from evidence.build import EvidenceBuilder, candidates, top_features  # noqa: E402
from layer1.engine import Layer1Engine  # noqa: E402
from retrieval import novelty, reference_set  # noqa: E402
from retrieval.retriever import SHAPKNNRetriever  # noqa: E402
from utils.io import git_commit  # noqa: E402

VARIANTS = ["ae_xgboost", "svdd_xgboost"]
# Cac lan chay truoc (v1-v3) deu dung dung tap nay; giu nguyen de so sanh duoc.
REFERENCE_RUN = paths.OUT / "rq4_vlm" / "pilot_results_ae_xgboost_neutral_v1.jsonl"
OUT_ROOT = paths.OUT / "rq4_vlm" / "setup1_bundle"


def reference_alerts():
    """(row_idx, nhan that) cua 140 alert da dung o cac lan chay truoc."""
    assert REFERENCE_RUN.exists(), (
        f"Khong tim thay {REFERENCE_RUN}. File nay giu danh sach alert cua cac lan chay truoc; "
        "thieu no thi ket qua moi khong so sanh truc tiep duoc voi chung.")
    rows = [json.loads(l) for l in REFERENCE_RUN.open(encoding="utf-8")]
    out = [(int(r["row_idx"]), r["Label_Text"]) for r in rows if r.get("set_type") == "tp"]
    assert len({i for i, _ in out}) == len(out), "row_idx trung nhau trong file tham chieu"
    return sorted(out)


def build(variant, alerts, val_df):
    t0 = time.time()
    out = paths.ensure(OUT_ROOT / variant)
    l1 = Layer1Engine(variant)

    npz = reference_set.protocol_b_path(variant)
    assert npz.exists(), f"Thieu {npz} (chay retrieval/reference_set.py truoc)"
    tau_zd = novelty.read_tau(npz)
    ret = SHAPKNNRetriever(str(npz), str(paths.PROFILES), tau_zd=tau_zd, k=config.RAG_NOVELTY_K)
    ev = EvidenceBuilder(ret)

    base = l1.base_feature_columns(val_df)
    idx = [i for i, _ in alerts]
    X = val_df.loc[idx, base].to_numpy("float32")
    A = l1.build_input(X)
    proba, phi = l1.score(X)
    names = list(l1.feature_names)
    results = ret.retrieve_batch(phi)

    manifest = []
    for j, (row_idx, true_label) in enumerate(alerts):
        aid = f"{true_label}__{row_idx}"
        r = results[j]
        cands = candidates([m["matched_label"] for m in r["matches"]], [m["distance"] for m in r["matches"]])
        imgs, _ = ev.make_images(val_df.loc[row_idx], phi[j], names,
                                 str(out / "images" / aid), float(proba[j]), model_input=A[j])
        manifest.append({
            "alert_id": aid, "row_idx": row_idx, "true_label": true_label,
            "layer1_proba": float(proba[j]), "layer1_threshold": float(l1.threshold),
            "novelty_score": float(r["novelty_score"]), "tau_zd": tau_zd,
            "flagged": bool(r["is_potential_zero_day"]),
            "candidates": cands, "retriever_top1": cands[0]["label"],
            "shap_top_features": top_features(phi[j], names),
            "images": [bundle.rel(x, out) for x in imgs],
        })
        if (j + 1) % 20 == 0:
            print(f"  {j + 1:>3}/{len(alerts)}", flush=True)

    top1 = sum(m["retriever_top1"] == m["true_label"] for m in manifest)
    hit3 = sum(m["true_label"] in [c["label"] for c in m["candidates"]] for m in manifest)
    bundle.write(out, manifest, {
        "setup": "Setup 1 (Protocol B, known attack types)", "variant": variant,
        "n_alerts": len(manifest), "source": "val_full.csv",
        "alert_selection": f"the {len(manifest)} alerts of {REFERENCE_RUN.name}",
        "retriever_top1_accuracy": round(top1 / len(manifest), 4),
        "retriever_hit_at_3": round(hit3 / len(manifest), 4),
        "n_flagged": int(sum(m["flagged"] for m in manifest)),
        "tau_zd": tau_zd, "k": config.RAG_NOVELTY_K,
        "code_commit": git_commit(paths.LAYER2, paths.WS / "UTS_URA"),
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "build_seconds": round(time.time() - t0, 1),
    }, phi=phi)
    print(f"{variant}: {len(manifest)} alert | retriever top-1 {top1}/{len(manifest)} "
          f"({top1 / len(manifest):.1%}) | Hit@3 {hit3 / len(manifest):.1%} -> {out}", flush=True)


def main():
    want = sys.argv[1:] or VARIANTS
    alerts = reference_alerts()
    print(f"{len(alerts)} alert tu {REFERENCE_RUN.name}", flush=True)
    val_df = pd.read_csv(config.VAL_BINARY_CSV)
    for v in want:
        print(f"\n=== {v} ===", flush=True)
        build(v, alerts, val_df)


if __name__ == "__main__":
    main()
