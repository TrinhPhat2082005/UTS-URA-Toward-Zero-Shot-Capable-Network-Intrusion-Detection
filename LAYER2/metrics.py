"""
Cham diem Layer 2 — xem METRIC.md de biet dinh nghia day du cua tung chi so.

Trong tam: chat luong REPORT, khong phai phan xu alert dung/sai.
Nhanh "VLM phu dinh Layer 1" (correction/retention) da bi bo vi RAG truy hoi
bang chinh vector SHAP cua Layer 1 - dung no de phu dinh Layer 1 la lap luan
vong tron; them nua nhan Benign cua CSE-CIC-IDS2018 khong dang tin.

    G1  M1 attack type accuracy | M2 per-class | M3 MITRE accuracy
    G2  M4 RAG adherence | M5 deviation harm | M6 VLM vs RAG | M7 schema
    G3  M8 evidence grounding | M9 hallucinated evidence | M10 convergence
    G4  M11 latency

M6 la chi so ban chinh: VLM co cong them gi so voi chi dung RAG khong.
"""
import json
import math
import re

import numpy as np
import pandas as pd

import config


# --------------------------------------------------------------------------
def load_results(path=None):
    """Doc JSONL ket qua thanh DataFrame."""
    path = path or config.RESULTS_JSONL
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    df = pd.DataFrame(rows)
    print(f"{len(df)} ban ghi tu {path}")
    return df


def _wilson(k, n, z=1.96):
    """Khoang tin cay Wilson - dung hon Wald khi n nho hoac ty le gan 0/1."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _norm(s):
    """Chuan hoa ten nhan de so sanh: bo _ - space, ve lowercase."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"[\s_\-]+", "", s).lower()


def _words(s):
    """Tach thanh tu con theo _/-/khoang trang, bo 's' so nhieu cuoi tu."""
    return [w.lower().rstrip("s") for w in re.split(r"[\s_\-]+", s or "") if w]


def _match(a, b):
    """Hai ten co chi cung mot lop khong.

    Hai buoc: (1) so prefix sau khi chuan hoa het ky tu ngan cach - bat duoc
    da so ca; (2) so tung tu con, bo qua 's' so nhieu cuoi moi tu - bat duoc
    ca lech kieu "DDoS_attacks_LOIC_HTTP" vs "DDOS_attack_LOIC_HTTP" (VLM
    doi dung 's' giua chuoi, khong phai sai nghia). Phat hien tren 140 mau
    dau tien: thieu buoc 2 lam M1 bao 61% thay vi dung 65%.
    """
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na == nb or na.startswith(nb) or nb.startswith(na):
        return True
    return _words(a) == _words(b)


def usable(df):
    """Ban ghi dung duoc: khong loi chay, parse duoc predicted_class."""
    d = df[df["run_error"].isna()] if "run_error" in df else df
    return d[d["predicted_class"].notna()]


# ==========================================================================
# G1 — Attack identification
# ==========================================================================
def attack_id_metrics(df, verbose=True):
    """M1 attack type accuracy, M3 MITRE coverage."""
    d = usable(df)
    n_drop = len(df) - len(d)

    hit = d.apply(lambda r: _match(r["predicted_class"], r.get("Label_Text")), axis=1)
    k, n = int(hit.sum()), len(d)
    lo, hi = _wilson(k, n)

    out = {
        "n_total": len(df), "n_usable": n, "n_dropped": n_drop,
        "m1_accuracy": k / n if n else float("nan"),
        "m1_ci": (lo, hi), "m1_correct": k,
    }

    if "mitre_id" in d:
        m = d[d["mitre_id"].notna()]
        out["m3_n_with_mitre"] = len(m)
        out["m3_coverage"] = len(m) / n if n else float("nan")

    if verbose:
        print(f"  M1 attack type accuracy  {out['m1_accuracy']:6.1%}  "
              f"[{lo:.1%}, {hi:.1%}]  {k}/{n}")
        if n_drop:
            print(f"     (bo {n_drop} ban ghi loi/khong parse duoc)")
        if "m3_coverage" in out:
            print(f"  M3 co MITRE id           {out['m3_coverage']:6.1%}  "
                  f"{out.get('m3_n_with_mitre')}/{n}")
    return out


def per_class(df, verbose=True):
    """M2 — accuracy tung nhan. CHAN DOAN, khong phai metric.

    n=10/nhan nen CI +-14-23pp: chi dung phat hien sap han (0/10, 1/10),
    khong so sanh cac nhan voi nhau.
    """
    d = usable(df)
    rows = []
    for lab, g in d.groupby("Label_Text"):
        hit = g.apply(lambda r: _match(r["predicted_class"], lab), axis=1)
        k, n = int(hit.sum()), len(g)
        lo, hi = _wilson(k, n)
        rows.append({"label": lab, "n": n, "correct": k,
                     "accuracy": k / n if n else float("nan"),
                     "ci_lo": lo, "ci_hi": hi})
    res = pd.DataFrame(rows).sort_values("accuracy")
    if verbose and len(res):
        print("\n  M2 per-class (n=10/nhan -> CI rong, chi de phat hien sap han)")
        for _, r in res.iterrows():
            flag = "  <-- SAP" if r["accuracy"] <= 0.1 else ""
            print(f"    {r['label']:<26}{r['correct']:>3}/{r['n']:<3}"
                  f"{r['accuracy']:>7.1%}{flag}")
    return res


# ==========================================================================
# G2 — Grounding
# ==========================================================================
def grounding_metrics(df, verbose=True):
    """M4 RAG adherence, M5 deviation harm, M6 VLM vs RAG, M7 schema."""
    d = usable(df)
    n = len(d)
    if not n:
        return {}

    vlm_ok = d.apply(lambda r: _match(r["predicted_class"], r.get("Label_Text")), axis=1)
    rag_ok = d.apply(lambda r: _match(r.get("rag_predicted_label"), r.get("Label_Text")), axis=1)
    follow = d.apply(lambda r: _match(r["predicted_class"], r.get("rag_predicted_label")), axis=1)

    dev = ~follow
    n_dev = int(dev.sum())
    harm = int((dev & ~vlm_ok).sum()) / n_dev if n_dev else float("nan")
    rescue = int((dev & vlm_ok & ~rag_ok).sum())

    out = {
        "n": n,
        "m4_rag_adherence": float(follow.mean()),
        "m5_n_deviated": n_dev,
        "m5_deviation_harm": harm,
        "m5_n_rescued": rescue,
        "m6_vlm_accuracy": float(vlm_ok.mean()),
        "m6_rag_accuracy": float(rag_ok.mean()),
        "m6_delta": float(vlm_ok.mean() - rag_ok.mean()),
    }

    if "n_sections" in d:
        out["m7_schema_full"] = float((d["n_sections"] >= 6).mean())

    if verbose:
        print(f"\n  M4 RAG adherence         {out['m4_rag_adherence']:6.1%}"
              f"   (VLM lap lai goi y RAG)")
        print(f"  M5 deviation harm        {harm:6.1%}"
              f"   ({n_dev} ca di lech, {rescue} ca sua dung duoc RAG)")
        print("\n  M6 VLM vs RAG  [chi so ban chinh]")
        print(f"     RAG accuracy          {out['m6_rag_accuracy']:6.1%}")
        print(f"     VLM accuracy          {out['m6_vlm_accuracy']:6.1%}")
        print(f"     delta                 {out['m6_delta']:+6.1%}  ", end="")
        dlt = out["m6_delta"]
        print("VLM cong them" if dlt > 0.02 else
              ("VLM LAM HONG" if dlt < -0.02 else "VLM chi chep lai RAG"))
        if "m7_schema_full" in out:
            print(f"\n  M7 schema du 6 muc       {out['m7_schema_full']:6.1%}")
    return out


# ==========================================================================
# G3 — Report quality
# ==========================================================================
def report_quality(df, verbose=True):
    """M8 evidence grounding, M9 hallucinated evidence, M10 convergence.

    M8/M9 doi chieu ten feature VLM neu (Top indicators + muc 3) voi
    shap_top_features cua chinh flow do.
    """
    d = usable(df)
    n = len(d)
    if not n or "shap_top_features" not in d:
        if verbose:
            print("\n  (khong co shap_top_features - bo qua G3)")
        return {}

    all_names = set()
    try:
        fn = json.load(open(config.LAYER1_ARTIFACTS[config.VARIANT]["features"],
                            encoding="utf-8"))
        all_names = {_norm(x) for x in fn}
    except Exception:
        pass

    grounded, halluc, conv = [], [], []
    for _, r in d.iterrows():
        text = " ".join(str(r.get(k) or "") for k in ("top_indicators", "reasoning"))
        tnorm = _norm(text)

        top = {_norm(f["name"]) for f in (r.get("shap_top_features") or [])}
        grounded.append(any(t and t in tnorm for t in top))

        # Cum giong ten feature ma khong thuoc feature_names -> nghi bia
        cand = re.findall(r"[A-Za-z][A-Za-z ]{3,28}(?:Len|Pkts|IAT|Flag|Port|"
                          r"Byts|Std|Mean|Max|Min|Cnt|error)", str(text))
        bad = [c for c in cand if all_names and _norm(c) not in all_names]
        halluc.append(len(bad) > 0)

        conv.append(bool(re.search(r"converg|contradict|conflict|disagree|agree",
                                   str(r.get("reasoning") or ""), re.I)))

    # Qwen thi thoang ro ngon ngu goc (vd "packets流向目的地端口8080"). Report
    # lan tieng Trung thi khong dung duoc, nen phai dem.
    cjk = re.compile(r"[一-鿿]")
    non_en = [bool(cjk.search(str(r.get("raw_output") or ""))) for _, r in d.iterrows()]

    out = {
        "n": n,
        "m8_evidence_grounding": float(np.mean(grounded)),
        "m9_hallucinated_evidence": float(np.mean(halluc)),
        "m10_convergence_stated": float(np.mean(conv)),
        "non_english_rate": float(np.mean(non_en)),
    }
    if verbose:
        print(f"\n  M8 nhac dung feature SHAP {out['m8_evidence_grounding']:6.1%}")
        print(f"  M9 nhac feature khong co  {out['m9_hallucinated_evidence']:6.1%}"
              f"   (uoc luong, can ra tay xac nhan)")
        print(f"  M10 co cau ket luan hoi tu {out['m10_convergence_stated']:6.1%}")
        if out["non_english_rate"]:
            print(f"  !! report lan chu Han     {out['non_english_rate']:6.1%}"
                  f"   ({sum(non_en)}/{n} ban ghi)")
    return out


# ==========================================================================
# G3 phu — confidence
# ==========================================================================
def confidence_auroc(df, conf_col="confidence_logprob", verbose=True):
    """AUROC cua confidence khi du doan 'predicted_class dung'."""
    d = usable(df)
    c = pd.to_numeric(d.get(conf_col), errors="coerce")
    ok = d.apply(lambda r: _match(r["predicted_class"], r.get("Label_Text")), axis=1)
    m = c.notna()
    if m.sum() < 5 or ok[m].nunique() < 2:
        if verbose:
            print(f"\n  ({conf_col}: khong du du lieu de tinh AUROC)")
        return float("nan")
    from sklearn.metrics import roc_auc_score
    a = float(roc_auc_score(ok[m], c[m]))
    if verbose:
        print(f"\n  AUROC({conf_col}) = {a:.3f}  ", end="")
        print("dung duoc lam nguong" if a > 0.7 else
              ("co tin hieu yeu" if a > 0.6 else "KHONG phan biet duoc dung/sai"))
    return a


def confidence_sweep(df, conf_col="confidence_logprob", steps=8):
    """Accuracy theo tung nguong confidence (kem ty le abstain)."""
    d = usable(df)
    c = pd.to_numeric(d.get(conf_col), errors="coerce").dropna()
    if c.empty:
        print(f"Khong co gia tri o cot {conf_col!r}")
        return pd.DataFrame()

    lo, hi = float(c.min()), float(c.max())
    if math.isclose(lo, hi):
        print(f"!! {conf_col} gan nhu hang so ({lo:.3f}) - khong quet duoc nguong.")

    rows = []
    for t in np.linspace(lo, hi, steps):
        sub = d[pd.to_numeric(d[conf_col], errors="coerce") >= t]
        if not len(sub):
            continue
        hit = sub.apply(lambda r: _match(r["predicted_class"], r.get("Label_Text")), axis=1)
        rows.append({"threshold": round(float(t), 4), "n_kept": len(sub),
                     "n_abstain": len(d) - len(sub),
                     "accuracy": float(hit.mean())})
    return pd.DataFrame(rows)


# ==========================================================================
# G4 — Cost
# ==========================================================================
def latency_breakdown(df):
    """M11 — thoi gian trung binh tung chang."""
    cols = [("t_layer1_s", "Layer 1 + SHAP"), ("t_rag_s", "RAG k-NN"),
            ("t_images_s", "Ve 5 anh"), ("t_vlm_s", "VLM"), ("latency_s", "TONG")]
    rows = []
    for c, name in cols:
        if c in df:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s):
                rows.append({"stage": name, "mean_s": s.mean(),
                             "median_s": s.median(), "p95_s": s.quantile(0.95),
                             "n": len(s)})
    return pd.DataFrame(rows)


# ==========================================================================
def report_all(df):
    """In het G1-G4 mot luot."""
    print("=" * 66)
    print(f"  {len(df)} ban ghi")
    print("=" * 66)
    attack_id_metrics(df)
    per_class(df)
    grounding_metrics(df)
    report_quality(df)
    confidence_auroc(df)
    print("\n  M11 latency")
    for _, r in latency_breakdown(df).iterrows():
        print(f"    {r['stage']:<18}{r['mean_s']:>8.3f}s  (p95 {r['p95_s']:.2f}s)")


def prompt_sensitivity(df):
    """So cac prompt variant tren cung tap mau + dem so mau doi ket qua."""
    if "prompt_variant" not in df or df["prompt_variant"].nunique() < 2:
        print("Chi co mot prompt variant - khong so duoc.")
        return None, None

    per = []
    for v, g in df.groupby("prompt_variant"):
        m = attack_id_metrics(g, verbose=False)
        gr = grounding_metrics(g, verbose=False)
        per.append({"variant": v, "n": m["n_usable"],
                    "m1_accuracy": m["m1_accuracy"],
                    "m4_rag_adherence": gr.get("m4_rag_adherence")})
    per = pd.DataFrame(per).sort_values("m1_accuracy", ascending=False)

    piv = df.pivot_table(index="row_idx", columns="prompt_variant",
                         values="predicted_class", aggfunc="first").dropna()
    n_flip = int((piv.nunique(axis=1) > 1).sum()) if len(piv) else 0
    if len(piv):
        print(f"\n{n_flip}/{len(piv)} mau doi predicted_class giua cac variant "
              f"({n_flip / len(piv):.0%})")
    return per, piv
