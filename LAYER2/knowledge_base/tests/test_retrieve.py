"""
Kiem luat chon doan cua knowledge_base/retrieve.py tren cac flow GIA LAP (khong dung du lieu that).

    python -m pytest LAYER2/knowledge_base/tests -q
    python LAYER2/knowledge_base/tests/test_retrieve.py        # chay khong can pytest
"""
import json
import sys
from pathlib import Path

LAYER2 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAYER2))

from knowledge_base.retrieve import KnowledgeBase  # noqa: E402

KB = KnowledgeBase()
TCP_WEB = {"Protocol": 6, "Dst Port": 80, "TotLen Fwd Pkts": 120, "TotLen Bwd Pkts": 900,
           "Init Fwd Win Byts": 8192, "Init Bwd Win Byts": 26883, "Flow Duration": 5000}


def test_label_passages_and_mitigations():
    s = KB.select("SQL_Injection", False, ["Fwd Seg Size Min", "Dst Port"], TCP_WEB)
    assert s.pids()[:3] == ["A-SQL_Injection", "B-T1190", "B-CAPEC-66"]
    # checklist 1.4: A'/B' mo ta loai tan cong cua nhan tien le, khong mo ta flow
    assert all(p["role"] == "attack_type_background" for p in s.passages if p["part"] in ("A", "B"))
    assert "ATT&CK technique of this label: T1190" in s.passages[0]["text"], "khong co co: co cau technique"
    assert s.mitigations and all(m["pid"].startswith("G-T1190-") for m in s.mitigations)
    assert not any(p["pid"].startswith("G-") for p in s.passages), "G' khong duoc dua cho VLM"


def test_zero_payload_adds_caveat():
    x = dict(TCP_WEB, **{"TotLen Fwd Pkts": 0, "TotLen Bwd Pkts": 0})
    s = KB.select("SQL_Injection", False, ["Fwd Seg Size Min"], x)
    assert "H3-NOPAYLOAD" in s.pids()
    assert "H3-NOPAYLOAD" not in KB.select("SQL_Injection", False, ["Fwd Seg Size Min"], TCP_WEB).pids()


def test_udp_flow():
    x = {"Protocol": 17, "Dst Port": 80, "TotLen Fwd Pkts": 10, "TotLen Bwd Pkts": 0,
         "Init Fwd Win Byts": -1, "Init Bwd Win Byts": -1, "Flow Duration": 119_999_000}
    s = KB.select("DDOS_attack_LOIC_UDP", False, ["Init Fwd Win Byts"], x)
    for pid in ("H2-PROTO-17", "H3-UDP", "H3-NOWIN", "H3-TIMEOUT", "B-T1498.001"):
        assert pid in s.pids(), pid


def test_flagged_alert_marks_comparison_only_and_has_no_mitigation():
    s = KB.select("DoS_attacks_Hulk", True, ["Fwd Seg Size Min"], TCP_WEB, contrast_features=["Flow Duration"])
    assert all(p["role"] == "comparison_only" for p in s.passages if p["part"] in ("A", "B"))
    assert "H1-Flow_Duration" in s.pids()
    assert s.mitigations == []
    # checklist 1.6 (U11 = A): khi bat co khong co technique trong text, tieu de hay ma trich dan
    shown = " ".join(f"{p['cite']} {p['title']} {p['text']}" for p in s.passages if p["part"] in ("A", "B"))
    assert "T1499" not in shown and "Service Exhaustion" not in shown and "ATT&CK technique" not in shown
    assert [p["cite"] for p in s.passages if p["part"] == "B"] == ["REF-MECH-1", "REF-MECH-2"]


def test_replaced_columns_start_with_real_meaning():
    # checklist 1.1: cot co ghi chu `replaces` khong con dinh nghia ReadMe trong text gui VLM
    import yaml
    readme = {f["csv"]: f["definition"] for f in yaml.safe_load(
        (LAYER2 / "knowledge_base" / "authored" / "feature_names.yaml").read_text(encoding="utf-8")) if f["readme"]}
    replaced = [p for p in KB.passages.values() if p.get("readme_replaced")]
    assert len(replaced) >= 20
    for p in replaced:
        col = p["keys"]["feature"]
        assert readme[col].rstrip(".") not in KB._text(p), col
        assert p["text"].startswith(f"{col}: In ") or p["text"].startswith(f"{col}: The column name"), col
    rst = KB._text(KB.passages["H1-RST_Flag_Cnt"])
    assert "ECE flag" in rst and "Number of packets with RST" not in rst


def test_no_percent_in_vlm_text():
    # checklist 4.6 / D1: so lieu chi qua claim dung o moi nguon, khong co phan tram (tru nguong 99.9%)
    import re
    for p in KB.passages.values():
        if p["for_vlm"]:
            for m in re.findall(r"\d+(?:[.,]\d+)?\s*%", KB._text(p)):
                assert m == "99.9%", (p["pid"], m)


def test_non_syn_tcp_first_packet_adds_winscale():
    # checklist 1.5a: cot 'PSH Flag Cnt' chua co SYN cua goi dau
    s = KB.select("DoS_attacks_Hulk", False, ["Init Fwd Win Byts"], dict(TCP_WEB, **{"PSH Flag Cnt": 0}))
    assert "H3-WINSCALE" in s.pids()
    s = KB.select("DoS_attacks_Hulk", False, ["Init Fwd Win Byts"], dict(TCP_WEB, **{"PSH Flag Cnt": 1}))
    assert "H3-WINSCALE" not in s.pids()


def test_port_passage_warns_registration_is_not_usage():
    # checklist 1.5d: IANA 80/udp cung ghi http; doan cong co them cau RFC 6335
    assert "does not mean" in KB._text(KB.passages["H2-PORT-80"])


def test_flag_order_agrees_with_data_claims():
    # checklist 4.3 (ban nhe): thu tu HashMap tinh lai tu ma nguon phai khop cac cap cot trung nhau trong du lieu
    claims = json.loads((LAYER2 / "knowledge_base" / "store" / "claims.json").read_text(encoding="utf-8"))
    holds = KB.index["flag_column_holds"]
    assert claims["identical__Fwd PSH Flags__SYN Flag Cnt"]["holds_all"] and holds["SYN Flag Cnt"] == "PSH"
    assert claims["identical__Fwd URG Flags__CWE Flag Count"]["holds_all"] and holds["CWE Flag Count"] == "URG"
    # ECE va CWR duoc dat cung nhau trong SYN ECN-setup (RFC 3168): hai cot chua ECE / CWR gan nhu trung nhau
    assert claims["near_identical__RST Flag Cnt__ECE Flag Cnt"]["holds_all"]
    assert {holds["RST Flag Cnt"], holds["ECE Flag Cnt"]} == {"ECE", "CWR"}


def test_infiltration_has_extra_mechanism_hidden_when_flagged():
    # checklist 6.2: B' phu T1018 (IP sweep), technique cua nhan van la T1046
    s = KB.select("Infiltration", False, ["Dst Port"], TCP_WEB)
    assert "B-T1018" in s.pids() and KB.index["labels"]["Infiltration"]["technique"] == "T1046"
    s = KB.select("Infiltration", True, ["Dst Port"], TCP_WEB)
    b = [p for p in s.passages if p["part"] == "B"]
    assert [p["cite"] for p in b] == ["REF-MECH-1", "REF-MECH-2", "REF-MECH-3"]
    assert not any("T10" in f"{p['cite']} {p['title']}" for p in b)


def test_mechanism_paragraph_selection():
    # checklist 6.5: T1499.002 lay ca doan HTTP flood
    p = KB.passages["B-T1499.002"]
    assert "simple HTTP flood" in p["text"] and "paragraphs 1-2" in p["source"]["locator"]
    assert "<code>" not in KB.passages["B-T1018"]["text"]


def test_dictionary_attacks_use_capec_16():
    # checklist 6.3: CAPEC-16 cho SSH/FTP (kich ban tu dien), BFW giu CAPEC-49
    assert KB.index["labels"]["SSH_Bruteforce"]["capec"] == "CAPEC-16"
    assert KB.index["labels"]["FTP_BruteForce"]["capec"] == "CAPEC-16"
    assert KB.index["labels"]["Brute_Force_Web"]["capec"] == "CAPEC-49"


def test_protocol_a_excludes_held_out_label():
    s = KB.select("Infiltration", False, ["Dst Port"], TCP_WEB, exclude_labels=["Infiltration"])
    assert not any(p["part"] in ("A", "B") for p in s.passages)


def test_ae_latents_collapse_to_one_passage():
    s = KB.select("Bot", False, ["ae_latent_3", "ae_latent_7", "ae_recon_error"], TCP_WEB)
    assert s.pids().count("H1-ae_latent") == 1 and "H1-ae_recon_error" in s.pids()


def test_flag_column_maps_to_flag_it_actually_holds():
    s = KB.select("DoS_attacks_Hulk", False, ["SYN Flag Cnt", "RST Flag Cnt"], TCP_WEB)
    assert "H2-FLAG-PSH" in s.pids(), "cot 'SYN Flag Cnt' chua co PSH (ma nguon 2018)"
    assert "H2-FLAG-SYN" not in s.pids()
    assert "H2-FLAG-ECE" in s.pids() and "H2-FLAG-ECN-SETUP" in s.pids()


def test_nan_inf_raw_values_do_not_crash():
    # checklist 2.1: flow live co the co NaN/Inf; truoc day int(NaN) -> ValueError
    x = dict(TCP_WEB, **{"Protocol": float("nan"), "Dst Port": float("inf"), "TotLen Fwd Pkts": float("inf")})
    s = KB.select("SQL_Injection", False, ["Fwd Seg Size Min"], x)
    assert not any(p.startswith(("H2-PROTO-", "H2-PORT-")) for p in s.pids())
    assert "H3-NOPAYLOAD" not in s.pids(), "Inf khong duoc lam dieu kien dung"


def test_missing_passage_is_reported():
    # checklist 2.2: truoc day pid khong co trong KB bi bo qua im lang
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = KB.select("SQL_Injection", False, ["no_such_feature"], dict(TCP_WEB, **{"Dst Port": 4444}))
    assert {"pid": "H1-no_such_feature", "reason": "top-|phi| feature"} in s.missing
    assert any("H1-no_such_feature" in str(x.message) for x in w)
    assert any(m["pid"] == "H2-PORT-4444" for m in s.missing), "cong ngoai KB: ghi missing"
    assert not any("H2-PORT-4444" in str(x.message) for x in w), "cong ngoai KB: khong canh bao"
    strict = KnowledgeBase(strict=True)
    try:
        strict.select("SQL_Injection", False, ["no_such_feature"], TCP_WEB)
        raise AssertionError("strict=True phai raise")
    except KeyError:
        pass


def test_contrast_latent_row_maps_to_latent_passage():
    # checklist 2.3: feature_key cua dong latent gop trong bang C la "ae_latent"
    s = KB.select("DoS_attacks_Hulk", True, ["Fwd Seg Size Min"], TCP_WEB, contrast_features=["ae_latent"])
    assert "H1-ae_latent" in s.pids() and not s.missing


def test_every_label_has_label_card_and_mechanism():
    for label, info in KB.index["labels"].items():
        assert f"A-{label}" in KB.passages and f"B-{info['technique']}" in KB.passages


def test_index_matches_passages_file():
    import hashlib
    sha = hashlib.sha256((LAYER2 / "knowledge_base" / "store" / "passages.jsonl").read_bytes()).hexdigest()
    assert KB.index["kb_sha256"] == sha, "index.json cu: chay lai build_kb.py"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("  ok", t.__name__)
    print(f"{len(tests)} test qua")
