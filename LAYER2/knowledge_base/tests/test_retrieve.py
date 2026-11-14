"""
Kiem luat chon doan cua knowledge_base/retrieve.py tren cac flow GIA LAP (khong dung du lieu that).

    python -m pytest LAYER2/knowledge_base/tests -q
    python LAYER2/knowledge_base/tests/test_retrieve.py        # chay khong can pytest
"""
import hashlib
import re
import sys
import warnings
from pathlib import Path

import yaml

LAYER2 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAYER2))

from knowledge_base.retrieve import KnowledgeBase  # noqa: E402

KB_DIR = LAYER2 / "knowledge_base"
KB = KnowledgeBase()
TCP_WEB = {"Protocol": 6, "Dst Port": 80, "TotLen Fwd Pkts": 120, "TotLen Bwd Pkts": 900,
           "Init Fwd Win Byts": 8192, "Init Bwd Win Byts": 26883, "Flow Duration": 5000, "PSH Flag Cnt": 1}


def _table():
    return yaml.safe_load((KB_DIR / "mapping" / "borisenko2022_table2.yaml").read_text(encoding="utf-8"))


# --- khoa nhan -> technique (Borisenko 2022, Table II) ------------------------------------------
def test_index_follows_borisenko_table():
    for label, row in _table()["labels"].items():
        want = [e["sub_technique"] or e["technique"] for e in row["entries"]]
        assert KB.index["labels"][label]["techniques"] == want, label
    assert KB.index["labels"]["Infiltration"]["techniques"] == ["T1204.002", "T1046"]
    assert KB.index["labels"]["Bot"]["techniques"] == ["T1583.005", "T1113"]
    assert KB.index["labels"]["DDOS_attack_LOIC_UDP"]["techniques"] == ["T1499.002"]


def test_every_label_has_mapping_and_technique_passages():
    for label, info in KB.index["labels"].items():
        assert f"M-{label}" in KB.passages
        for t in info["techniques"]:
            assert f"B-{t}" in KB.passages
            assert (f"B-DET-{t}" in KB.passages) == info["detection"][t]
            assert all(f"B-{c}" in KB.passages for c in info["capec"][t])


def test_label_passages_and_mitigations():
    s = KB.select("SQL_Injection", False, ["Fwd Seg Size Min", "Dst Port"], TCP_WEB)
    assert s.pids()[:3] == ["M-SQL_Injection", "B-T1190", "B-DET-T1190"]
    assert s.techniques == ["T1190"]
    assert all(p["role"] == "attack_type_background" for p in s.passages if p["part"] in ("M", "B"))
    assert "Borisenko et al. (2022), Table II" in s.passages[0]["text"]
    assert s.mitigations and all(m["pid"].startswith("G-T1190-") for m in s.mitigations)
    assert not any(p["part"] == "G" for p in s.passages), "G khong duoc dua cho VLM"


def test_two_technique_label_gets_both():
    s = KB.select("Infiltration", False, ["Dst Port"], TCP_WEB)
    for pid in ("M-Infiltration", "B-T1204.002", "B-T1046", "B-DET-T1046", "B-CAPEC-300"):
        assert pid in s.pids(), pid
    assert {m["pid"].split("-")[1] for m in s.mitigations} <= {"T1204.002", "T1046"}


def test_flagged_alert_hides_technique():
    s = KB.select("DoS_attacks_Hulk", True, ["Fwd Seg Size Min"], TCP_WEB, contrast_features=["Flow Duration"])
    assert "M-DoS_attacks_Hulk" not in s.pids() and s.techniques == []
    assert all(p["role"] == "comparison_only" for p in s.passages if p["part"] == "B")
    assert "H1-Flow_Duration" in s.pids()
    assert s.mitigations == []
    shown = " ".join(f"{p['cite']} {p['title']} {p['text']}" for p in s.passages if p["part"] == "B")
    assert "T1499" not in shown and "Service Exhaustion Flood" not in shown and "CAPEC" not in shown
    b = [p["cite"] for p in s.passages if p["part"] == "B"]
    assert b == [f"REF-MECH-{i}" for i in range(1, len(b) + 1)] and len(b) >= 2


def test_protocol_a_excludes_held_out_label():
    s = KB.select("Infiltration", False, ["Dst Port"], TCP_WEB, exclude_labels=["Infiltration"])
    assert not any(p["part"] in ("M", "B") for p in s.passages) and s.mitigations == []


# --- H1 -------------------------------------------------------------------------------------------
def test_replaced_columns_drop_readme_definition():
    readme = {f["csv"]: f["definition"] for f in yaml.safe_load(
        (KB_DIR / "definitions" / "feature_names.yaml").read_text(encoding="utf-8")) if f["readme"]}
    replaced = [p for p in KB.passages.values() if p["part"] == "H1" and p["kind"] == "defined"
                and p["keys"]["feature"] in readme]
    assert len(replaced) >= 20
    for p in replaced:
        col = p["keys"]["feature"]
        assert readme[col].rstrip(".") not in KB._text(p), col
    rst = KB._text(KB.passages["H1-RST_Flag_Cnt"])
    assert "ECE flag" in rst and "Number of packets with RST" not in rst


def test_ae_features_have_generic_definitions():
    s = KB.select("Bot", False, ["ae_latent_3", "ae_latent_7", "ae_recon_error"], TCP_WEB)
    assert s.pids().count("H1-ae_latent") == 1 and "H1-ae_recon_error" in s.pids()
    assert "mean squared error" in KB.passages["H1-ae_recon_error"]["text"]
    assert "latent (bottleneck) layer" in KB.passages["H1-ae_latent"]["text"]


def test_contrast_latent_row_maps_to_latent_passage():
    s = KB.select("DoS_attacks_Hulk", True, ["Fwd Seg Size Min"], TCP_WEB, contrast_features=["ae_latent"])
    assert "H1-ae_latent" in s.pids() and not s.missing


# --- H2 -------------------------------------------------------------------------------------------
def test_every_iana_registered_port_has_passage_and_note():
    ports = [p for p in KB.passages.values() if p["pid"].startswith("H2-PORT-") and p["pid"] != "H2-PORT-NOTE"]
    assert len(ports) > 5000, "moi cong co dang ky IANA (khong chon loc)"
    s = KB.select("SQL_Injection", False, ["Dst Port"], TCP_WEB)
    i = s.pids().index("H2-PORT-80")
    assert s.pids()[i + 1] == "H2-PORT-NOTE" and "does not mean" in s.passages[i + 1]["text"]


def test_unregistered_port_is_quiet_missing():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = KB.select("SQL_Injection", False, ["Dst Port"], dict(TCP_WEB, **{"Dst Port": 65535}))
    assert any(m["pid"] == "H2-PORT-65535" for m in s.missing)
    assert "H2-PORT-NOTE" not in s.pids()
    assert not any("H2-PORT" in str(x.message) for x in w)


def test_flag_column_maps_to_flag_it_actually_holds():
    s = KB.select("DoS_attacks_Hulk", False, ["SYN Flag Cnt", "RST Flag Cnt"], TCP_WEB)
    assert "H2-FLAG-PSH" in s.pids(), "cot 'SYN Flag Cnt' chua co PSH (ma nguon 2018)"
    assert "H2-FLAG-SYN" not in s.pids()
    assert "H2-FLAG-ECE" in s.pids() and "H2-FLAG-ECN-SETUP" in s.pids()


# --- H3 -------------------------------------------------------------------------------------------
def test_zero_payload_adds_caveat():
    x = dict(TCP_WEB, **{"TotLen Fwd Pkts": 0, "TotLen Bwd Pkts": 0})
    assert "H3-NOPAYLOAD" in KB.select("SQL_Injection", False, ["Fwd Seg Size Min"], x).pids()
    assert "H3-NOPAYLOAD" not in KB.select("SQL_Injection", False, ["Fwd Seg Size Min"], TCP_WEB).pids()


def test_udp_flow():
    x = {"Protocol": 17, "Dst Port": 53, "TotLen Fwd Pkts": 10, "TotLen Bwd Pkts": 0,
         "Init Fwd Win Byts": -1, "Init Bwd Win Byts": -1, "Flow Duration": 119_999_000}
    s = KB.select("DDOS_attack_LOIC_UDP", False, ["Init Fwd Win Byts"], x)
    for pid in ("H2-PROTO-17", "H2-PORT-53", "H3-UDP", "H3-NOWIN", "H3-TIMEOUT", "B-T1499.002"):
        assert pid in s.pids(), pid


def test_non_syn_tcp_first_packet_adds_winscale():
    assert KB.index["flag_column_holds"]["PSH Flag Cnt"] == "SYN"
    s = KB.select("DoS_attacks_Hulk", False, ["Init Fwd Win Byts"], dict(TCP_WEB, **{"PSH Flag Cnt": 0}))
    assert "H3-WINSCALE" in s.pids()
    s = KB.select("DoS_attacks_Hulk", False, ["Init Fwd Win Byts"], dict(TCP_WEB, **{"PSH Flag Cnt": 1}))
    assert "H3-WINSCALE" not in s.pids()


# --- loi va tinh toan ven --------------------------------------------------------------------------
def test_nan_inf_raw_values_do_not_crash():
    x = dict(TCP_WEB, **{"Protocol": float("nan"), "Dst Port": float("inf"), "TotLen Fwd Pkts": float("inf")})
    s = KB.select("SQL_Injection", False, ["Fwd Seg Size Min"], x)
    assert not any(p.startswith(("H2-PROTO-", "H2-PORT-")) for p in s.pids())
    assert "H3-NOPAYLOAD" not in s.pids(), "Inf khong duoc lam dieu kien dung"


def test_missing_passage_is_reported():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = KB.select("SQL_Injection", False, ["no_such_feature"], TCP_WEB)
    assert {"pid": "H1-no_such_feature", "reason": "top-|phi| feature"} in s.missing
    assert any("H1-no_such_feature" in str(x.message) for x in w)
    try:
        KnowledgeBase(strict=True).select("SQL_Injection", False, ["no_such_feature"], TCP_WEB)
        raise AssertionError("strict=True phai raise")
    except KeyError:
        pass


def test_no_percent_in_defined_text():
    for p in KB.passages.values():
        if p["kind"] == "defined" or p["notes"]:
            assert not re.findall(r"\d+(?:[.,]\d+)?\s*%", KB._text(p)), p["pid"]


def test_no_dataset_numbers_or_author_decisions_in_support():
    for p in KB.passages.values():
        for e in p["support"] + [s for n in p["notes"] for s in n["support"]]:
            assert set(e) <= {"src", "locator", "quote"}, (p["pid"], e)


def test_index_matches_passages_file():
    sha = hashlib.sha256((KB_DIR / "store" / "passages.jsonl").read_bytes()).hexdigest()
    assert KB.index["kb_sha256"] == sha, "index.json cu: chay lai build_kb.py"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("  ok", t.__name__)
    print(f"{len(tests)} test qua")
