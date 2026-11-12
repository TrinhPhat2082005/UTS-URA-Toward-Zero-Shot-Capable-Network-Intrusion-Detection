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
    assert all(p["role"] == "alert_label" for p in s.passages if p["part"] in ("A", "B"))
    assert s.mitigations and all(m["pid"].startswith("G-T1190-") for m in s.mitigations)
    assert not any(p["pid"].startswith("G-") for p in s.passages), "G' khong duoc dua cho VLM"


def test_zero_payload_adds_required_caveat():
    x = dict(TCP_WEB, **{"TotLen Fwd Pkts": 0, "TotLen Bwd Pkts": 0})
    s = KB.select("SQL_Injection", False, ["Fwd Seg Size Min"], x)
    assert "H3-NOPAYLOAD" in s.pids() and "H3-NOPAYLOAD" in s.required
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
    assert "H1-Flow_Duration" in s.pids() and "H1-Flow_Duration" not in s.required
    assert s.mitigations == []


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
