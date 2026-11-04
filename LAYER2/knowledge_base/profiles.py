"""Profile MITRE ATT&CK cua 14 loai tan cong (+ unknown_anomaly) va bang tra nhan -> profile."""
import json
from pathlib import Path

PROFILES_DIR = Path(__file__).resolve().parent / "attack_profiles"

# Label_Text (ten nhan trong LAYER2/data/*.csv va khoa label__<nhan> cua tap R)
# -> id cua profile trong attack_profiles/*.json.
LABEL_TO_PROFILE_ID = {
    "Bot": "bot",
    "Brute_Force_Web": "brute_force_web",
    "Brute_Force_XSS": "web_attack_xss",
    "DDOS_attack_HOIC": "ddos_hoic",
    "DDOS_attack_LOIC_UDP": "ddos_udp_flood",
    "DDoS_attacks_LOIC_HTTP": "ddos_loic_http",
    "DoS_attacks_GoldenEye": "dos_goldeneye",
    "DoS_attacks_Hulk": "dos_hulk",
    "DoS_attacks_SlowHTTPTest": "dos_slowhttptest",
    "DoS_attacks_Slowloris": "dos_slowloris",
    "FTP_BruteForce": "ftp_bruteforce",
    "Infiltration": "infiltration",
    "SQL_Injection": "web_attack_sql_injection",
    "SSH_Bruteforce": "ssh_bruteforce",
}


def load_profiles(profiles_dir=PROFILES_DIR):
    """{id: doc} cua moi file JSON trong thu muc profile."""
    docs = (json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(profiles_dir).glob("*.json")))
    return {d["id"]: d for d in docs}
