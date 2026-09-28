"""
Prompt + parser cho Layer 2.

Schema bao cao (6 muc):
  1. Executive summary
  2. Attack assessment  (Predicted class, Confidence, Severity,
                         Top indicators, RAG match, Endpoints)
  3. Forensic reasoning
  4. MITRE mapping
  5. Mitigations
  6. Analyst notes

BA QUYET DINH THIET KE

1. KHONG co truong Verdict (bo 2026-09-28).
   Truoc day muc 2 co "Verdict: CONFIRMED_ATTACK | FALSE_POSITIVE" de VLM phan
   xu lai Layer 1. Da bo vi day la lap luan vong tron: RAG truy hoi bang chinh
   vector SHAP cua Layer 1, nen khong dung no de phu dinh Layer 1 duoc. Do thuc
   te: ca 5 false positive deu khop ATTACK profile voi sim = 1.000. Them nua,
   nhan Benign cua CSE-CIC-IDS2018 khong dang tin (audit DistriNet). Nhiem vu
   cua VLM gio la XAC DINH LOAI TAN CONG va viet report, khong phai phan xu.

2. Predicted class phai chon trong 14 ten co dinh.
   Khong cho phep "Benign" (knowledge base da bo benign.json, SHAP pool da bo
   label__Benign), khong cho phep ho chung chung ("DoS attack"). Co danh sach
   ngay trong prompt de M1 cham diem duoc.

3. Top indicators phai copy tu anh SHAP.
   Rang buoc nay de M8 (evidence grounding) va M9 (hallucinated evidence) do
   duoc: doi chieu ten feature VLM neu voi feature_names that.

VE PROMPT VARIANTS
   Run cu (v6/v7/v13, cung 40 mau) cho 37/40 doi ket qua chi vi doi chu trong
   prompt. NEUTRAL duoi day co y khong mom cau tra loi; cac bien the lech duoc
   giu lai trong PROMPT_VARIANTS de DO DO NHAY, khong phai de chon cai co so
   dep nhat.
"""
import re

# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a Tier-3 SOC analyst reviewing one candidate intrusion alert.

The images attached are internal evidence. Derive insight from them, but do not
reproduce chart values, axis labels, or legend text verbatim in your report.

Answer using the exact section headings given in the task. Do not add sections,
omit sections, or reorder them.\
"""

# --------------------------------------------------------------------------
# Khoi huong dan chinh (trung tinh)
# --------------------------------------------------------------------------
_NEUTRAL_INSTRUCTIONS = """\
Layer 1, an XGBoost gate, flagged this flow as malicious. Your task is to
identify WHICH attack this is and to write the forensic report an analyst would
act on. Determine the class from the evidence below, not from the retrieved
name alone.

Weigh three sources. None of them is the default answer.

  A. SHAP WATERFALL (image 1) - the per-feature attribution behind Layer 1's
     score. Values were computed for the malicious class: positive pushes
     toward attack, negative pushes toward benign. Judge bar MAGNITUDE, not
     only sign. Small or mixed bars are weak evidence in either direction.

  B. FLOW-STATISTIC CHARTS (images 2-5) - packet length profile, TCP flag
     composition, inter-arrival timing, directional balance.

  C. RETRIEVED CONTEXT (text below) - the nearest known profiles, matched by
     cosine similarity on this flow's SHAP vector. A high-similarity match is
     evidence; a low-similarity best match means this flow resembles nothing
     in the reference set and may be novel.

Where the three disagree, say so in the report rather than resolving the
conflict silently.\
"""

# --------------------------------------------------------------------------
# Schema bao cao
# --------------------------------------------------------------------------
_REPORT_SCHEMA = """\
Produce the report in exactly this form.

## 1. EXECUTIVE SUMMARY
Three to four sentences: what was detected, whether the evidence corroborates
it, the best-matching profile, and your conclusion.

## 2. ATTACK ASSESSMENT
Predicted class: <exactly one name from the list below>
Confidence: <integer 0-100>
Severity: <Low, Medium, High or Critical>
Top indicators: <three feature names, copied verbatim from the SHAP chart>
RAG match: <name of the best-matching profile and its similarity>
Endpoints: <source, destination, direction of the activity>

## 3. FORENSIC REASONING
One bullet per evidence source (SHAP, flow charts, retrieved context). Close
with one sentence stating explicitly whether the sources converge or conflict.

## 4. MITRE MAPPING
Tactic and technique ID for the class named above.

## 5. MITIGATIONS
Ranked recommendations for that technique.

## 6. ANALYST NOTES
Data gaps, caveats, recommended next steps.

Three rules:
  - Predicted class must be exactly one of the 14 names listed below. Do not
    invent a name, do not answer with a generic family such as "DoS attack",
    and do not answer "Benign".
  - Top indicators must be feature names that actually appear in the SHAP
    chart, copied as written. Do not name a feature you did not read there.
  - Confidence is your certainty in the class you chose. Use the full 0-100
    range: report low confidence when the evidence is genuinely ambiguous.

ATTACK CLASSES (choose exactly one):
  Bot                       Brute_Force_Web           Brute_Force_XSS
  DDOS_attack_HOIC          DDOS_attack_LOIC_UDP      DDoS_attacks_LOIC_HTTP
  DoS_attacks_GoldenEye     DoS_attacks_Hulk          DoS_attacks_SlowHTTPTest
  DoS_attacks_Slowloris     FTP_BruteForce            Infiltration
  SQL_Injection             SSH_Bruteforce\
"""

# --------------------------------------------------------------------------
# Cac bien the de do do nhay voi prompt (ablation)
#
# Ba bien the chi khac nhau o mot doan khung, phan schema giu nguyen, nen
# chenh lech ket qua quy duoc ve khung do.
# --------------------------------------------------------------------------
# Bien the 1: uu tien ket qua truy hoi.
_FRAME_TRUST_RAG = """\
Layer 1, an XGBoost gate, flagged this flow as malicious. The retrieved profile
below was matched on this flow's own attribution vector and is a reliable
indicator of the attack type. Identify the class and write the report.\
"""

# Bien the 2: uu tien bang chung hinh anh.
_FRAME_TRUST_VISUAL = """\
Layer 1, an XGBoost gate, flagged this flow as malicious. Retrieval can match
the wrong profile when two attack types share a traffic signature. Read the
charts first, then identify the class and write the report.\
"""

PROMPT_VARIANTS = {
    "neutral":      _NEUTRAL_INSTRUCTIONS,
    "trust_rag":    _FRAME_TRUST_RAG,
    "trust_visual": _FRAME_TRUST_VISUAL,
}

DEFAULT_VARIANT = "neutral"


def build_user_prompt(rag_context: str, variant: str = DEFAULT_VARIANT) -> str:
    """Ghep prompt gui kem anh.

    Args:
        rag_context: khoi text tu retriever.format_context_for_vlm().
        variant: khoa trong PROMPT_VARIANTS.
    """
    if variant not in PROMPT_VARIANTS:
        raise KeyError(f"variant khong hop le: {variant!r}; co: {list(PROMPT_VARIANTS)}")

    return (
        f"{PROMPT_VARIANTS[variant]}\n\n"
        f"{rag_context}\n\n"
        f"{_REPORT_SCHEMA}"
    )


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
_CONF_RE = re.compile(r"confidence\s*[::]\s*\**\s*(\d{1,3})\s*%?", re.IGNORECASE)
_TOPIND_RE = re.compile(r"top\s*indicators?\s*[::]\s*\**\s*([^\n]+)", re.IGNORECASE)
_RAGM_RE = re.compile(r"rag\s*match\s*[::]\s*\**\s*([^\n]+)", re.IGNORECASE)
_CLASS_RE = re.compile(r"predicted\s*class\s*[::]\s*\**\s*([^\n*]+)", re.IGNORECASE)
_SEV_RE = re.compile(r"severity\s*[::]\s*\**\s*(low|medium|high|critical)", re.IGNORECASE)
_MITRE_RE = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b")
_TACTIC_RE = re.compile(r"tactic\s*[::]\s*\**\s*([^\n*]+)", re.IGNORECASE)


def _section(text: str, num: int) -> str:
    """Cat noi dung muc <num> theo tieu de '## <num>.'."""
    m = re.search(rf"^#{{1,4}}\s*{num}\.\s*[^\n]*\n(.*?)(?=^#{{1,4}}\s*\d+\.|\Z)",
                  text, re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_report(raw: str) -> dict:
    """Tach cac truong may doc duoc tu bao cao.

    Khong doan khi thieu: verdict khong tim thay -> None kem parse_error, de
    truong hop hong hien ra trong ket qua thay vi bi tinh nham thanh mot phia.
    (v8 cu co 35/40 parse fail ma van ra duoc 'ty le', do la so rac.)
    """
    out = {
        "confidence": None,
        "predicted_class": None,
        "severity": None,
        "top_indicators": None,
        "rag_match": None,
        "mitre_id": None,
        "mitre_tactic": None,
        "executive_summary": None,
        "reasoning": None,
        "analyst_notes": None,
        "n_sections": 0,
        "parse_error": None,
    }
    if not raw or not raw.strip():
        out["parse_error"] = "empty_output"
        return out

    m = _CLASS_RE.search(raw)
    if m:
        out["predicted_class"] = m.group(1).strip().strip("*").strip()
    else:
        out["parse_error"] = "predicted_class_not_found"

    m = _CONF_RE.search(raw)
    if m:
        c = int(m.group(1))
        out["confidence"] = c if 0 <= c <= 100 else None
        if out["confidence"] is None:
            out["parse_error"] = (out["parse_error"] or "") + "|confidence_out_of_range"

    m = _SEV_RE.search(raw)
    if m:
        out["severity"] = m.group(1).capitalize()

    m = _TOPIND_RE.search(raw)
    if m:
        out["top_indicators"] = m.group(1).strip().strip("*").strip()

    m = _RAGM_RE.search(raw)
    if m:
        out["rag_match"] = m.group(1).strip().strip("*").strip()

    out["executive_summary"] = _section(raw, 1) or None
    out["reasoning"] = _section(raw, 3) or None
    out["analyst_notes"] = _section(raw, 6) or None

    mitre_sec = _section(raw, 4)
    m = _MITRE_RE.search(mitre_sec)
    if m:
        out["mitre_id"] = m.group(1)
    m = _TACTIC_RE.search(mitre_sec)
    if m:
        out["mitre_tactic"] = m.group(1).strip().strip("*").strip()

    # M7 schema compliance: dem so muc co mat trong 6 muc bat buoc.
    out["n_sections"] = sum(1 for i in range(1, 7) if _section(raw, i))
    if out["n_sections"] < 6:
        out["parse_error"] = (out["parse_error"] or "") + f"|only_{out['n_sections']}_sections"

    return out
