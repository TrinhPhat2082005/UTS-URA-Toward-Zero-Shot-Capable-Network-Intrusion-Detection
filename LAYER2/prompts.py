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
import random
import re

# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a Tier-3 SOC analyst reviewing one candidate intrusion alert.

Write the entire report in English. Do not use any other language, not even for
a single word or phrase.

The images attached are internal evidence. Derive insight from them, but do not
reproduce chart values, axis labels, or legend text verbatim in your report.

State only what the evidence shows. If a detail is not in the evidence, write
"not available" rather than inventing a value.

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

Weigh three sources. None of them is the default answer. Exactly five images
are attached (image 1 and images 2-5); the retrieved context is text, not an
image.

  A. SHAP WATERFALL (image 1) - the per-feature attribution behind Layer 1's
     score. Values were computed for the malicious class: positive pushes
     toward attack, negative pushes toward benign. Judge bar MAGNITUDE, not
     only sign. Small or mixed bars are weak evidence in either direction.
     Features named ae_latent_0 ... ae_latent_15 and ae_recon_error are not
     traffic fields: they are the latent code and the reconstruction error of
     an autoencoder trained on benign traffic. A large ae_recon_error means the
     flow is unlike benign traffic; the latent dimensions have no named meaning.

  B. FLOW-STATISTIC CHARTS (images 2-5) - packet length profile, TCP flag
     composition, inter-arrival timing, directional balance.

  C. RETRIEVED CONTEXT (text below) - the three known attack types whose
     reference flows lie nearest to this flow's SHAP vector, listed nearest
     first, each labelled with
     its attack type and the distance to its nearest reference flow (Euclidean;
     smaller is closer, near 0 means an almost identical known flow). If the
     novelty flag is raised, the context also contains the UNKNOWN profile:
     the flow lies outside the region occupied by known attack types and may be
     a novel attack.

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
RAG match: <attack type and distance of Candidate 1, copied from the context>
Endpoints: <destination port and protocol if the SHAP chart names them,
           otherwise "not available"; IP addresses are not in the evidence>

## 3. FORENSIC REASONING
One bullet per evidence source (SHAP, flow charts, retrieved context). Close
with one sentence stating explicitly whether the sources converge or conflict.

## 4. MITRE MAPPING
Tactic and technique ID for the class named above. For UNKNOWN: the tactic the
observed behavior suggests, and "Technique: N/A".

## 5. MITIGATIONS
Ranked recommendations for that technique.

## 6. ANALYST NOTES
Data gaps, caveats, recommended next steps.

Three rules:
  - Predicted class must be exactly one of the 15 names listed below, copied
    as written: one of the 14 attack types, or UNKNOWN when the evidence
    matches none of them. Do not write a profile title or a tactic there, do
    not invent a name, do not answer with a generic family such as "DoS
    attack", and do not answer "Benign".
  - Top indicators must be feature names that actually appear in the SHAP
    chart, copied as written. Do not name a feature you did not read there.
  - Confidence is your certainty in the class you chose. Use the full 0-100
    range: report low confidence when the evidence is genuinely ambiguous.

ATTACK CLASSES (choose exactly one):
  Bot                       Brute_Force_Web           Brute_Force_XSS
  DDOS_attack_HOIC          DDOS_attack_LOIC_UDP      DDoS_attacks_LOIC_HTTP
  DoS_attacks_GoldenEye     DoS_attacks_Hulk          DoS_attacks_SlowHTTPTest
  DoS_attacks_Slowloris     FTP_BruteForce            Infiltration
  SQL_Injection             SSH_Bruteforce            UNKNOWN\
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


def format_retrieval_context(retrieval_result: dict, k: int, tau_zd, unknown_profile: dict) -> str:
    """Context block: three candidate profiles, nearest first, each labelled with its
    attack type and distance. Every alert gets the novelty score as a distance
    scale; when the flag is raised, the unknown_anomaly profile is added first
    under the attack type UNKNOWN.

    Args:
        retrieval_result: mot phan tu cua SHAPKNNRetriever.retrieve_batch().
        k, tau_zd, unknown_profile: retriever.k, retriever.tau_zd, retriever.unknown_profile.
    """
    flagged = retrieval_result["is_potential_zero_day"]
    context = "### RETRIEVED CONTEXT (MITRE ATT&CK PROFILES)\n"

    if not flagged:
        context += (f"Novelty: mean L2 distance to the {k} nearest known attack flows = "
                    f"{retrieval_result['novelty_score']:.3f}"
                    + (f" (threshold {tau_zd:.3f})" if tau_zd is not None else "") + "\n\n")

    if flagged:
        doc = unknown_profile
        context += (f"NOVELTY FLAG RAISED: the mean L2 distance from this flow to its {k} "
                    f"nearest known attack flows is {retrieval_result['novelty_score']:.3f}"
                    + (f", above the threshold {tau_zd:.3f}" if tau_zd is not None else "")
                    + ". The flow lies outside the region occupied by known attack types and "
                    "may be a novel attack.\n\n")
        context += "#### Attack type: UNKNOWN\n"
        context += f"- **Profile:** {doc['title']}\n"
        context += f"- **Description:** {doc['description']}\n"
        context += f"- **Typical Pattern:** {doc['network_indicators']['traffic_pattern']}\n"
        context += f"- **Mitigation:** {', '.join([m['name'] for m in doc['mitigation']])}\n\n"

    context += "Candidates are listed nearest first.\n\n"
    for i, match in enumerate(retrieval_result["matches"]):
        doc = match["doc"]
        context += (f"#### Candidate {i+1} - Attack type: {match['matched_label']} "
                    f"(distance to nearest known flow: {match['distance']:.3f})\n")
        context += f"- **Technique:** {doc['title']} ({doc['mitre_id']})\n"
        context += f"- **Tactic:** {doc['mitre_tactic']}\n"
        context += f"- **Description:** {doc['description']}\n"
        context += f"- **Typical Pattern:** {doc['network_indicators']['traffic_pattern']}\n"
        context += f"- **Mitigation:** {', '.join([m['name'] for m in doc['mitigation']])}\n"
        if doc.get("mitre_url") and doc["mitre_url"] != "N/A":
            context += f"- **Reference:** {doc['mitre_url']}\n"
        context += "\n"

    return context


def build_user_prompt(rag_context: str, variant: str = DEFAULT_VARIANT) -> str:
    """Ghep prompt gui kem anh.

    Args:
        rag_context: khoi text tu format_retrieval_context().
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


# ==========================================================================
# Prompt chon ung vien (Section 3.6, 4.5; Setup 2 va Setup 1)
#
# Khong gian dau ra thong nhat: 3 loai tan cong do retriever tra ve + UNKNOWN.
#   - Khong co co zero-day: 3 ung vien (co profile) roi UNKNOWN o cuoi, khong profile.
#   - Co co zero-day: profile unknown_anomaly va canh bao dat LEN DAU, roi 3 ung vien.
# Context duoc dung tu du lieu co cau truc (nhan, khoang cach, id profile) chu khong
# tu chuoi format_retrieval_context(), nen sua prompt khong phai dung lai evidence.
# ==========================================================================
UNKNOWN = "UNKNOWN"

MITRE_TACTICS = [
    "Reconnaissance", "Resource Development", "Initial Access", "Execution",
    "Persistence", "Privilege Escalation", "Defense Evasion", "Credential Access",
    "Discovery", "Lateral Movement", "Collection", "Command and Control",
    "Exfiltration", "Impact",
]

_CANDIDATE_INSTRUCTIONS = """\
Layer 1, an XGBoost gate, flagged this flow as malicious. A retriever compared the
flow's SHAP attribution vector with the attribution vectors of known attack flows
and returned the options listed under RETRIEVED CONTEXT. Decide which option the
evidence supports, then write the forensic report an analyst would act on.

Weigh three sources:

  A. SHAP WATERFALL (image 1) - the per-feature attribution behind Layer 1's
     score. Positive values push toward attack, negative toward benign. Judge
     bar MAGNITUDE, not only sign.

  B. FLOW-STATISTIC CHARTS (images 2-5) - packet length profile, TCP flag
     composition, inter-arrival timing, directional balance.

  C. RETRIEVED CONTEXT (text below) - the options you may choose from. Each known
     attack option gives the attack type, its MITRE ATT&CK profile and the
     distance from this flow to the nearest reference flow of that type (smaller
     is closer). UNKNOWN means the flow matches none of the known attack types.

Choose exactly one option. Choose a known attack type only if the evidence is
consistent with its profile; otherwise choose UNKNOWN.\
"""

# Ban khong co UNKNOWN: dung cho Setup 1 (Protocol B), noi moi alert deu thuoc 14
# loai da biet. Giu nguyen phan A/B/C, chi bo moi loi nhac toi UNKNOWN.
_CANDIDATE_INSTRUCTIONS_NO_UNKNOWN = _CANDIDATE_INSTRUCTIONS.replace(
    " UNKNOWN means the flow matches none of the known attack types.", ""
).replace(
    "Choose exactly one option. Choose a known attack type only if the evidence is\n"
    "consistent with its profile; otherwise choose UNKNOWN.",
    "Choose exactly one option: the attack type whose profile and reference distance\n"
    "the evidence supports best.",
)

_CANDIDATE_SCHEMA = """\
Produce the report in exactly this form.

## 1. EXECUTIVE SUMMARY
Three to four sentences: what was detected, which option you selected and why.

## 2. ATTACK ASSESSMENT
Selected option: <copy exactly one option name: {options}>
Tactic: <exactly one MITRE ATT&CK tactic from: {tactics}>
Severity: <Low, Medium, High or Critical>
Top indicators: <three feature names, copied verbatim from the SHAP chart>
Endpoints: <destination port and protocol if the SHAP chart names them,
           otherwise "not available"; IP addresses are not in the evidence>

## 3. FORENSIC REASONING
One bullet per evidence source (SHAP, flow charts, retrieved context). Close
with one sentence stating whether the sources converge or conflict.

## 4. MITRE MAPPING
Two labelled lines, in this order. For a known attack type copy both from its
profile; for UNKNOWN name the tactic the behaviour suggests and write N/A:
Tactic: <the tactic name, for example Credential Access>
Technique: <the identifier only, for example T1110.001, or N/A>

## 5. MITIGATIONS
Ranked recommendations.

## 6. ANALYST NOTES
Data gaps, caveats, recommended next steps.

Rules:
  - Selected option must be one of the option names above, copied exactly. Do not
    write a profile title, a tactic or any other name there.
  - Top indicators must be feature names that appear in the SHAP chart.\
"""

# Ban khong co UNKNOWN cua schema: muc 4 chi con nhanh "loai da biet". Hai dong
# duoc dan nhan ro rang vi model tung dao chung - ghi technique ID vao o tactic.
_CANDIDATE_SCHEMA_NO_UNKNOWN = _CANDIDATE_SCHEMA.replace(
    "Two labelled lines, in this order. For a known attack type copy both from its\n"
    "profile; for UNKNOWN name the tactic the behaviour suggests and write N/A:\n"
    "Tactic: <the tactic name, for example Credential Access>\n"
    "Technique: <the identifier only, for example T1110.001, or N/A>",
    "Copy both lines from the profile of the option you selected:\n"
    "Tactic: <the tactic name, for example Credential Access>\n"
    "Technique: <the identifier only, for example T1110.001>",
)


def _profile_block(doc, variant=None, include_observed=True):
    mit = ", ".join(m["name"] for m in doc.get("mitigation", []))
    lines = [f"- Technique: {doc['title']} ({doc['mitre_id']})",
             f"- Tactic: {doc['mitre_tactic']}",
             f"- Description: {doc['description']}",
             f"- Typical pattern: {doc['network_indicators']['traffic_pattern']}"]
    # observed_features duoc tinh tu top-5 mean |SHAP| tren chinh tap tham chieu —
    # cung dai luong retriever dung lam khoa xep hang. Dua no vao day thi ban do
    # SHAP cua alert va danh sach cua ung vien gan nhat trung nhau san (do duoc:
    # overlap 0.83), nen cau tra loi nam san trong cau hoi. Mac dinh TAT.
    # Checkpoint Protocol A mang hau to _mono, khong gian attribution van la cua
    # chinh variant do.
    if include_observed:
        key = variant[:-5] if variant and variant.endswith("_mono") else variant
        obs = (doc.get("network_indicators", {}).get("observed_features") or {}).get(key)
        if obs:
            lines.append(f"- Attribution usually strongest for this type: {', '.join(obs)}")
    if mit:
        lines.append(f"- Mitigations: {mit}")
    return "\n".join(lines)


def render_candidate_context(candidates, flagged, profiles, novelty=None, tau_zd=None,
                             variant=None, allow_unknown=True, include_observed=False,
                             shuffle_seed=None):
    """Khoi RETRIEVED CONTEXT cho prompt chon ung vien.

    Args:
        candidates: list dict {"label", "distance", "profile_id"}, gan nhat truoc.
        flagged: co zero-day cua retriever.
        profiles: {profile_id: doc} (knowledge_base/attack_profiles).
        novelty, tau_zd: in cho moi alert (thuoc do khoang cach); co co thi kem canh bao.
        variant: bien the Layer 1 ("ae_xgboost" / "svdd_xgboost") de lay dung
            observed_features — moi variant co khong gian attribution rieng.
        allow_unknown: False thi khong them lua chon UNKNOWN. Dung cho Setup 1
            (Protocol B), noi moi alert deu thuoc 14 loai da biet nen UNKNOWN
            luon sai; Setup 2 giu True vi do chinh la dap an dung.
        shuffle_seed: None thi in theo thu tu khoang cach tang dan. Dat mot so
            thi hoan vi thu tu in: khoang cach van hien day du nhung khong con
            trung voi vi tri, nen tach duoc "chon theo khoang cach" khoi "chon
            muc dau danh sach".
    """
    order = _option_order(candidates, flagged, allow_unknown, shuffle_seed)
    parts = ["### RETRIEVED CONTEXT"]
    # Diem novelty in giong nhau cho moi alert, khong phan biet co co: con so va
    # nguong da du de mo hinh tu doc, khong can mot canh bao rieng.
    if novelty is not None:
        parts.append(f"Novelty: mean distance to the 10 nearest known attack flows = {novelty:.3f}"
                     + (f" (threshold {tau_zd:.3f})" if tau_zd is not None else "") + ".")
    for item in order:
        if item is UNKNOWN:
            parts.append(f"#### Option: {UNKNOWN}\n"
                         f"{_profile_block(profiles['unknown_anomaly'], variant, include_observed)}")
        else:
            parts.append(
                f"#### Option: {item['label']} "
                f"(distance to nearest known flow: {item['distance']:.3f})\n"
                f"{_profile_block(profiles[item['profile_id']], variant, include_observed)}")
    return "\n\n".join(parts)


def _option_order(candidates, flagged, allow_unknown, shuffle_seed=None):
    """Thu tu cac lua chon se duoc in ra.

    UNKNOWN chi co mat khi retriever bat co zero-day. Khong co co nghia la
    retriever ket luan flow nam trong vung da biet, nen dua UNKNOWN vao se mau
    thuan voi chinh ket luan do; no cung la lua chon cua tang retrieval, khong
    phai thu mo hinh sinh bao cao tu quyet. Khi co mat, no la mot phan tu nhu
    moi lua chon khac: cung do dai khoi, cung tham gia hoan vi.
    """
    items = list(candidates)
    if allow_unknown and flagged:
        items.append(UNKNOWN)
    if shuffle_seed is not None:
        # Hoan vi on dinh theo alert: cung seed + cung tap ung vien -> cung thu tu.
        key = f"{shuffle_seed}|{'|'.join(c['label'] for c in candidates)}|{int(flagged)}"
        random.Random(key).shuffle(items)
    return items


def candidate_options(candidates, flagged, allow_unknown=True, shuffle_seed=None):
    """Ten cac lua chon, dung thu tu chung duoc in trong context."""
    return [UNKNOWN if x is UNKNOWN else x["label"]
            for x in _option_order(candidates, flagged, allow_unknown, shuffle_seed)]


def build_candidate_prompt(candidates, flagged, profiles, novelty=None, tau_zd=None,
                           variant=None, allow_unknown=True, include_observed=False,
                           shuffle_seed=None):
    """Tra (user_prompt, options), options theo dung thu tu da in.

    include_observed: mac dinh False. Bat len chi de chay lai nhanh doi chung
    cua ablation, khong dung cho ket qua chinh — xem chu thich o _profile_block.
    shuffle_seed: dat mot so de hoan vi thu tu in cac lua chon.
    """
    options = candidate_options(candidates, flagged, allow_unknown, shuffle_seed)
    ctx = render_candidate_context(candidates, flagged, profiles, novelty, tau_zd,
                                   variant, allow_unknown, include_observed, shuffle_seed)
    head = _CANDIDATE_INSTRUCTIONS if allow_unknown else _CANDIDATE_INSTRUCTIONS_NO_UNKNOWN
    tmpl = _CANDIDATE_SCHEMA if allow_unknown else _CANDIDATE_SCHEMA_NO_UNKNOWN
    schema = tmpl.format(options=" | ".join(options), tactics=", ".join(MITRE_TACTICS))
    return f"{head}\n\n{ctx}\n\n{schema}", options


_SELECTED_RE = re.compile(r"selected\s*option\s*[::]\s*\**\s*([^\n]+)", re.IGNORECASE)


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def parse_candidate_report(raw, options):
    """Tach truong cua bao cao chon ung vien.

    selected: mot phan tu cua options, hoac None neu khong khop (selected_raw giu
    nguyen van ban de xem lai). Khong doan: chuoi khop nhieu lua chon -> None.
    tactic: mot tactic trong MITRE_TACTICS (lay o muc 2, neu thieu thi muc 4).
    technique_ids: moi ID T#### xuat hien trong muc 4.
    """
    out = {"selected": None, "selected_raw": None, "tactic": None, "tactic_raw": None,
           "technique_ids": [], "severity": None, "top_indicators": None,
           "n_sections": 0, "parse_error": None}
    if not raw or not raw.strip():
        out["parse_error"] = "empty_output"
        return out

    sec2, sec4 = _section(raw, 2), _section(raw, 4)
    m = _SELECTED_RE.search(sec2 or raw)
    if m:
        txt = m.group(1).strip().strip("*`").strip()
        out["selected_raw"] = txt
        hits = [o for o in options if _norm(o) and _norm(o) in _norm(txt)]
        exact = [o for o in options if _norm(o) == _norm(txt)]
        pick = exact or hits
        if len(pick) == 1:
            out["selected"] = pick[0]
        else:
            out["parse_error"] = "selected_not_an_option" if not pick else "selected_ambiguous"
    else:
        out["parse_error"] = "selected_not_found"

    for src in (sec2, sec4):
        m = _TACTIC_RE.search(src or "")
        if m:
            txt = m.group(1).strip().strip("*`").strip()
            out["tactic_raw"] = out["tactic_raw"] or txt
            hit = [t for t in MITRE_TACTICS if _norm(t) == _norm(txt)] or \
                  [t for t in MITRE_TACTICS if _norm(t) in _norm(txt)]
            if hit:
                out["tactic"] = max(hit, key=len)
                break

    out["technique_ids"] = sorted(set(_MITRE_RE.findall(sec4 or "")))
    m = _SEV_RE.search(raw)
    if m:
        out["severity"] = m.group(1).capitalize()
    m = _TOPIND_RE.search(raw)
    if m:
        out["top_indicators"] = m.group(1).strip().strip("*").strip()
    out["n_sections"] = sum(1 for i in range(1, 7) if _section(raw, i))
    if out["n_sections"] < 6:
        out["parse_error"] = (out["parse_error"] or "") + f"|only_{out['n_sections']}_sections"
    return out


# =============================================================================
# Nhanh BAT CO "khong co tien le gan" (thiet ke chot 2026-11-13; thay unknown_anomaly.json)
#
# Day la CHI DAN cho VLM, khong phai tri thuc, nen nam o prompts.py chu khong o knowledge_base/:
#   FLAGGED_NOTICE    code dien so (s, tau_zd, nhan gan nhat, so tien le) -> dat dau context
#   FLAGGED_ADDENDUM  chi them khi co bat: phan tich khac biet qua bang C, mot danh gia trong 3 lua chon
#   render_contrast_table   bang C (evidence/contrast.py) -> markdown voi ID [C1].. de VLM trich
#   parse_assessment        doc danh gia cua VLM (cham tren Protocol A: loai bi giu lai vs loai da biet)
# Chua noi vao luong prompt hien tai (build_candidate_prompt): viec do thuoc dot viet lai prompt (D6).
# =============================================================================
FLAGGED_NOTICE = """\
NOVELTY FLAG: this alert is unusual with respect to the reference set of known attack flows.
Its mean distance to its {k} nearest reference flows (s = {novelty:.4f}) is above the threshold
(tau = {tau:.4f}) that {pct:.0f}% of known alerts fall under. The reference set holds no close
precedent; this does not establish that the attack type is new.
Nearest known type (for comparison only): {label} ({n} of the {k_vote} nearest precedents).\
"""

FLAGGED_ADDENDUM = """\
The novelty flag is raised, so this alert may be an attack behaviour the system has not seen.
Do not assign an attack type, ATT&CK technique or tactic. The profile of {label} is given only
to compare against.

In section 4:
1. Using table C, state how this alert differs from the {label} precedents: in the gate's
   reasons (phi) and in the traffic values (x). Cite the rows, e.g. [C1].
2. For each behaviour in the {label} profile, state whether it is observed, contradicted, or
   not observable in this alert. Cite [K:...] or [E...].
3. Give exactly one assessment, supported by the cited rows:
   - "Variant of {label}": the differences are of degree; the profile's behaviours hold.
   - "Different behaviour": key behaviours of the profile are contradicted.
   - "Undetermined": the evidence does not decide.
Do not name another attack type, technique or tactic.\
"""

# Dau khoi KNOWLEDGE cua nhanh bat co: cac doan B (role "nearest_background" trong knowledge_base/retrieve.py) da an
# ID / tieu de technique (cite REF-MECH-n); cau nay noi ro chung la boi canh cua tien le gan nhat, khong cua alert.
NEAREST_BACKGROUND_HEADER = """\
#### nearest_background
Nearest-precedent background (for comparison only). These passages describe the attack type of the closest
known precedent, not this alert. Use them to judge where this alert resembles or departs from that known
behaviour; do not attribute their technique, tactic or name to this alert.\
"""

ASSESSMENTS = ("variant", "different", "undetermined")
_ASSESS_RE = re.compile(r"(variant of|different behaviou?r|undetermined)", re.IGNORECASE)


def render_flagged_notice(look, k_vote=5, quantile=0.95):
    """look: retrieval.store.Lookup cua alert bi bat co."""
    return FLAGGED_NOTICE.format(k=len(look.neighbours), novelty=look.novelty, tau=look.tau_zd,
                                 pct=100 * quantile, label=look.label, n=len(look.winners), k_vote=k_vote)


def render_contrast_table(table):
    """Bang C (evidence/contrast.contrast_table) -> markdown, moi dong co ID de trich [C1]."""
    def num(v):
        return "-" if v is None else (v if isinstance(v, str) else f"{v:.6g}")
    n, lab = table["n_precedents"], table["label"]
    lines = [f"Table C: this alert vs the mean of its {n} nearest {lab} precedents "
             "(phi = SHAP contribution in log-odds; x = raw flow value).",
             "| ID | Feature | phi alert | phi precedents | delta phi | x alert | x precedents |",
             "|---|---|---|---|---|---|---|"]
    for r in table["rows"]:
        lines.append(f"| {r['id']} | {r['feature']} | {r['phi_alert']:+.3f} | {r['phi_ref']:+.3f} | "
                     f"{r['delta_phi']:+.3f} | {num(r['x_alert'])} | {num(r['x_ref'])} |")
    return "\n".join(lines)


def parse_assessment(raw):
    """'variant' | 'different' | 'undetermined' | None tu muc 4 cua report (lay lan xuat hien cuoi)."""
    hits = _ASSESS_RE.findall(_section(raw, 4) or raw or "")
    if not hits:
        return None
    h = hits[-1].lower()
    return "variant" if h.startswith("variant") else "different" if h.startswith("different") else "undetermined"
