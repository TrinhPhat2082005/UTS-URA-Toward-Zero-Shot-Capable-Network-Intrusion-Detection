"""
Prompt + parser cua Layer 2 (bao cao 6 muc co trich dan).

Dau vao: ban ghi bang chung cua MOT alert (evidence/build.py: EvidenceBuilder.build). Hai nhanh theo co cua
buoc tra tap R (retrieval/store.py):

  co TAT  nhan l* = nhan tien le gan nhat (vote 5-NN). KB: M + B la boi canh cua l* (attack_type_background),
          H1/H2 dinh nghia, H3 canh bao, G ung vien mitigation. Bao cao gan technique tu [K:M-...] va chon
          mitigation trong [K:G-...].
  co BAT  khong co tien le gan. Them thong bao co + bang C (alert so voi tien le cua l*). B la nearest_background
          (an ID, cite REF-MECH-n); khong M, khong G. Bao cao khong gan technique; danh gia variant /
          different / undetermined.

Trich dan trong bao cao: [I1]..[I5] anh, [E1]..[E7] bang chung so cua alert, [C1].. dong bang C,
[K:<id>] doan KB. build_report_prompt tra ve them `contexts` (moi khoi text co id, de danh gia RAGAS) va
`allowed_ids` (moi id duoc phep trich, de cham groundedness).

    KB_MODE "off": khong co khoi KNOWLEDGE (ablation khong KB); KB_PARTS loc nhom doan gui di.
"""
from __future__ import annotations

import re

KB_ALL_PARTS = ("M", "B", "H1", "H2", "H3", "G")
UNKNOWN = "UNKNOWN"
ATTACK_CLASSES = (
    "Bot", "Brute_Force_Web", "Brute_Force_XSS", "DDOS_attack_HOIC", "DDOS_attack_LOIC_UDP",
    "DDoS_attacks_LOIC_HTTP", "DoS_attacks_GoldenEye", "DoS_attacks_Hulk", "DoS_attacks_SlowHTTPTest",
    "DoS_attacks_Slowloris", "FTP_BruteForce", "Infiltration", "SQL_Injection", "SSH_Bruteforce",
)
IMAGES = (
    ("I1", "SHAP waterfall of Layer 1: per-feature contributions to the attack score; positive bars push "
           "towards attack. ae_latent_* and ae_recon_error are autoencoder features, not traffic fields."),
    ("I2", "Packet length profile (forward vs backward, min / mean / max, bytes)"),
    ("I3", "TCP flags of the first packet of the flow (0/1)"),
    ("I4", "Inter-arrival timing by direction (microseconds, mean and standard deviation)"),
    ("I5", "Directional balance of packets and bytes (forward vs backward)"),
)

# --------------------------------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a Tier-3 SOC analyst writing the forensic report for one network intrusion alert.

Write the entire report in English.

Every statement of fact must be supported by the material given in the task and must end with the
identifiers of what supports it, in square brackets: [I1]-[I5] for the images, [E1]-[E7] for the alert
evidence, [C1], [C2], ... for rows of table C, and [K:<id>] for knowledge passages, for example
[E2][K:H1-Flow_Duration]. Cite only identifiers that appear in the task. If a detail is not in the
material, write "not available" instead of inventing it.

Knowledge passages describe attack types, features and protocols in general; they do not describe this
alert. Use them to interpret the evidence, not as evidence that something happened.

Use the exact section headings given in the task. Do not add, omit or reorder sections.\
"""

# --------------------------------------------------------------------------------------------------
# Nhanh bat co: thong bao, dau khoi nearest_background, cach viet muc 4
# --------------------------------------------------------------------------------------------------
FLAGGED_NOTICE = """\
NOVELTY FLAG: this alert is unusual with respect to the reference set of known attack flows.
Its mean distance to its {k} nearest reference flows (s = {novelty:.4f}) is above the threshold
(tau = {tau:.4f}) that {pct:.0f}% of known alerts fall under. The reference set holds no close
precedent; this does not establish that the attack type is new.
Nearest known type (for comparison only): {label} ({n} of the {k_vote} nearest precedents).\
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

# --------------------------------------------------------------------------------------------------
# Khoi KB theo role (knowledge_base/retrieve.py)
# --------------------------------------------------------------------------------------------------
_BLOCKS = (
    ("attack_type_background", "#### attack_type_background\n"
     "Background on the attack type of the nearest reference label {label}. It describes that attack type "
     "in general, not this alert."),
    ("nearest_background", NEAREST_BACKGROUND_HEADER),
    ("definition", "#### definitions\nDefinitions of the features, protocol, port and TCP flags that appear "
     "in the evidence, for the CICFlowMeter version that produced this data."),
    ("caveat", "#### caveats\nLimits of what this flow record can show."),
    ("mitigation_option", "#### mitigation_options\nCandidate mitigations for the technique of {label}. "
     "Choose only those that fit what this alert shows."),
)

# --------------------------------------------------------------------------------------------------
# Schema bao cao
# --------------------------------------------------------------------------------------------------
_SCHEMA_KNOWN = """\
Produce the report in exactly this form.

## 1. EXECUTIVE SUMMARY
Two to four sentences: what was detected, the reference label, and whether the evidence supports it.

## 2. ATTACK ASSESSMENT
Predicted class: <the reference label {label}, copied exactly; write UNKNOWN only if the evidence clearly contradicts it>
Severity: <Low, Medium, High or Critical>
Top indicators: <three feature names copied from image [I1]>

## 3. FORENSIC REASONING
Bullets. Each bullet links one observation in the evidence ([I..], [E..]) to what it means ([K:H1-..],
[K:H2-..]) and to the attack type ([K:B-..]). Say where the evidence and the background agree or disagree.

## 4. MITRE MAPPING
{mitre}

## 5. MITIGATIONS
{mitig}

## 6. ANALYST NOTES
Limits of the evidence and next steps for the analyst{caveat_hint}.\
"""

_SCHEMA_FLAGGED = """\
Produce the report in exactly this form.

## 1. EXECUTIVE SUMMARY
Two to four sentences: what was detected, that the reference set holds no close precedent, and what the
evidence shows.

## 2. ATTACK ASSESSMENT
Predicted class: UNKNOWN
Severity: <Low, Medium, High or Critical>
Top indicators: <three feature names copied from image [I1]>

## 3. FORENSIC REASONING
Bullets. Each bullet links one observation in the evidence ([I..], [E..]) to what it means ([K:H1-..],
[K:H2-..]). Then state how this alert differs from the {label} precedents, in the gate's reasons (phi) and in
the traffic values (x), citing the rows of table C ([C1], ...).

## 4. MITRE MAPPING
Technique: N/A (no technique is assigned to a flagged alert).
Assessment: <exactly one of: Variant of {label} | Different behaviour | Undetermined>
Then two or three sentences that justify the assessment from table C{refmech_hint}. Do not name an
attack type, technique or tactic for this alert.

## 5. MITIGATIONS
N/A: no technique is assigned to a flagged alert.

## 6. ANALYST NOTES
Limits of the evidence and next steps for the analyst{caveat_hint}.\
"""


def _fmt(v, nd=6):
    if v is None:
        return "not available"
    if isinstance(v, str):
        return v
    return f"{v:.{nd}g}"


def render_flagged_notice(lookup, k_vote=5, quantile=0.95):
    """lookup: ban ghi 'lookup' cua evidence (hoac retrieval.store.Lookup)."""
    g = lookup.get if isinstance(lookup, dict) else (lambda k: getattr(lookup, k))
    n_nb = len(g("neighbours"))
    n_win = g("n_winners") if isinstance(lookup, dict) else len(lookup.winners)
    return FLAGGED_NOTICE.format(k=n_nb, novelty=g("novelty"), tau=g("tau_zd"), pct=100 * quantile,
                                 label=g("label"), n=n_win, k_vote=k_vote)


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


# --------------------------------------------------------------------------------------------------
# Dung prompt
# --------------------------------------------------------------------------------------------------
def build_report_prompt(ev, kb_mode="on", kb_parts=KB_ALL_PARTS):
    """Prompt cho MOT alert.

    Args:
        ev: ban ghi cua EvidenceBuilder.build (alert = True).
        kb_mode: "on" | "off".
        kb_parts: cac nhom KB duoc gui (tap con cua KB_ALL_PARTS).

    Returns:
        {"system", "user", "contexts": [{"id", "kind", "text"}], "allowed_ids": [..], "flagged", "label"}
    """
    look = ev["lookup"]
    flagged, label = bool(look["flagged"]), look["label"]
    use_kb = kb_mode == "on"
    contexts, allowed = [], [i for i, _ in IMAGES]

    # --- bang chung cua alert --------------------------------------------------------------------
    ev_lines = [f"Layer 1 attack probability p(x) = {ev['p']:.4f} (alert threshold {ev['tau_alpha']:.4f}).",
                "Largest SHAP contributions (log-odds; positive pushes towards attack):"]
    for f in ev["top_features"]:
        raw = "" if f["raw_value"] is None else f"; raw value {_fmt(f['raw_value'])}"
        ev_lines.append(f"[{f['id']}] {f['name']}: SHAP {f['phi']:+.3f}{raw}")
    if ev["x_facts"]:
        ev_lines.append("Flow facts:")
        ev_lines += [f"[{f['id']}] {f['name']} = {_fmt(f['raw_value'])}" for f in ev["x_facts"]]
    evidence_text = "\n".join(ev_lines)
    allowed += [f["id"] for f in ev["top_features"]] + [f["id"] for f in ev["x_facts"]]
    contexts.append({"id": "EVIDENCE", "kind": "evidence", "text": evidence_text})

    # --- ket qua tra tap R -----------------------------------------------------------------------
    if flagged:
        lookup_text = render_flagged_notice(look)
        if ev.get("contrast"):
            table_text = render_contrast_table(ev["contrast"])
            lookup_text += "\n\n" + table_text
            allowed += [r["id"] for r in ev["contrast"]["rows"]]
            contexts.append({"id": "TABLE_C", "kind": "contrast", "text": table_text})
    else:
        n_lab = look["votes"].get(label, 0)
        lookup_text = (f"Nearest reference label: {label} ({n_lab} of the 5 nearest known attack flows). "
                       f"Novelty s = {look['novelty']:.4f}, below the threshold tau = {look['tau_zd']:.4f}: "
                       "the reference set holds close precedents of this alert.")
    contexts.append({"id": "LOOKUP", "kind": "lookup", "text": lookup_text})

    # --- doan KB -------------------------------------------------------------------------------
    kb_text, cites = "", []
    if use_kb:
        parts = set(kb_parts)
        ps = [p for p in ev["kb"]["passages"] if p["part"] in parts]
        blocks = []
        for role, head in _BLOCKS:
            sel = [p for p in ps if p["role"] == role]
            if not sel:
                continue
            body = []
            for p in sel:
                body.append(f"[K:{p['cite']}] {p['title']}\n{p['text']}")
                cites.append(f"K:{p['cite']}")
                contexts.append({"id": f"K:{p['cite']}", "kind": f"kb_{p['part']}", "text": p["text"]})
            blocks.append(head.format(label=label) + "\n\n" + "\n\n".join(body))
        if blocks:
            kb_text = ("### KNOWLEDGE BASE EXCERPTS\nCite a passage as [K:<id>], for example [K:"
                       f"{ps[0]['cite']}].\n\n" + "\n\n".join(blocks))
    allowed += cites
    has = {c.split("-")[0] for c in cites}       # "K:M", "K:B", "K:H1", "K:G", "K:REF"

    # --- schema -------------------------------------------------------------------------------
    caveat_hint = " (cite the caveat passages [K:H3-..] or [K:H2-PORT-NOTE] where they apply)" \
        if any(c.startswith(("K:H3-", "K:H2-PORT-NOTE")) for c in cites) else ""
    if flagged:
        refmech = " and from the nearest_background passages ([K:REF-MECH-..])" if "K:REF" in has else ""
        schema = _SCHEMA_FLAGGED.format(label=label, refmech_hint=refmech, caveat_hint=caveat_hint)
    else:
        mitre = ("Tactic and technique copied from [K:M-" + label + "], then one sentence on how the observed "
                 "traffic fits the technique, with citations.") if "K:M" in has else \
                ("Tactic and technique ID for the predicted class, then one sentence on how the observed traffic "
                 "fits it.")
        mitig = ("Choose two to four of the mitigation options [K:G-..] that fit what this alert shows. For each "
                 "give its identifier, then one sentence on why it fits this alert, citing the evidence.") \
            if "K:G" in has else "Two to four ranked recommendations that fit what this alert shows."
        schema = _SCHEMA_KNOWN.format(label=label, mitre=mitre, mitig=mitig, caveat_hint=caveat_hint)

    head = ("Layer 1 (an XGBoost gate) raised this alert. Its SHAP attribution vector was looked up in a "
            "reference set of known attack flows. Write the forensic report for this alert.")
    images = "### IMAGES\n" + "\n".join(f"[{i}] {d}" for i, d in IMAGES)
    user = "\n\n".join(s for s in (head, images, "### ALERT EVIDENCE\n" + evidence_text,
                                   "### REFERENCE LOOKUP\n" + lookup_text, kb_text,
                                   "### REPORT FORMAT\n" + schema) if s)
    return {"system": SYSTEM_PROMPT, "user": user, "contexts": contexts, "allowed_ids": allowed,
            "flagged": flagged, "label": label}


# --------------------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------------------
_CLASS_RE = re.compile(r"predicted\s*class\s*[::]\s*\**\s*([^\n*]+)", re.IGNORECASE)
_SEV_RE = re.compile(r"severity\s*[::]\s*\**\s*(low|medium|high|critical)", re.IGNORECASE)
_TOPIND_RE = re.compile(r"top\s*indicators?\s*[::]\s*\**\s*([^\n]+)", re.IGNORECASE)
_MITRE_RE = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b")
_TACTIC_RE = re.compile(r"tactic\s*[::]\s*\**\s*([^\n*]+)", re.IGNORECASE)
_BRACKET_RE = re.compile(r"\[([^\[\]\n]{1,200})\]")
_TOKEN_RE = re.compile(r"^(?:K:\s*[A-Za-z0-9_.\-]+|[IEC]\d{1,2})$")
SECTION_NAMES = {1: "executive_summary", 2: "attack_assessment", 3: "forensic_reasoning",
                 4: "mitre_mapping", 5: "mitigations", 6: "analyst_notes"}


def _section(text, num):
    """Noi dung muc <num> theo tieu de '## <num>.'."""
    m = re.search(rf"^#{{1,4}}\s*{num}\.\s*[^\n]*\n(.*?)(?=^#{{1,4}}\s*\d+\.|\Z)",
                  text or "", re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else ""


def citations(text):
    """Moi id trich dan trong text, theo thu tu xuat hien: 'I1', 'E2', 'C3', 'K:H1-Flow_Duration'.
    Chap nhan nhieu id trong mot ngoac: [E1, K:H1-X]."""
    out = []
    for m in _BRACKET_RE.finditer(text or ""):
        for tok in re.split(r"[,;]", m.group(1)):
            tok = tok.strip()
            if _TOKEN_RE.match(tok):
                out.append(re.sub(r"^K:\s*", "K:", tok))
    return out


def parse_assessment(raw):
    """'variant' | 'different' | 'undetermined' | None tu muc 4 (lay lan xuat hien cuoi); khong co muc 4 thi tim
    trong ca van ban."""
    hits = _ASSESS_RE.findall(_section(raw, 4) or raw or "")
    if not hits:
        return None
    h = hits[-1].lower()
    return "variant" if h.startswith("variant") else "different" if h.startswith("different") else "undetermined"


def parse_report(raw, allowed_ids=None):
    """Tach cac truong may doc duoc tu bao cao. Khong doan khi thieu: truong khong tim thay -> None + parse_error."""
    out = {"predicted_class": None, "severity": None, "top_indicators": None, "mitre_id": None,
           "mitre_tactic": None, "assessment": None, "sections": {}, "n_sections": 0,
           "citations": [], "citations_by_section": {}, "mitigations_chosen": [], "unknown_citations": [],
           "parse_error": None}
    if not raw or not raw.strip():
        out["parse_error"] = "empty_output"
        return out

    m = _CLASS_RE.search(raw)
    if m:
        out["predicted_class"] = m.group(1).strip().strip("*`").strip()
    else:
        out["parse_error"] = "predicted_class_not_found"
    m = _SEV_RE.search(raw)
    if m:
        out["severity"] = m.group(1).capitalize()
    m = _TOPIND_RE.search(raw)
    if m:
        out["top_indicators"] = m.group(1).strip().strip("*").strip()

    for i, name in SECTION_NAMES.items():
        s = _section(raw, i)
        if s:
            out["sections"][name] = s
            out["citations_by_section"][name] = citations(s)
    out["n_sections"] = len(out["sections"])
    if out["n_sections"] < 6:
        out["parse_error"] = (out["parse_error"] or "") + f"|only_{out['n_sections']}_sections"

    sec4 = out["sections"].get("mitre_mapping", "")
    m = _MITRE_RE.search(sec4)
    if m:
        out["mitre_id"] = m.group(1)
    m = _TACTIC_RE.search(sec4)
    if m:
        out["mitre_tactic"] = m.group(1).strip().strip("*").strip()
    out["assessment"] = parse_assessment(raw)

    out["citations"] = citations(raw)
    out["mitigations_chosen"] = list(dict.fromkeys(
        c for c in out["citations_by_section"].get("mitigations", []) if c.startswith("K:G-")))
    if allowed_ids is not None:
        ok = set(allowed_ids)
        out["unknown_citations"] = sorted({c for c in out["citations"] if c not in ok})
    return out
