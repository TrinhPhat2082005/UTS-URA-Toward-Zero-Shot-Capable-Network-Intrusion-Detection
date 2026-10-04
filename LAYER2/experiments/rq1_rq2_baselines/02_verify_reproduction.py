"""Step 2 - check that the checkpoints are loaded and preprocessed as their trainer did.

Applies each trainer's OWN threshold (from its metrics.json) to the scores of step 1 and
compares the resulting macro metrics with the trainer's reported summary. A match to 4
decimals means the scoring path of baselines.py reproduces the original evaluation, so
the re-thresholded numbers of step 3 refer to the same models.

Output: OUT_BASE/verify.md
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from baselines import REGISTRY, folds, own_threshold  # noqa: E402

OUT = paths.ensure(paths.OUT_BASE)


def metrics_file(name):
    root = paths.BASE_ART / name
    for cand in [root / "metrics.json"] + sorted(root.glob("*metrics*.json")):
        if cand.exists():
            return cand
    return None


def evaluate(S, name, fold, t, strict):
    hit = (lambda a: a > t) if strict else (lambda a: a >= t)
    vb, va, zs = (S[f"{name}|{fold}|{p}"] for p in ("vb", "va", "zs"))
    tp, fp = int(hit(va).sum()), int(hit(vb).sum())
    pre = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / len(va)
    return {"zs": float(hit(zs).mean()), "f1": 2 * pre * rec / (pre + rec) if pre + rec else 0.0,
            "fpr": fp / len(vb)}


def main():
    lines = ["| Model | Trainer's rule | ZS-Acc reported / rescored | KA-F1 reported / rescored | "
             "KA-FPR reported / rescored | Match |", "|---|---|---|---|---|---|"]
    for name in REGISTRY:
        f = OUT / f"scores_{name}.npz"
        mf = metrics_file(name)
        if not f.exists() or mf is None:
            continue
        S, m = np.load(f), json.loads(mf.read_text(encoding="utf-8"))
        rows = []
        for fold in folds():
            t, strict = own_threshold(name, m, fold)
            rows.append(evaluate(S, name, fold, t, strict))
        got = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
        rep = {"zs": m["summary"]["zero_shot_macro_avg"]["accuracy"],
               "f1": m["summary"]["known_attack_macro_avg"]["f1"],
               "fpr": m["summary"]["known_attack_macro_avg"]["fpr"]}
        ok = all(abs(got[k] - rep[k]) < 5e-4 for k in rep)
        _, strict0 = own_threshold(name, m, folds()[0])
        lines.append(f"| {REGISTRY[name].label} | {'>' if strict0 else '>='} {mf.name} threshold | "
                     f"{rep['zs']:.4f} / {got['zs']:.4f} | {rep['f1']:.4f} / {got['f1']:.4f} | "
                     f"{rep['fpr']:.4f} / {got['fpr']:.4f} | {'yes' if ok else 'NO'} |")
    text = "## Reproduction of each trainer's own evaluation\n\n" + "\n".join(lines) + "\n"
    (OUT / "verify.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
