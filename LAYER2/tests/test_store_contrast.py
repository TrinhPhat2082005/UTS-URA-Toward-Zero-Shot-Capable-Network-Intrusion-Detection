"""
Kiem retrieval/store.py (vote 5-NN, novelty, khop dong SHAP <-> gia tri tho) va evidence/contrast.py.

    python -m pytest LAYER2/tests -q
    python LAYER2/tests/test_store_contrast.py

Test tren R that (Protocol B, ae_xgboost) tu bo qua neu khong co file.
"""
import sys
from pathlib import Path

import numpy as np

LAYER2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAYER2))

import config  # noqa: E402
import prompts  # noqa: E402
from evidence.contrast import contrast_table  # noqa: E402
from retrieval.store import ReferenceStore  # noqa: E402

REAL_R = Path(config.REFERENCE_SETS_DIR) / "protocol_b" / "ae_xgboost.npz"


def toy_store(tau=None):
    # 12 vector 2 chieu: A gan goc, B o (1,0), C xa
    phi = np.array([[0, 0], [0.1, 0], [0, 0.1], [1, 0], [1.05, 0], [0.95, 0], [5, 5], [5, 5.1],
                    [5.1, 5], [5.2, 5], [5, 5.2], [5.1, 5.1]], dtype=float)
    labels = ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C", "C", "C"]
    x = np.arange(24, dtype=float).reshape(12, 2)
    return ReferenceStore(phi, labels, x=x, row_ids=np.arange(12), feature_names=["f0", "f1"],
                          base_feature_names=["f0", "f1"], tau_zd=tau)


def test_vote_majority_and_winners():
    look = toy_store().lookup([0.02, 0.0])
    assert look.label == "A" and look.votes["A"] == 3 and set(look.winners.tolist()) <= {0, 1, 2}
    assert len(look.neighbours) == 10 and np.all(np.diff(look.distances) >= 0)


def test_vote_tie_goes_to_label_of_nearest_flow():
    # tim 2 A + 2 B + 1 C trong 5-NN; diem gan B hon -> B thang khi hoa 2-2
    phi = np.array([[0, 0], [0.2, 0], [1, 0], [0.9, 0], [3, 0], [9, 9], [9, 9], [9, 9], [9, 9], [9, 9]], float)
    s = ReferenceStore(phi, ["A", "A", "B", "B", "C", "D", "D", "D", "D", "D"])
    look = s.lookup([0.8, 0.0])
    assert look.votes["A"] == 2 and look.votes["B"] == 2 and look.label == "B"


def test_novelty_flag_uses_mean_of_10_nn():
    s = toy_store(tau=0.5)
    near = s.lookup([5.05, 5.05])
    assert near.novelty == np.mean(near.distances) and near.flagged == (near.novelty > 0.5)


def test_contrast_table_shape_and_discrete_counts():
    s = toy_store()
    look = s.lookup([1.0, 0.02])
    t = contrast_table([1.0, 0.02], {"f0": 7.0, "f1": 1.0}, s, look.winners, look.label, discrete=["f1"])
    assert t["label"] == "B" and t["n_precedents"] == 3
    assert [r["id"] for r in t["rows"]] == [f"C{i}" for i in range(1, len(t["rows"]) + 1)]
    f1 = next(r for r in t["rows"] if r["feature"] == "f1")
    assert isinstance(f1["x_ref"], str) and "(1/3)" in f1["x_ref"]
    table = prompts.render_contrast_table(t)
    assert "| C1 |" in table


def test_parse_assessment():
    rep = "4. Comparison\n... so overall: Different behaviour, see [C1].\n5. Reliability\n..."
    assert prompts.parse_assessment(rep) in ("different", None)
    assert prompts.parse_assessment("Variant of DoS_attacks_Hulk") == "variant"
    assert prompts.parse_assessment("Assessment: Undetermined") == "undetermined"


def test_real_r_rows_align_with_train_csv():
    if not REAL_R.exists():
        return
    import pandas as pd
    s = ReferenceStore.load(REAL_R)
    assert s.x is not None and s.row_ids is not None, "R chua co x__/row__: chay reference_set.py --add-raw"
    df = pd.read_csv(config.TRAIN_FULL_CSV)
    pick = np.random.RandomState(0).choice(len(s), 50, replace=False)
    got = s.x[pick]
    want = df.iloc[s.row_ids[pick]][s.base_feature_names].to_numpy(dtype="float64")
    assert np.array_equal(got, want), "x trong R khong khop dong goc"
    assert (df.iloc[s.row_ids[pick]]["Label_Text"].to_numpy() == s.labels[pick]).all()


def test_real_r_contrast_groups_latents():
    if not REAL_R.exists():
        return
    s = ReferenceStore.load(REAL_R)
    i = 12345
    q = s.phi[i] + np.random.RandomState(1).normal(0, 0.05, s.phi.shape[1])
    look = s.lookup(q)
    x_q = dict(zip(s.base_feature_names, s.x[i]))
    t = contrast_table(q, x_q, s, look.winners, look.label, discrete=["Dst Port", "Protocol"])
    assert 1 <= len(t["rows"]) <= 8
    assert sum(r["feature"].startswith("ae_latent (") for r in t["rows"]) <= 1
    assert not any(r["feature"].startswith("ae_latent_") for r in t["rows"])


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("  ok", t.__name__)
    print(f"{len(tests)} test qua")
