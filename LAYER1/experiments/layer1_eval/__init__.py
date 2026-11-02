"""Danh gia Layer 1 (RQ1, RQ2 — Protocol A, 14 fold leave-one-attack-out).

    paths    duong dan (bien moi truong)        data     doc split CSV
    scorers  checkpoint -> score(X)             scores   diem 4 phan moi fold, co cache
    metrics  nguong theo FPR, F1 / ZS-Acc / pAUC
    tables   bang tren slide (layer1_results.ipynb)
    extras   latency, hinh, bang nguong rieng tung fold, kiem tra tai hien (layer1_extras.ipynb)
"""
