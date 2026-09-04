# k-NN vs Centroid retrieval eval — cap=20,000/label, Infilteration excluded

Generated 2026-08-23. Source data: `shap_full_{variant}.npz` (20K-cap, see
`kaggle_build_all_variants_shap.py --cap 20000 --seed 42`). Eval script:
`eval_retrieval_no_infilt.py` (70/30 build/eval split per label, cosine
similarity). Infilteration excluded from both build pool and eval set
(zero-shot label, not a fair retrieval target — see PROJECT_STATE.md /
CLAUDE.md note on `binary_split_dataset_z_Infilteration`).

## Macro-average summary

| Variant | Method | Top-1 | Top-3 | Top-5 |
|---|---|---|---|---|
| XGBoost-only | centroid | 0.691 | 0.932 | 0.979 |
| XGBoost-only | k-NN | 0.763 | 0.996 | 1.000 |
| AE-augmented | centroid | 0.586 | 0.897 | 0.959 |
| AE-augmented | k-NN | 0.819 | 0.999 | 1.000 |
| SVDD-augmented | centroid | 0.728 | 0.914 | 0.982 |
| SVDD-augmented | k-NN | 0.778 | 0.988 | 1.000 |

k-NN outperforms centroid-mean at every variant (+7.2pp to +23.3pp top-1),
consistently with the earlier 2K-cap run. AE-augmented + k-NN is the best
overall config at this scale (top1=0.819).

## Full per-label tables

### XGBoost-only

```
--- method: centroid ---
label                         n_build  n_eval   top1   top3   top5
Benign                          14000    6000  0.864  0.885  0.915
Bot                             14000    6000  0.924  0.940  0.942
Brute Force -Web                  428     183  0.000  0.727  0.869
Brute Force -XSS                  161      69  0.623  0.841  1.000
DDOS attack-HOIC                14000    6000  0.767  0.826  1.000
DDOS attack-LOIC-UDP             1211     519  1.000  1.000  1.000
DDoS attacks-LOIC-HTTP          14000    6000  0.507  0.971  0.998
DoS attacks-GoldenEye           14000    6000  0.647  1.000  1.000
DoS attacks-Hulk                14000    6000  0.914  0.945  0.997
DoS attacks-SlowHTTPTest        14000    6000  0.117  1.000  1.000
DoS attacks-Slowloris            7693    3297  0.850  0.989  0.990
FTP-BruteForce                  14000    6000  0.885  1.000  1.000
SQL Injection                      61      26  0.577  0.923  1.000
SSH-Bruteforce                  14000    6000  0.999  1.000  1.000
MACRO AVERAGE                                  0.691  0.932  0.979

--- method: knn ---
label                         n_build  n_eval   top1   top3   top5
Benign                          14000    6000  0.998  1.000  1.000
Bot                             14000    6000  1.000  1.000  1.000
Brute Force -Web                  428     183  0.634  1.000  1.000
Brute Force -XSS                  161      69  0.812  0.986  1.000
DDOS attack-HOIC                14000    6000  0.699  1.000  1.000
DDOS attack-LOIC-UDP             1211     519  0.081  1.000  1.000
DDoS attacks-LOIC-HTTP          14000    6000  1.000  1.000  1.000
DoS attacks-GoldenEye           14000    6000  1.000  1.000  1.000
DoS attacks-Hulk                14000    6000  1.000  1.000  1.000
DoS attacks-SlowHTTPTest        14000    6000  0.000  1.000  1.000
DoS attacks-Slowloris            7693    3297  1.000  1.000  1.000
FTP-BruteForce                  14000    6000  1.000  1.000  1.000
SQL Injection                      61      26  0.462  0.962  1.000
SSH-Bruteforce                  14000    6000  1.000  1.000  1.000
MACRO AVERAGE                                  0.763  0.996  1.000
```

### AE-augmented

```
--- method: centroid ---
label                         n_build  n_eval   top1   top3   top5
Benign                          14000    6000  0.853  0.955  0.956
Bot                             14000    6000  0.493  0.994  0.999
Brute Force -Web                  428     183  0.175  0.585  0.639
Brute Force -XSS                  161      69  0.014  0.957  0.957
DDOS attack-HOIC                14000    6000  0.768  1.000  1.000
DDOS attack-LOIC-UDP             1211     519  1.000  1.000  1.000
DDoS attacks-LOIC-HTTP          14000    6000  0.496  0.993  1.000
DoS attacks-GoldenEye           14000    6000  0.773  0.850  1.000
DoS attacks-Hulk                14000    6000  0.848  0.965  0.970
DoS attacks-SlowHTTPTest        14000    6000  0.099  1.000  1.000
DoS attacks-Slowloris            7693    3297  0.704  0.911  0.911
FTP-BruteForce                  14000    6000  0.901  1.000  1.000
SQL Injection                      61      26  0.654  0.846  1.000
SSH-Bruteforce                  14000    6000  0.421  0.499  1.000
MACRO AVERAGE                                  0.586  0.897  0.959

--- method: knn ---
label                         n_build  n_eval   top1   top3   top5
Benign                          14000    6000  0.998  1.000  1.000
Bot                             14000    6000  1.000  1.000  1.000
Brute Force -Web                  428     183  0.650  1.000  1.000
Brute Force -XSS                  161      69  0.928  0.986  1.000
DDOS attack-HOIC                14000    6000  0.977  1.000  1.000
DDOS attack-LOIC-UDP             1211     519  0.372  1.000  1.000
DDoS attacks-LOIC-HTTP          14000    6000  0.999  1.000  1.000
DoS attacks-GoldenEye           14000    6000  1.000  1.000  1.000
DoS attacks-Hulk                14000    6000  1.000  1.000  1.000
DoS attacks-SlowHTTPTest        14000    6000  0.000  1.000  1.000
DoS attacks-Slowloris            7693    3297  1.000  1.000  1.000
FTP-BruteForce                  14000    6000  1.000  1.000  1.000
SQL Injection                      61      26  0.538  1.000  1.000
SSH-Bruteforce                  14000    6000  1.000  1.000  1.000
MACRO AVERAGE                                  0.819  0.999  1.000
```

### SVDD-augmented

```
--- method: centroid ---
label                         n_build  n_eval   top1   top3   top5
Benign                          14000    6000  0.864  0.864  0.864
Bot                             14000    6000  0.512  0.997  0.998
Brute Force -Web                  428     183  0.005  0.574  0.896
Brute Force -XSS                  161      69  0.884  0.957  1.000
DDOS attack-HOIC                14000    6000  0.768  0.768  1.000
DDOS attack-LOIC-UDP             1211     519  1.000  1.000  1.000
DDoS attacks-LOIC-HTTP          14000    6000  0.927  0.995  1.000
DoS attacks-GoldenEye           14000    6000  0.644  0.786  1.000
DoS attacks-Hulk                14000    6000  0.926  0.945  0.997
DoS attacks-SlowHTTPTest        14000    6000  0.993  1.000  1.000
DoS attacks-Slowloris            7693    3297  0.864  0.990  0.990
FTP-BruteForce                  14000    6000  0.338  1.000  1.000
SQL Injection                      61      26  0.462  0.923  1.000
SSH-Bruteforce                  14000    6000  1.000  1.000  1.000
MACRO AVERAGE                                  0.728  0.914  0.982

--- method: knn ---
label                         n_build  n_eval   top1   top3   top5
Benign                          14000    6000  0.998  1.000  1.000
Bot                             14000    6000  1.000  1.000  1.000
Brute Force -Web                  428     183  0.639  1.000  1.000
Brute Force -XSS                  161      69  0.826  0.986  1.000
DDOS attack-HOIC                14000    6000  0.970  1.000  1.000
DDOS attack-LOIC-UDP             1211     519  0.081  1.000  1.000
DDoS attacks-LOIC-HTTP          14000    6000  0.999  1.000  1.000
DoS attacks-GoldenEye           14000    6000  1.000  1.000  1.000
DoS attacks-Hulk                14000    6000  1.000  1.000  1.000
DoS attacks-SlowHTTPTest        14000    6000  0.000  1.000  1.000
DoS attacks-Slowloris            7693    3297  1.000  1.000  1.000
FTP-BruteForce                  14000    6000  1.000  1.000  1.000
SQL Injection                      61      26  0.385  0.846  1.000
SSH-Bruteforce                  14000    6000  1.000  1.000  1.000
MACRO AVERAGE                                  0.778  0.988  1.000
```

## Key finding: SlowHTTPTest / FTP-BruteForce collapse persists at scale

`DoS attacks-SlowHTTPTest` top-1 stays near-zero under k-NN across all 3
variants (0.000, 0.000, 0.000) and under centroid for XGBoost-only/AE
(0.117, 0.099) — confirms this is NOT a small-sample artifact. Root cause
traced to the original CIC-IDS2018 source data itself: 139,890/139,890
SlowHTTPTest-labeled flows in the public dataset are on Dst Port 21 (not
the tool's documented HTTP ports 80/443), making them feature-identical to
FTP-BruteForce flows. SVDD-augmented + centroid is the one exception
(top1=0.993) but this comes at the cost of FTP-BruteForce collapsing
instead (0.338) — the confusion direction flips, it does not disappear.
