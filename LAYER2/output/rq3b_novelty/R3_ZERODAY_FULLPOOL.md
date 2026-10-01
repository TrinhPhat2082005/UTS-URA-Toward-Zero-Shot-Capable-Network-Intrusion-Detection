# R3 — Zero-day voi pool DAY DU (khop pipeline)

Ngay chay: 2026-10-01. Moi round: pool = TOAN BO attack trong train_benign_attack.csv cua round (13 loai, 106,090–122,048 flow, khong benign, khong gioi han); query = TOAN BO alert (known: attack cua val_binary; unknown: loai bi giau) o nguong FPR alpha tren benign train. Novelty = trung binh L2 toi 10 lang gieng gan nhat. Round hop le: >= 20 alert unknown va known. Wilcoxon cap theo round. End-to-end = ZS-Acc x Detect@5%, round khong hop le tinh = 0. Dau ‡ = cap song sinh FTP/SlowHTTPTest.

Thay the ket qua pool gioi han 2,000/loai trong R3_ZERODAY.md.

## alpha = 0.4%

### XGBoost-only — 12 round hop le (pool = toan bo attack train cua round, 106,090–122,048 flow)

| Khoa | AUROC macro (tat ca) | AUROC macro (bo song sinh) | Detect@5% macro |
|---|---|---|---|
| SHAP day du | 0.950 | 0.960 | 0.774 |
| SHAP 78 | 0.950 | 0.960 | 0.774 |
| Feature tho | 0.801 | 0.935 | 0.705 |

- SHAP day du vs Feature tho: chenh TB +0.149, hon o 10/12 round, Wilcoxon p = 0.034
- SHAP 78 vs Feature tho: chenh TB +0.149, hon o 10/12 round, Wilcoxon p = 0.034
- SHAP day du vs SHAP 78: chenh TB +0.000, hon o 0/12 round, Wilcoxon p = 1.000
- End-to-end zero-day (ZS-Acc x Detect@5%, SHAP day du, 14 round, round khong hop le = 0): 0.4213 (bo song sinh: 0.4436)

| Round | n unk | n known | ZS-Acc | Trung lap (unk) | AUROC SHAP day du | AUROC SHAP 78 | AUROC tho | Det@5% SHAP | Tactic (SHAP) | Tactic (tho) |
|---|---|---|---|---|---|---|---|---|---|---|
| Bot | 0 | 23341 | 0.000 | — | — | — | — | — | — | — |
| Brute_Force_Web | 263 | 27,220 | 0.474 | 0.065 | 0.990 | 0.990 | 0.929 | 0.909 | 0.000 | 0.000 |
| Brute_Force_XSS | 220 | 27,234 | 0.965 | 0.055 | 0.983 | 0.983 | 0.964 | 0.923 | 0.214 | 0.009 |
| DDOS_attack_HOIC | 259 | 23,324 | 0.013 | 0.000 | 1.000 | 1.000 | 0.999 | 1.000 | 0.004 | 0.015 |
| DDOS_attack_LOIC_UDP | 1,730 | 26,915 | 1.000 | 0.013 | 0.967 | 0.967 | 0.962 | 0.912 | 1.000 | 1.000 |
| DDoS_attacks_LOIC_HTTP | 43 | 23,308 | 0.002 | 0.326 | 0.706 | 0.706 | 0.771 | 0.372 | 0.767 | 0.767 |
| DoS_attacks_GoldenEye | 19,919 | 23,312 | 0.996 | 0.000 | 1.000 | 1.000 | 0.998 | 1.000 | 0.667 | 0.816 |
| DoS_attacks_Hulk | 19,838 | 23,302 | 0.992 | 0.000 | 0.998 | 0.998 | 0.990 | 1.000 | 0.498 | 0.805 |
| DoS_attacks_SlowHTTPTest ‡ | 55 | 27,270 | 1.000 | 0.964 | 0.895 | 0.895 | 0.153 | 0.255 | 0.000 | 0.000 |
| DoS_attacks_Slowloris | 7,619 | 25,346 | 0.769 | 0.000 | 0.995 | 0.995 | 0.997 | 1.000 | 0.948 | 0.970 |
| FTP_BruteForce ‡ | 53 | 27,282 | 1.000 | 0.981 | 0.907 | 0.907 | 0.110 | 0.321 | 0.019 | 0.019 |
| Infiltration | 195 | 26,523 | 0.010 | 0.010 | 0.999 | 0.999 | 0.972 | 0.995 | 0.000 | 0.000 |
| SQL_Injection | 43 | 27,284 | 0.512 | 0.256 | 0.962 | 0.962 | 0.767 | 0.605 | 0.140 | 0.047 |
| SSH_Bruteforce | 1 | 23306 | 0.000 | — | — | — | — | — | — | — |

### AE-XGBoost — 12 round hop le (pool = toan bo attack train cua round, 106,090–122,048 flow)

| Khoa | AUROC macro (tat ca) | AUROC macro (bo song sinh) | Detect@5% macro |
|---|---|---|---|
| SHAP day du | 0.939 | 0.965 | 0.795 |
| SHAP 78 | 0.938 | 0.963 | 0.786 |
| Feature tho | 0.806 | 0.942 | 0.721 |

- SHAP day du vs Feature tho: chenh TB +0.133, hon o 10/12 round, Wilcoxon p = 0.016
- SHAP 78 vs Feature tho: chenh TB +0.132, hon o 11/12 round, Wilcoxon p = 0.016
- SHAP day du vs SHAP 78: chenh TB +0.001, hon o 6/12 round, Wilcoxon p = 0.939
- End-to-end zero-day (ZS-Acc x Detect@5%, SHAP day du, 14 round, round khong hop le = 0): 0.4366 (bo song sinh: 0.4862)

| Round | n unk | n known | ZS-Acc | Trung lap (unk) | AUROC SHAP day du | AUROC SHAP 78 | AUROC tho | Det@5% SHAP | Tactic (SHAP) | Tactic (tho) |
|---|---|---|---|---|---|---|---|---|---|---|
| Bot | 4 | 23308 | 0.000 | — | — | — | — | — | — | — |
| Brute_Force_Web | 384 | 27,207 | 0.692 | 0.042 | 0.997 | 0.996 | 0.947 | 1.000 | 0.000 | 0.000 |
| Brute_Force_XSS | 188 | 27,247 | 0.825 | 0.053 | 0.990 | 0.988 | 0.970 | 0.926 | 0.074 | 0.011 |
| DDOS_attack_HOIC | 924 | 23,283 | 0.046 | 0.000 | 1.000 | 1.000 | 0.999 | 1.000 | 0.000 | 0.024 |
| DDOS_attack_LOIC_UDP | 1,730 | 26,915 | 1.000 | 0.013 | 0.982 | 0.980 | 0.962 | 0.953 | 1.000 | 1.000 |
| DDoS_attacks_LOIC_HTTP | 38 | 23,292 | 0.002 | 0.368 | 0.713 | 0.691 | 0.741 | 0.500 | 0.737 | 0.868 |
| DoS_attacks_GoldenEye | 19,996 | 23,297 | 1.000 | 0.000 | 0.999 | 1.000 | 0.999 | 1.000 | 1.000 | 0.817 |
| DoS_attacks_Hulk | 19,761 | 23,291 | 0.988 | 0.000 | 0.995 | 0.997 | 0.990 | 1.000 | 0.979 | 0.807 |
| DoS_attacks_SlowHTTPTest ‡ | 55 | 27,284 | 1.000 | 0.964 | 0.804 | 0.812 | 0.152 | 0.109 | 0.000 | 0.000 |
| DoS_attacks_Slowloris | 7,625 | 25,326 | 0.770 | 0.000 | 0.997 | 0.998 | 0.997 | 1.000 | 0.991 | 0.969 |
| FTP_BruteForce ‡ | 53 | 27,282 | 1.000 | 0.981 | 0.812 | 0.817 | 0.109 | 0.170 | 0.019 | 0.019 |
| Infiltration | 282 | 26,521 | 0.014 | 0.011 | 1.000 | 0.999 | 0.982 | 1.000 | 0.000 | 0.000 |
| SQL_Injection | 58 | 27,299 | 0.690 | 0.190 | 0.979 | 0.978 | 0.828 | 0.879 | 0.121 | 0.034 |
| SSH_Bruteforce | 3 | 23311 | 0.000 | — | — | — | — | — | — | — |

### SVDD-XGBoost — 12 round hop le (pool = toan bo attack train cua round, 106,090–122,048 flow)

| Khoa | AUROC macro (tat ca) | AUROC macro (bo song sinh) | Detect@5% macro |
|---|---|---|---|
| SHAP day du | 0.924 | 0.947 | 0.752 |
| SHAP 78 | 0.925 | 0.948 | 0.736 |
| Feature tho | 0.805 | 0.940 | 0.715 |

- SHAP day du vs Feature tho: chenh TB +0.119, hon o 8/12 round, Wilcoxon p = 0.176
- SHAP 78 vs Feature tho: chenh TB +0.120, hon o 8/12 round, Wilcoxon p = 0.110
- SHAP day du vs SHAP 78: chenh TB -0.001, hon o 3/12 round, Wilcoxon p = 0.401
- End-to-end zero-day (ZS-Acc x Detect@5%, SHAP day du, 14 round, round khong hop le = 0): 0.4107 (bo song sinh: 0.4514)

| Round | n unk | n known | ZS-Acc | Trung lap (unk) | AUROC SHAP day du | AUROC SHAP 78 | AUROC tho | Det@5% SHAP | Tactic (SHAP) | Tactic (tho) |
|---|---|---|---|---|---|---|---|---|---|---|
| Bot | 0 | 23330 | 0.000 | — | — | — | — | — | — | — |
| Brute_Force_Web | 259 | 27,209 | 0.467 | 0.066 | 0.987 | 0.985 | 0.928 | 0.900 | 0.000 | 0.000 |
| Brute_Force_XSS | 217 | 27,240 | 0.952 | 0.046 | 0.977 | 0.973 | 0.972 | 0.894 | 0.055 | 0.009 |
| DDOS_attack_HOIC | 124 | 23,295 | 0.006 | 0.000 | 1.000 | 1.000 | 0.999 | 1.000 | 0.000 | 0.008 |
| DDOS_attack_LOIC_UDP | 1,730 | 26,921 | 1.000 | 0.013 | 0.945 | 0.947 | 0.962 | 0.835 | 1.000 | 1.000 |
| DDoS_attacks_LOIC_HTTP | 37 | 23,312 | 0.002 | 0.378 | 0.629 | 0.630 | 0.734 | 0.378 | 0.892 | 0.892 |
| DoS_attacks_GoldenEye | 19,954 | 23,273 | 0.998 | 0.000 | 0.999 | 1.000 | 0.999 | 1.000 | 0.992 | 0.816 |
| DoS_attacks_Hulk | 19,838 | 23,284 | 0.992 | 0.000 | 0.988 | 0.990 | 0.990 | 1.000 | 0.996 | 0.805 |
| DoS_attacks_SlowHTTPTest ‡ | 55 | 27,299 | 1.000 | 0.964 | 0.808 | 0.806 | 0.152 | 0.182 | 0.000 | 0.000 |
| DoS_attacks_Slowloris | 7,625 | 25,330 | 0.770 | 0.000 | 0.996 | 0.997 | 0.997 | 1.000 | 0.981 | 0.969 |
| FTP_BruteForce ‡ | 53 | 27,274 | 1.000 | 0.981 | 0.810 | 0.820 | 0.110 | 0.151 | 0.019 | 0.019 |
| Infiltration | 178 | 26,520 | 0.009 | 0.011 | 0.999 | 0.999 | 0.975 | 0.994 | 0.000 | 0.000 |
| SQL_Injection | 65 | 27,297 | 0.774 | 0.169 | 0.955 | 0.955 | 0.845 | 0.692 | 0.108 | 0.031 |
| SSH_Bruteforce | 1 | 23301 | 0.000 | — | — | — | — | — | — | — |


## alpha = 1.5%

### XGBoost-only — 12 round hop le (pool = toan bo attack train cua round, 106,090–122,048 flow)

| Khoa | AUROC macro (tat ca) | AUROC macro (bo song sinh) | Detect@5% macro |
|---|---|---|---|
| SHAP day du | 0.959 | 0.973 | 0.791 |
| SHAP 78 | 0.959 | 0.973 | 0.791 |
| Feature tho | 0.814 | 0.951 | 0.709 |

- SHAP day du vs Feature tho: chenh TB +0.145, hon o 10/12 round, Wilcoxon p = 0.016
- SHAP 78 vs Feature tho: chenh TB +0.145, hon o 10/12 round, Wilcoxon p = 0.016
- SHAP day du vs SHAP 78: chenh TB +0.000, hon o 0/12 round, Wilcoxon p = 1.000
- End-to-end zero-day (ZS-Acc x Detect@5%, SHAP day du, 14 round, round khong hop le = 0): 0.4791 (bo song sinh: 0.5234)

| Round | n unk | n known | ZS-Acc | Trung lap (unk) | AUROC SHAP day du | AUROC SHAP 78 | AUROC tho | Det@5% SHAP | Tactic (SHAP) | Tactic (tho) |
|---|---|---|---|---|---|---|---|---|---|---|
| Bot | 4 | 23929 | 0.000 | — | — | — | — | — | — | — |
| Brute_Force_Web | 400 | 27,860 | 0.721 | 0.043 | 0.989 | 0.989 | 0.943 | 0.940 | 0.000 | 0.000 |
| Brute_Force_XSS | 226 | 27,881 | 0.991 | 0.062 | 0.979 | 0.979 | 0.952 | 0.894 | 0.221 | 0.022 |
| DDOS_attack_HOIC | 8,299 | 23,942 | 0.415 | 0.000 | 1.000 | 1.000 | 0.998 | 1.000 | 0.216 | 0.296 |
| DDOS_attack_LOIC_UDP | 1,730 | 27,573 | 1.000 | 0.013 | 0.963 | 0.963 | 0.953 | 0.870 | 1.000 | 1.000 |
| DDoS_attacks_LOIC_HTTP | 83 | 23,930 | 0.004 | 0.169 | 0.838 | 0.838 | 0.850 | 0.663 | 0.398 | 0.398 |
| DoS_attacks_GoldenEye | 19,977 | 23,926 | 0.999 | 0.000 | 0.999 | 0.999 | 0.996 | 1.000 | 0.665 | 0.817 |
| DoS_attacks_Hulk | 19,838 | 23,940 | 0.992 | 0.000 | 0.996 | 0.996 | 0.985 | 1.000 | 0.498 | 0.805 |
| DoS_attacks_SlowHTTPTest ‡ | 55 | 27,920 | 1.000 | 0.964 | 0.883 | 0.883 | 0.150 | 0.200 | 0.000 | 0.000 |
| DoS_attacks_Slowloris | 7,678 | 25,978 | 0.775 | 0.000 | 0.991 | 0.991 | 0.994 | 1.000 | 0.940 | 0.963 |
| FTP_BruteForce ‡ | 53 | 27,927 | 1.000 | 0.981 | 0.895 | 0.895 | 0.108 | 0.226 | 0.019 | 0.019 |
| Infiltration | 657 | 26,523 | 0.033 | 0.005 | 1.000 | 1.000 | 0.984 | 0.998 | 0.000 | 0.000 |
| SQL_Injection | 76 | 27,940 | 0.905 | 0.145 | 0.973 | 0.973 | 0.860 | 0.697 | 0.105 | 0.026 |
| SSH_Bruteforce | 17 | 23931 | 0.001 | — | — | — | — | — | — | — |

### AE-XGBoost — 13 round hop le (pool = toan bo attack train cua round, 106,090–122,048 flow)

| Khoa | AUROC macro (tat ca) | AUROC macro (bo song sinh) | Detect@5% macro |
|---|---|---|---|
| SHAP day du | 0.955 | 0.984 | 0.831 |
| SHAP 78 | 0.956 | 0.983 | 0.825 |
| Feature tho | 0.834 | 0.962 | 0.755 |

- SHAP day du vs Feature tho: chenh TB +0.121, hon o 12/13 round, Wilcoxon p = 0.000
- SHAP 78 vs Feature tho: chenh TB +0.122, hon o 12/13 round, Wilcoxon p = 0.002
- SHAP day du vs SHAP 78: chenh TB -0.001, hon o 7/13 round, Wilcoxon p = 0.866
- End-to-end zero-day (ZS-Acc x Detect@5%, SHAP day du, 14 round, round khong hop le = 0): 0.5204 (bo song sinh: 0.5870)

| Round | n unk | n known | ZS-Acc | Trung lap (unk) | AUROC SHAP day du | AUROC SHAP 78 | AUROC tho | Det@5% SHAP | Tactic (SHAP) | Tactic (tho) |
|---|---|---|---|---|---|---|---|---|---|---|
| Bot | 168 | 23,848 | 0.008 | 0.000 | 0.999 | 0.999 | 0.995 | 1.000 | 0.000 | 0.000 |
| Brute_Force_Web | 405 | 27,747 | 0.730 | 0.042 | 0.995 | 0.995 | 0.945 | 0.998 | 0.000 | 0.000 |
| Brute_Force_XSS | 223 | 27,799 | 0.978 | 0.063 | 0.978 | 0.977 | 0.953 | 0.910 | 0.067 | 0.013 |
| DDOS_attack_HOIC | 16,488 | 23,853 | 0.824 | 0.000 | 1.000 | 1.000 | 0.998 | 1.000 | 0.000 | 0.239 |
| DDOS_attack_LOIC_UDP | 1,730 | 27,490 | 1.000 | 0.013 | 0.978 | 0.977 | 0.955 | 0.953 | 1.000 | 1.000 |
| DDoS_attacks_LOIC_HTTP | 119 | 23,831 | 0.006 | 0.118 | 0.905 | 0.898 | 0.904 | 0.832 | 0.235 | 0.277 |
| DoS_attacks_GoldenEye | 20,000 | 23,839 | 1.000 | 0.000 | 0.999 | 0.999 | 0.997 | 1.000 | 1.000 | 0.817 |
| DoS_attacks_Hulk | 19,872 | 23,842 | 0.994 | 0.000 | 0.991 | 0.994 | 0.986 | 1.000 | 0.973 | 0.803 |
| DoS_attacks_SlowHTTPTest ‡ | 55 | 27,848 | 1.000 | 0.964 | 0.794 | 0.804 | 0.150 | 0.091 | 0.000 | 0.000 |
| DoS_attacks_Slowloris | 7,678 | 25,886 | 0.775 | 0.000 | 0.994 | 0.996 | 0.995 | 1.000 | 0.985 | 0.963 |
| FTP_BruteForce ‡ | 53 | 27,824 | 1.000 | 0.981 | 0.802 | 0.808 | 0.108 | 0.151 | 0.019 | 0.019 |
| Infiltration | 654 | 26,523 | 0.033 | 0.005 | 1.000 | 1.000 | 0.988 | 1.000 | 0.000 | 0.000 |
| SQL_Injection | 81 | 27,854 | 0.964 | 0.136 | 0.982 | 0.980 | 0.869 | 0.864 | 0.198 | 0.062 |
| SSH_Bruteforce | 18 | 23847 | 0.001 | — | — | — | — | — | — | — |

### SVDD-XGBoost — 13 round hop le (pool = toan bo attack train cua round, 106,090–122,048 flow)

| Khoa | AUROC macro (tat ca) | AUROC macro (bo song sinh) | Detect@5% macro |
|---|---|---|---|
| SHAP day du | 0.945 | 0.971 | 0.798 |
| SHAP 78 | 0.946 | 0.971 | 0.791 |
| Feature tho | 0.833 | 0.961 | 0.752 |

- SHAP day du vs Feature tho: chenh TB +0.112, hon o 9/13 round, Wilcoxon p = 0.146
- SHAP 78 vs Feature tho: chenh TB +0.113, hon o 9/13 round, Wilcoxon p = 0.068
- SHAP day du vs SHAP 78: chenh TB -0.001, hon o 8/13 round, Wilcoxon p = 0.946
- End-to-end zero-day (ZS-Acc x Detect@5%, SHAP day du, 14 round, round khong hop le = 0): 0.5058 (bo song sinh: 0.5684)

| Round | n unk | n known | ZS-Acc | Trung lap (unk) | AUROC SHAP day du | AUROC SHAP 78 | AUROC tho | Det@5% SHAP | Tactic (SHAP) | Tactic (tho) |
|---|---|---|---|---|---|---|---|---|---|---|
| Bot | 0 | 23845 | 0.000 | — | — | — | — | — | — | — |
| Brute_Force_Web | 403 | 27,754 | 0.726 | 0.042 | 0.988 | 0.987 | 0.944 | 0.933 | 0.000 | 0.000 |
| Brute_Force_XSS | 223 | 27,787 | 0.978 | 0.063 | 0.961 | 0.958 | 0.953 | 0.852 | 0.063 | 0.013 |
| DDOS_attack_HOIC | 3,898 | 23,823 | 0.195 | 0.000 | 1.000 | 1.000 | 0.998 | 1.000 | 0.000 | 0.066 |
| DDOS_attack_LOIC_UDP | 1,730 | 27,484 | 1.000 | 0.013 | 0.942 | 0.944 | 0.955 | 0.830 | 1.000 | 1.000 |
| DDoS_attacks_LOIC_HTTP | 98 | 23,838 | 0.005 | 0.143 | 0.856 | 0.855 | 0.885 | 0.765 | 0.337 | 0.337 |
| DoS_attacks_GoldenEye | 19,993 | 23,828 | 1.000 | 0.000 | 0.998 | 0.999 | 0.997 | 1.000 | 0.990 | 0.817 |
| DoS_attacks_Hulk | 19,864 | 23,810 | 0.993 | 0.000 | 0.983 | 0.985 | 0.986 | 1.000 | 0.995 | 0.804 |
| DoS_attacks_SlowHTTPTest ‡ | 55 | 27,852 | 1.000 | 0.964 | 0.800 | 0.798 | 0.150 | 0.109 | 0.000 | 0.000 |
| DoS_attacks_Slowloris | 7,678 | 25,885 | 0.775 | 0.000 | 0.993 | 0.994 | 0.995 | 1.000 | 0.974 | 0.963 |
| FTP_BruteForce ‡ | 53 | 27,817 | 1.000 | 0.981 | 0.802 | 0.812 | 0.108 | 0.151 | 0.019 | 0.019 |
| Infiltration | 652 | 26,523 | 0.033 | 0.005 | 1.000 | 1.000 | 0.987 | 0.998 | 0.000 | 0.000 |
| SQL_Injection | 81 | 27,851 | 0.964 | 0.136 | 0.960 | 0.960 | 0.868 | 0.741 | 0.173 | 0.062 |
| SSH_Bruteforce | 15,337 | 23,864 | 0.767 | 0.000 | 1.000 | 1.000 | 0.998 | 1.000 | 0.999 | 0.146 |
