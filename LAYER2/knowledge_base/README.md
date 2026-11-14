# Knowledge base (KB) cho báo cáo VLM

KB là kho đoạn văn có nguồn mà VLM được trích dẫn khi viết báo cáo cho một alert. Một lần tra cứu
(`retrieval/store.py`) trả nhãn ℓ* (vote 5-NN trên vector SHAP) và cờ zero-day; `retrieve.py` dùng nhãn,
các feature có |φ| lớn nhất và giá trị thô của flow để chọn đoạn.

## Nguyên tắc

1. **Một khóa duy nhất:** nhãn → tactic / technique / sub-technique ATT&CK theo **Table II của Borisenko et al.
   (2022)** (IEEE SYNCHROINFO, DOI 10.1109/SYNCHROINFO55067.2022.9840920), chép nguyên văn vào
   `mapping/borisenko2022_table2.yaml`. Mọi tri thức tấn công sau đó lấy từ MITRE theo technique ID
   (ATT&CK v20.1, CAPEC v3.9); không có gì lấy từ dữ liệu CSE-CIC-IDS2018.
2. **Một loại bằng chứng:** trích dẫn `{src, locator, quote}`; `build/validate.py` kiểm `quote` nằm nguyên văn
   trong file nguồn đã khóa sha256. Không có quyết định của tác giả, không có số liệu tự tính từ dataset.
3. **Định nghĩa feature theo đúng CICFlowMeter-V3 mà dataset dùng** (mã nguồn @50d668db); khi ReadMe và mã
   nguồn khác nhau thì theo mã nguồn. Feature của AE chỉ có định nghĩa chung (tầng latent; MSE tái tạo).
4. **Mọi câu trong KB là tiếng Anh.**

Ba loại đoạn (`kind`):

| kind | text là gì | code kiểm gì |
|---|---|---|
| `verbatim` | chuỗi con nguyên văn của nguồn (mô tả ATT&CK/CAPEC, câu RFC, câu ReadMe) | text nằm nguyên văn trong nguồn |
| `rendered` | code ghép từ các trường nguyên văn (dòng Table II, dòng IANA, analytic ATT&CK, ReadMe) | mỗi support nguyên văn |
| `defined` | định nghĩa viết trong `definitions/` (cột có ghi chú `replaces`, feature AE, artefact H3) | mỗi support nguyên văn; không có `%` |

## Cấu trúc

```
knowledge_base/
├── mapping/
│   └── borisenko2022_table2.yaml   14 nhãn: dòng Table II (trích nguyên văn từ PDF) + entries {tactic, technique, sub_technique}
├── definitions/
│   ├── feature_names.yaml    78 cột CSV ↔ mã nguồn CICFlowMeter 2018 ↔ ReadMe (SINH BẰNG MÁY: build/draft_features.py)
│   ├── feature_notes.yaml    ghi chú cách CICFlowMeter 2018 tính cột (replaces | adds), trích mã nguồn + audit
│   ├── model_features.yaml   Dst Port, Protocol, ae_recon_error, ae_latent (không có trong ReadMe)
│   └── artefacts.yaml        H3: điều kiện `when` + câu cơ chế
├── build/                    CODE: NGUỒN → ĐOẠN VĂN (chạy offline)
│   ├── sources.yaml          20 nguồn: id, tầng, phiên bản, url, giấy phép (borisenko-2022: PDF đặt tay)
│   ├── fetch_sources.py      tải nguồn → $UTS_DATA/kb_sources/raw/<id>/, ghi sha256 vào store/sources.lock.json
│   ├── common.py, mitre.py   đường dẫn, chuẩn hóa văn bản, đọc STIX ATT&CK / CAPEC XML / PDF
│   ├── flag_order.py         cột cờ nào thực chứa cờ nào (thứ tự HashMap của OpenJDK 8)
│   ├── part_technique.py     M + B
│   ├── part_feature.py       H1
│   ├── part_protocol.py      H2
│   ├── part_artefact.py      H3
│   ├── part_mitigation.py    G
│   ├── build_kb.py           chạy mọi phần → store/
│   └── validate.py           kiểm khóa sha256, trích dẫn, khóa Borisenko, độ phủ
├── store/                    KẾT QUẢ (commit) — Kaggle chỉ cần thư mục này
│   ├── passages.jsonl        6.223 đoạn
│   ├── index.json            nhãn → techniques, CAPEC, có analytic mạng không; cột cờ → cờ thực chứa; cột rời rạc
│   ├── sources.lock.json     sha256 từng file nguồn
│   └── build_report.md       báo cáo duyệt: bảng Borisenko, gaps, mọi câu `defined` đặt cạnh bằng chứng
├── conditions.py             đánh giá `when` của H3 (không dùng eval)
├── retrieve.py               LÚC CHẠY: chọn đoạn cho một alert
└── tests/test_retrieve.py
```

## Các phần

| Phần | Giúp mục nào của báo cáo | Khóa tra | Nguồn | Số đoạn |
|---|---|---|---|---|
| M mapping | 1, 4: nhãn tiền lệ ứng với technique nào | nhãn | Borisenko et al. 2022, Table II (+ tên ID từ ATT&CK) | 14 |
| B technique | 4: cơ chế tấn công | technique ID | ATT&CK v20.1: mô tả (`B-<T>`), analytic dùng data component mạng (`B-DET-<T>`); CAPEC v3.9 mà CAPEC liên kết tới technique (`B-CAPEC-n`) | 8 + 4 + 8 |
| H1 feature | 3: feature trong waterfall là gì | tên feature | ReadMe + mã nguồn CICFlowMeter 2018, Engelen et al. 2021, DistriNet (Liu et al. 2022); mã Layer 1 | 81 |
| H2 giao thức/cổng/cờ | 2: hành vi quan sát | Protocol, Dst Port, cột cờ | IANA (mọi cổng có đăng ký tcp/udp, không chọn lọc), RFC 6335 §9, RFC 9293, RFC 3168 | 6.077 |
| H3 artefact | 5: giới hạn của bằng chứng | điều kiện trên giá trị thô | mã nguồn CICFlowMeter, RFC 9293/7323, JLS, IANA | 6 |
| G mitigation | 6 (code điền, không đưa VLM) | technique ID | quan hệ `mitigates` của ATT&CK | 25 |

Gaps (build_report §3): T1110.001, T1204.002, T1583.005, T1113 không có analytic ATT&CK nào dùng data component
mạng, nên các nhãn đó chỉ có `B-<T>` (+ CAPEC nếu có).

Cột cờ: chỉ phản ánh **gói đầu tiên** của flow và bị **hoán vị** so với tên cột (ví dụ `SYN Flag Cnt` chứa PSH,
`PSH Flag Cnt` chứa SYN). H1 của mỗi cột cờ nói cờ thực chứa; H2 chọn định nghĩa RFC của cờ thực chứa.

## Build lại và kiểm

```bash
export UTS_DATA=<workspace>/LAYER2
python LAYER2/knowledge_base/build/fetch_sources.py            # tải nguồn (--verify: chỉ so sha256)
python LAYER2/knowledge_base/build/build_kb.py                 # sinh store/ + build_report.md + validate (~5 s)
python LAYER2/knowledge_base/tests/test_retrieve.py
```

PDF Borisenko 2022 không tải tự động: đặt tay vào `$UTS_DATA/kb_sources/raw/borisenko-2022/borisenko2022.pdf`.

## Dùng lúc chạy

```python
from knowledge_base.retrieve import KnowledgeBase
sel = KnowledgeBase().select(label, flagged, top_features, x_raw, contrast_features=..., exclude_labels=...)
sel.passages      # đoạn đưa VLM, theo thứ tự; mỗi đoạn có cite (mã để VLM trích) và role
                  #   (attack_type_background | comparison_only | definition | caveat)
sel.techniques    # technique của nhãn theo Borisenko (rỗng khi bật cờ)
sel.mitigations   # G cho code điền mục 6
sel.missing       # [{pid, reason}] đoạn lẽ ra được chọn nhưng KB không có
```

- Không cờ: M-<nhãn>, rồi với mỗi technique: `B-<T>`, `B-DET-<T>` (nếu có), các `B-CAPEC-n`; G cho mọi technique.
- Bật cờ zero-day: bỏ M, B giữ nội dung nhưng ẩn tiêu đề và `cite` = `REF-MECH-n`; không có G.
- Protocol A: nhãn trong `exclude_labels` không bao giờ được chọn.
- Có đoạn cổng thì kèm `H2-PORT-NOTE` (RFC 6335: cổng đăng ký không cho biết dịch vụ thực dùng).
- `contrast_features` nhận `row["feature_key"]` của bảng C (`evidence/contrast.py`); dòng latent gộp có khóa `ae_latent`.
- Giá trị thô không hữu hạn (NaN, Inf, chuỗi) không gây lỗi: H2 của trường đó bị bỏ, điều kiện `when` coi là không đúng.
- Thiếu `H2-PORT-*` (cổng không đăng ký IANA) / `H2-PROTO-*` (giao thức 0) chỉ ghi vào `sel.missing`.
  Thiếu M/B/H1/H3 thì có `warnings.warn`; `KnowledgeBase(strict=True)` thì raise `KeyError`.

Chưa nối vào `evidence/build.py` và prompt (bước tiếp theo).
