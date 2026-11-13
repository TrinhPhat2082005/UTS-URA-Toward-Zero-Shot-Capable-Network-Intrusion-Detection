# Knowledge base (KB) cho báo cáo VLM

KB là kho đoạn văn có nguồn mà VLM được trích dẫn khi viết báo cáo cho một alert. Một lần tra cứu
(`retrieval/store.py`) trả nhãn ℓ* (vote 5-NN trên vector SHAP) và cờ "không có tiền lệ gần";
`retrieve.py` dùng nhãn, các feature có |φ| lớn nhất và giá trị thô của flow để chọn đoạn.

## Ba nguyên tắc

1. **Câu nào tác giả tự viết đều nằm trong `authored/`.** Người duyệt chỉ cần đọc các file đó.
2. **Câu nào trích nguyên văn đều được code kiểm lại với file nguồn đã khóa sha256** (`build/validate.py`).
3. **Con số nào về dataset cũng do code tính**, và đoạn gửi VLM chỉ được dùng **claim dừng** (`build/claims.py`):
   claim phải đúng trên train, val, từng file CSV gốc đã công bố và hai nửa của mỗi nhãn trong từng file.
   Số không dừng (ví dụ tỉ lệ cổng theo nhãn) bị bỏ hoặc đổi thành phát biểu bất biến. Text gửi VLM và
   rationale không có `%` (trừ ngưỡng "99.9%"). `build/data_facts.py` chỉ còn số mô tả trên train cho báo cáo.
4. **Mọi câu trong KB là tiếng Anh** và là trích dẫn nguyên văn hoặc trích dẫn diễn giải có nguồn.

Mỗi giá trị trong yaml thuộc đúng một loại:

| Loại | Dạng | Code kiểm gì |
|---|---|---|
| Trích dẫn | `{src, locator, quote}` | `quote` nằm nguyên văn trong nguồn `src` |
| Số liệu | `{data: <khóa>}` | khóa có trong `store/data_facts.json` |
| Quyết định | `{decision, support, rationale}` | có bằng chứng và lý do; nội dung do người duyệt |

Quyết định chỉ có trong `authored/mapping.yaml`: nhãn → technique ATT&CK và CAPEC.

## Cấu trúc

```
knowledge_base/
├── authored/                 TÁC GIẢ QUYẾT (có bằng chứng) — người duyệt đọc các file này
│   ├── mapping.yaml          14 nhãn → câu hành vi, technique, CAPEC (+ lý do, so với profile cũ)
│   ├── feature_names.yaml    78 cột CSV ↔ mã nguồn CICFlowMeter 2018 ↔ ReadMe (SINH BẰNG MÁY)
│   ├── extractor_notes.yaml  ghi chú cách CICFlowMeter 2018 tính từng cột (trích mã nguồn)
│   ├── extra_features.yaml   Dst Port, Protocol, feature của AE (không có trong ReadMe)
│   └── artefacts.yaml        H3: điều kiện `when` + câu cơ chế
├── build/                    CODE: NGUỒN → ĐOẠN VĂN (chạy offline)
│   ├── sources.yaml          16 nguồn: id, tầng, phiên bản, url, giấy phép
│   ├── fetch_sources.py      tải nguồn → $UTS_DATA/kb_sources/, ghi sha256 (file trong repo: sha của nội dung LF + commit)
│   ├── data_facts.py         số liệu mô tả trên train → store/data_facts.json (chỉ cho báo cáo + danh sách cột trùng)
│   ├── claims.py             claim số liệu có kiểm tính dừng → store/claims.json (--raw: đọc 10 file CSV gốc)
│   ├── draft_features.py     sinh authored/feature_names.yaml
│   ├── common.py, mitre.py   đường dẫn, chuẩn hóa văn bản, đọc STIX/CAPEC (thư viện chuẩn)
│   ├── part_label.py         A'  thẻ nhãn
│   ├── part_mechanism.py     B'  technique ATT&CK + CAPEC (nguyên văn)
│   ├── part_feature.py       H1  định nghĩa feature + ghi chú
│   ├── part_protocol.py      H2  giao thức, cổng, cờ TCP
│   ├── part_artefact.py      H3  artefact có điều kiện
│   ├── part_mitigation.py    G'  mitigation (code điền báo cáo, không đưa VLM)
│   ├── build_kb.py           chạy mọi phần → store/
│   └── validate.py           kiểm trích dẫn, số liệu, quyết định, độ phủ
├── store/                    KẾT QUẢ (commit) — Kaggle chỉ cần thư mục này
│   ├── passages.jsonl        155 đoạn
│   ├── claims.json           45 claim, giá trị trên từng nguồn, holds_all
│   ├── index.json            nhãn → technique/CAPEC; cột cờ → cờ thực chứa; cột rời rạc
│   ├── sources.lock.json     sha256 từng file nguồn
│   ├── data_facts.json       số liệu dataset (+ sha256 của train_full.csv)
│   └── build_report.md       báo cáo duyệt: mọi câu tác giả viết đặt cạnh bằng chứng
├── conditions.py             đánh giá `when` của H3 (không dùng eval)
├── retrieve.py               LÚC CHẠY: chọn đoạn cho một alert
├── tests/test_retrieve.py
└── attack_profiles/, profiles.py   profile cũ, giữ để tái lập Setup 1/2 đã chạy
```

## Các phần

| Phần | Giúp mục nào của báo cáo | Khóa tra | Nguồn | Số đoạn |
|---|---|---|---|---|
| A' thẻ nhãn | 1, 4: ℓ* là hành vi gì | nhãn | trang UNB + số liệu; technique theo `mapping.yaml` | 14 |
| B' cơ chế | 4: đối chiếu với technique/CAPEC | nhãn → technique, CAPEC | ATT&CK v20.1, CAPEC v3.9 (nguyên văn) | 14 |
| H1 feature | 3: feature trong waterfall là gì | tên feature | ReadMe + mã nguồn CICFlowMeter 2018; mã Layer 1 | 81 |
| H2 giao thức/cổng/cờ | 2: hành vi quan sát | Protocol, Dst Port, cột cờ | IANA, RFC 9293, RFC 3168 | 21 |
| H3 artefact | 5: giới hạn của bằng chứng | điều kiện trên giá trị thô | mã nguồn CICFlowMeter, RFC 9293/7323, JLS + claim | 6 |
| G' mitigation | 6 (code điền) | technique | quan hệ `mitigates` của ATT&CK | 19 |

Phát hiện quan trọng từ mã nguồn CICFlowMeter 2018, đã đưa vào H1/H2: các cột cờ chỉ phản ánh **gói đầu tiên**
của flow và bị **hoán vị** so với tên cột (ví dụ cột `SYN Flag Cnt` chứa cờ PSH). Xem `store/build_report.md` mục 6.

## Build lại và kiểm

```bash
export UTS_DATA=<workspace>/LAYER2
python LAYER2/knowledge_base/build/fetch_sources.py            # tải nguồn (--verify: chỉ so sha256)
python LAYER2/knowledge_base/build/data_facts.py
export UTS_BENCH=<workspace>/zeroshot_benchmark                # chỉ cần cho --raw
python LAYER2/knowledge_base/build/claims.py --raw             # claim trên train, val + 10 file gốc (~15 phút);
                                                               # không --raw: chỉ tính lại train/val, giữ giá trị file gốc
python LAYER2/knowledge_base/build/build_kb.py                 # sinh store/ + build_report.md + validate
python LAYER2/knowledge_base/tests/test_retrieve.py
```

## Dùng lúc chạy

```python
from knowledge_base.retrieve import KnowledgeBase
sel = KnowledgeBase().select(label, flagged, top_features, x_raw, contrast_features=..., exclude_labels=...)
sel.passages      # đoạn đưa VLM, theo thứ tự; mỗi đoạn có cite (mã để VLM trích) và role
                  #   (attack_type_background | comparison_only | definition | caveat)
sel.mitigations   # G' cho code điền mục 6
sel.missing       # [{pid, reason}] đoạn lẽ ra được chọn nhưng KB không có
```

- `contrast_features` nhận `row["feature_key"]` của bảng C (`evidence/contrast.py`); dòng latent gộp có khóa `ae_latent`.
- Giá trị thô không hữu hạn (NaN, Inf, chuỗi) không gây lỗi: H2 của trường đó bị bỏ, điều kiện `when` coi là không đúng.
- Thiếu `H2-PORT-*` / `H2-PROTO-*` là bình thường (KB chỉ có cổng chính) nên chỉ ghi vào `sel.missing`.
  Thiếu A′/B′/H1/H3 thì có `warnings.warn`; `KnowledgeBase(strict=True)` thì raise `KeyError` (dùng cho test và experiment).

Chưa nối vào `evidence/build.py` và prompt: việc đó thuộc đợt viết lại prompt (D6).

## Thay đổi

Mỗi thay đổi ghi mã mục trong `workspace/kb_checklist.md` và loại sửa (build = phải chạy lại build; runtime = chỉ đổi lúc chạy).

| Ngày | Mục | Loại | Thay đổi |
|---|---|---|---|
| 2026-11-13 | 0.1 | runtime | Bỏ `Selection.required`: đánh giá chỉ dùng faithfulness + groundedness |
| 2026-11-13 | 2.1 | runtime | `_finite_int` cho Protocol/Dst Port; `conditions._term` coi NaN/Inf/chuỗi là không đúng |
| 2026-11-13 | 2.2 | runtime | `Selection.missing`, cảnh báo khi thiếu A′/B′/H1/H3, tham số `strict` |
| 2026-11-13 | 2.3 | runtime | Bảng C có `feature_key`; `_h1_pid` nhận khóa `ae_latent` của dòng latent gộp |
| 2026-11-13 | D1, 4.10 | build | `build/claims.py` + `store/claims.json`: 45 claim, đều đúng trên train, val, 10 file gốc và hai nửa mỗi nhãn. validate: đoạn gửi VLM chỉ dùng claim dừng (ngoại lệ `flag_column_holds`). Câu "in this dataset" → "in the published CSE-CIC-IDS2018 CSV files" |
| 2026-11-13 | 4.5 | build | Nguồn `layer1-code`: sha256 của nội dung LF + commit; `fetch_sources` từ chối khóa file có thay đổi chưa commit |
| 2026-11-13 | 1.1 | build | Ghi chú có `relation: replaces | adds`; 22 cột `replaces` (12 cột cờ, 6 cột bulk, 2 Init Win, Fwd Act Data Pkts, Fwd Seg Size Min) mở đầu bằng nghĩa thật, định nghĩa ReadMe bỏ khỏi text gửi VLM (danh sách: build_report §6b) |
| 2026-11-13 | 1.2a–d | build | Thẻ BFW, Infiltration, Bot, FTP, SlowHTTPTest viết lại: tách kịch bản, dữ liệu (claim) và audit DistriNet; bỏ "attachment"; không có `%` |
| 2026-11-13 | 1.3 | build | Thẻ A′ mở đầu "Reference label X (CSE-CIC-IDS2018)" |
| 2026-11-13 | 1.4 | runtime | Role A′/B′: `attack_type_background` (không cờ) / `comparison_only` (cờ) |
| 2026-11-13 | 1.5a | build | Nguồn RFC 7323; H3-NOWIN trích đủ câu RFC 9293 về window scaling; đoạn mới H3-WINSCALE (TCP, gói đầu không phải SYN) |
| 2026-11-13 | 1.5b, 1.5c | build | Ghi chú XN-FIRST-PACKET-TWICE, XN-DOWNUP-INTEGER (`adds`) + claim bất biến |
| 2026-11-13 | 1.5d | build | Nguồn RFC 6335; mỗi H2-PORT có ghi chú "đăng ký cổng không cho biết dịch vụ thực dùng"; bỏ khóa `attack_main_ports` khỏi support |
| 2026-11-13 | 1.5e | quyết định | Không ghi số Init Win theo nhãn (không dừng) |
| 2026-11-13 | 1.6 | build + runtime | Câu technique của A′ ở trường `technique_text`, chỉ nối khi không cờ; khi cờ B′ ẩn tiêu đề, `cite` = REF-MECH-n (U11 = A) |
| 2026-11-13 | 4.6 | build | validate cấm `%` trong text gửi VLM và trong text/rationale của yaml (trừ "99.9%"); rationale Bot/Infiltration/GoldenEye/Hulk/LOIC-UDP trỏ claim |
| 2026-11-13 | 6.3 | build | SSH, FTP → CAPEC-16 qua route mới `child_of_linked` (CAPEC-16 ChildOf 49, 49 → T1110.001); BFW giữ CAPEC-49 |
| 2026-11-13 | nguồn | build | Thêm nguồn `distrinet-2018` (tầng audit, trang đi kèm Liu et al. 2022) |
