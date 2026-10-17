# ClassTwin: Classroom Digital Twin (SCIC 2026)

Hệ thống Bản sao số Lớp học Tối ưu hóa Chính sách Giảng dạy bằng **Offline Reinforcement Learning** và Hỗ trợ Ra Quyết định Sư phạm.

---

## 1. Cấu trúc Mô-đun Hệ thống (`src/classtwin`)

```
classtwin/
├── configs/                          # Cấu hình YAML (preprocess, estimator, policy_sim)
│   ├── preprocess.yaml
│   ├── estimator.yaml
│   └── policy_sim.yaml
├── docs/                             # Tài liệu đề án LaTeX & PDF
│   ├── ClassTwin_Proposal_SCIC2026.tex
│   └── ClassTwin_Proposal_SCIC2026.pdf
├── scripts/                          # Giao diện CLI dòng lệnh
│   ├── preprocess.py                 # Tiền xử lý ZIP -> Parquet theo luồng streaming
│   ├── train_estimator.py            # Huấn luyện mô hình GRU-DKT
│   ├── train_policy.py               # Huấn luyện chính sách Discrete CQL trên simulator
│   └── run_api.py                    # Khởi chạy dịch vụ FastAPI
├── src/classtwin/
│   ├── data/                         # Tiền xử lý dữ liệu & Data Contracts
│   │   ├── schemas.py                # Schema Pydantic canonical: events, attempts, decisions
│   │   ├── zip_reader.py             # Đọc streaming ZIP tránh tràn bộ nhớ
│   │   ├── adapters.py               # Adapter định dạng EdNet KT1/KT3 & Metadata
│   │   ├── preprocessing.py          # Quản lý split hash (Protocol A) & ghi Parquet shard
│   │   ├── features.py               # Trích xuất đặc trưng as-of không rò rỉ tương lai
│   │   └── sequences.py              # Sequence Dataset & DataLoader cho PyTorch
│   ├── state_estimation/             # Ước lượng trạng thái ẩn người học
│   │   ├── bkt.py                    # Mô hình Bayesian Knowledge Tracing baseline
│   │   ├── gru_dkt.py                # Mạng GRU Deep Knowledge Tracing PyTorch
│   │   ├── probe_mastery.py          # Đo lường mastery chuẩn tắc theo tập probe (Thuật toán 1)
│   │   └── calibration.py            # Đo lường ECE, Brier score và Temperature Scaling
│   ├── twin/                         # Bản sao số & Môi trường mô phỏng (Level M2)
│   │   ├── learner_dynamics.py       # Động lực học chuyển dịch trạng thái cá nhân
│   │   ├── classroom_env.py          # Môi trường lớp học ảo (40-100 sinh viên)
│   │   └── scenarios.py              # Mô phỏng phản thực Monte-Carlo (Thuật toán 2)
│   ├── policies/                     # Chính sách can thiệp sư phạm
│   │   ├── action_space.py           # Không gian 5 hành động, thời lượng & mặt nạ ràng buộc
│   │   ├── rules.py                  # Chính sách heuristic / rule-based baseline
│   │   └── cql.py                    # Discrete Conservative Q-Learning (Thuật toán 3)
│   ├── inference/                    # Vòng lặp suy luận trực tuyến (Level M3)
│   │   ├── state_store.py            # Bộ nhớ đệm snapshot trạng thái thời gian thực
│   │   ├── engine.py                 # Decision Support Engine & Safe Fallback (Thuật toán 4)
│   │   └── api.py                    # Dịch vụ FastAPI REST endpoints chuẩn
│   └── utils/
│       ├── config.py                 # Pydantic Configuration Loader
│       └── logging.py                # Structured Logger & Reproducibility RNG Seeds
└── tests/                            # Bộ kiểm thử tự động toàn diện
    ├── test_schemas.py
    ├── test_state_estimation.py
    ├── test_twin.py
    ├── test_cql.py
    └── test_inference.py
```

---

## 2. Hướng dẫn Chạy Thử nghiệm

### 2.1 Chạy Bộ Kiểm thử Toàn diện (Unit Tests)
```bash
python -c "
import sys, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path('src').resolve()))
for tf in ['tests/test_schemas.py', 'tests/test_state_estimation.py', 'tests/test_twin.py', 'tests/test_cql.py', 'tests/test_inference.py']:
    spec = importlib.util.spec_from_file_location(Path(tf).stem, tf)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for a in dir(mod):
        if a.startswith('test_'): getattr(mod, a)()
print('All tests passed!')
"
```

### 2.2 Tiền Xử lý Dữ liệu EdNet theo Luồng
```bash
python scripts/preprocess.py --config configs/preprocess.yaml --max-members 50
```

### 2.3 Huấn luyện Mô hình Ước lượng Trạng thái (GRU-DKT)
```bash
python scripts/train_estimator.py --config configs/estimator.yaml --epochs 5
```

### 2.4 Huấn luyện Chính sách Sư phạm Offline (Discrete CQL)
```bash
python scripts/train_policy.py --config configs/policy_sim.yaml --episodes 10 --train-steps 30
```

### 2.5 Khởi chạy Dịch vụ API Hỗ trợ Quyết định
```bash
python scripts/run_api.py
```
Tài liệu tương tác Swagger UI có sẵn tại: `http://localhost:8000/docs`
