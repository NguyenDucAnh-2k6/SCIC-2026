# ClassTwin: Classroom Digital Twin (SCIC 2026)

<p align="center">
  <img src="https://img.shields.io/badge/Project-SCIC%202026-007A87?style=for-the-badge" alt="SCIC 2026"/>
  <img src="https://img.shields.io/badge/Domain-AI%20in%20Education%20(AIED)-1A365D?style=for-the-badge" alt="AIED"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License MIT"/>
</p>

<p align="center">
  <!-- Backend & Core ML -->
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python 3.10+"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.12.0%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-2.3.5-013243?style=flat&logo=numpy&logoColor=white" alt="NumPy"/></a>
  <a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/Pandas-2.3.3-150458?style=flat&logo=pandas&logoColor=white" alt="Pandas"/></a>
  <a href="https://arrow.apache.org/"><img src="https://img.shields.io/badge/PyArrow-23.0.1-D22128?style=flat&logo=apachearrow&logoColor=white" alt="PyArrow"/></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.142.2-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI"/></a>
  <a href="https://docs.pydantic.dev/"><img src="https://img.shields.io/badge/Pydantic-v2.13.3-E92063?style=flat&logo=pydantic&logoColor=white" alt="Pydantic"/></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/Scikit--Learn-1.6.1-F7931E?style=flat&logo=scikitlearn&logoColor=white" alt="Scikit-Learn"/></a>
  <a href="https://scipy.org/"><img src="https://img.shields.io/badge/SciPy-1.16.3-8CAAE6?style=flat&logo=scipy&logoColor=white" alt="SciPy"/></a>
  <a href="https://www.kaggle.com/"><img src="https://img.shields.io/badge/Kaggle-GPU%20Ready-20BEFF?style=flat&logo=kaggle&logoColor=white" alt="Kaggle"/></a>
  <!-- Frontend UI & Web Dashboard -->
  <a href="https://react.dev/"><img src="https://img.shields.io/badge/React-19.x-61DAFB?style=flat&logo=react&logoColor=black" alt="React 19"/></a>
  <a href="https://vite.dev/"><img src="https://img.shields.io/badge/Vite-8.x-646CFF?style=flat&logo=vite&logoColor=white" alt="Vite"/></a>
  <a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript"><img src="https://img.shields.io/badge/JavaScript-ES2023-F7DF1E?style=flat&logo=javascript&logoColor=black" alt="JavaScript ES2023"/></a>
  <a href="https://nodejs.org/"><img src="https://img.shields.io/badge/Node.js-18%2B-339933?style=flat&logo=nodedotjs&logoColor=white" alt="Node.js"/></a>
</p>

---

## 📖 Giới thiệu Dự án

**ClassTwin (Classroom Digital Twin)** là đề án nghiên cứu khoa học tham dự cuộc thi **SCIC 2026**, giải quyết 3 điểm nghẽn lớn trong giảng đường đại học quy mô đông (60–150 sinh viên):
1. **Sự phân hóa năng lực người học:** Nhóm tiếp thu nhanh cảm thấy nhàm chán, nhóm yếu bị hổng kiến thức tiên quyết.
2. **Điểm mù trạng thái ẩn:** Năng lượng nhận thức, mức độ hiểu sâu và sự mệt mỏi (*cognitive fatigue*) không thể đo bằng mắt thường.
3. **Rủi ro thử-và-sai:** Tuyệt đối không thử nghiệm AI trực tiếp trên sinh viên thật gây gián đoạn học tập.

### Giải pháp Cốt lõi:
* **Deep Knowledge Tracing (GRU-DKT):** Ước lượng mức độ nắm vững kiến thức chuẩn tắc (*Probe Mastery*) và độ bất định mẫu mà không rò rỉ dữ liệu tương lai.
* **Classroom Digital Twin Simulator:** Giả lập động lực học lớp học ảo (40–100 người) cho phép chạy hàng nghìn kịch bản phản thực (*counterfactual rollouts*).
* **Discrete Conservative Q-Learning (CQL):** Tối ưu hóa chính sách sư phạm với chiết khấu Bán Markov $\gamma(\Delta t) = \gamma_{base}^{\Delta t / \tau_0}$ cùng ràng buộc trần khối lượng công việc của giảng viên (*Workload Budget*) và bảo vệ nhóm sinh viên yếu (*Fairness Constraint*).
* **Privacy-First (Tuyệt đối phi-camera):** Không nhận diện khuôn mặt, không ghi âm định danh. Toàn bộ dữ liệu dựa trên hành vi bài tập số và cảm biến vi khí hậu môi trường ($\text{CO}_2$, nhiệt độ, độ ồn).

---

## 💾 Dữ liệu Thực nghiệm & Tải về (Google Drive)

Toàn bộ dữ liệu thực địa đa nguồn của dự án (bao gồm `EdNet`, `ASSISTments`, `SocioPatterns`, `In-Gauge / En-Gage`, `OULAD`, `Bridge to Algebra`) được lưu trữ tập trung tại liên kết Google Drive chính thức:

<p align="center">
  <a href="https://drive.google.com/drive/folders/1kUhHPoQuXi-qhQR8K59SuRW_6PnX_teB?usp=sharing">
    <img src="https://img.shields.io/badge/Google_Drive-Tải_Dữ_Liệu_ClassTwin_Tại_Đây-34A853?style=for-the-badge&logo=googledrive&logoColor=white" alt="Google Drive Datasets"/>
  </a>
</p>

🔗 **Đường dẫn truy cập:** [Google Drive - Thư mục Dữ liệu ClassTwin](https://drive.google.com/drive/folders/1kUhHPoQuXi-qhQR8K59SuRW_6PnX_teB?usp=sharing)

### ⚠️ Hướng dẫn Sắp xếp & Giải nén Dữ liệu Bắt buộc:
Sau khi tải các tệp từ Drive về, hãy đặt toàn bộ vào thư mục `data/` trong thư mục gốc của dự án theo cấu trúc sau:

```
data/
├── EdNet-KT1.zip                <-- GIỮ NGUYÊN FILE .ZIP (TUYỆT ĐỐI KHÔNG GIẢI NÉN)
├── EdNet-KT2.zip                <-- GIỮ NGUYÊN FILE .ZIP (TUYỆT ĐỐI KHÔNG GIẢI NÉN)
├── EdNet-KT3.zip                <-- GIỮ NGUYÊN FILE .ZIP (TUYỆT ĐỐI KHÔNG GIẢI NÉN)
├── EdNet-KT4.zip                <-- GIỮ NGUYÊN FILE .ZIP (TUYỆT ĐỐI KHÔNG GIẢI NÉN)
├── assistments/                 <-- GIẢI NÉN TOÀN BỘ FILE ZIP BÊN TRONG RA THƯ MỤC NÀY
├── assistments_affect/          <-- GIẢI NÉN TOÀN BỘ FILE ZIP BÊN TRONG RA THƯ MỤC NÀY
├── in_gauge_engage/             <-- GIẢI NÉN FILE in_gauge_engage.zip RA THƯ MỤC NÀY
├── sociopatterns/               <-- GIẢI NÉN TOÀN BỘ FILE .CSV.GZ / ZIP BÊN TRONG
├── bridge_to_algebra_2006_2007/ <-- GIẢI NÉN TOÀN BỘ FILE ZIP BÊN TRONG RA THƯ MỤC NÀY
└── oulad/                       <-- GIẢI NÉN TOÀN BỘ FILE ZIP BÊN TRONG RA THƯ MỤC NÀY
```

> [!IMPORTANT]
> **Quy tắc giải nén quan trọng cần lưu ý:**
> 1. **GIỮ NGUYÊN ĐỊNH DẠNG `.zip` ĐỐI VỚI CÁC TỆP `EdNet-*.zip`** (`EdNet-KT1.zip`, `EdNet-KT2.zip`, ...): Module tiền xử lý `StreamingZipReader` của ClassTwin được thiết kế để đọc và trích xuất dữ liệu trực tiếp từ file zip qua luồng stream trong RAM, **không cần giải nén ra đĩa**, giúp tiết kiệm hơn 15 GB dung lượng ổ cứng.
> 2. **GIẢI NÉN TẤT CẢ CÁC TỆP NÉN CÒN LẠI**: Các tệp nén của `in_gauge_engage.zip`, `assistments`, `sociopatterns`, `oulad`, v.v. cần được giải nén ra các tệp `.csv` / `.tsv` trong thư mục con tương ứng để [MultiSourceDataManager](file:///d:/SCIC%202026/SCIC-2026/src/classtwin/data/multisource.py) có thể nạp trực tiếp vào môi trường Digital Twin.

---

## ⚙️ Cấu hình Hợp nhất & Tham số Điều khiển ([`configs/pipeline.yaml`](file:///d:/SCIC%202026/SCIC-2026/configs/pipeline.yaml))

Toàn bộ tham số của cả 3 giai đoạn (Tiền xử lý $\to$ Estimator GRU-DKT $\to$ Policy Discrete-CQL & Digital Twin Đa nguồn) được tập trung tại file **[`configs/pipeline.yaml`](file:///d:/SCIC%202026/SCIC-2026/configs/pipeline.yaml)**:

### 1. Tham số Dữ liệu & Tiền xử lý (`data:`)
* `raw_zip_path`: Đường dẫn tới file zip thô (`data/EdNet-KT1.zip`).
* `shards_dir`: Thư mục lưu trữ các file Parquet phân mảnh (`data/shards/train/`, `val/`, `test/`).
* `chunk_size_rows`: Kích thước mỗi shard Parquet (mặc định: `50,000` dòng/shard).
* `train_ratio`, `val_ratio`, `test_ratio`: Tỷ lệ chia tập (`0.8 / 0.1 / 0.1` theo hash `student_id`).

### 2. Tham số Mô hình Ước lượng Trạng thái (`estimator:`)
* `model_type`: Loại mô hình Knowledge Tracing (`gru_dkt`).
* `d_model`: Kích thước vector nhúng câu hỏi (mặc định: `64`).
* `d_hidden`: Kích thước vector ẩn của mạng GRU (mặc định: `128`).
* `batch_size`: Số lượng chuỗi trong mini-batch (mặc định: `128`).
* `max_epochs`: Số epoch huấn luyện tối đa (mặc định: `5`).
* `probe_sample_size`: Số câu hỏi kiểm định cố định theo kỹ năng để đo Probe Mastery.

### 3. Tham số Chính sách Offline RL (`policy:`)
* `policy_type`: Thuật toán Offline RL (`cql` - Discrete Conservative Q-Learning).
* `d_state`: Chiều vector trạng thái lớp học $\mathbf{x}_t$ (**12 chiều chuẩn hóa**).
* `cql_alpha`: Hệ số phạt bảo thủ CQL dìm giá trị Q ngoài phân phối (mặc định: `1.0`).
* `gamma_base`: Hệ số chiết khấu cơ sở (`0.99`).
* `tau_seconds`: Hằng số thời gian chuẩn hóa bước chuyển Bán Markov (`60.0` giây).
* `workload_budget`: Trần công sức tối đa của giảng viên trong tiết học (`15.0`).

### 4. Tham số Môi trường Bản sao số Lớp học Đa nguồn (`twin:`)
* `use_real_data`: Kích hoạt nạp dữ liệu thực địa đa nguồn (`true` / `false`).
* `class_id`: ID lớp học thực tế trích xuất từ ASSISTments (mặc định: `12262` gồm 53 học sinh).
* `default_cohort_size`: Sĩ số lớp học mô phỏng (mặc định: `53`).
* `episode_duration_minutes`: Thời lượng tiết học (chuẩn: `60` phút).
* `mc_rollout_samples`: Số mẫu Monte-Carlo khi chạy kịch bản phản thực What-if (`30`).

---

## ⚡ 1 Entrypoint Duy nhất Cho Toàn bộ Chu trình: `scripts/pipeline.py`

Hệ thống cung cấp một điểm thực thi duy nhất **[`scripts/pipeline.py`](file:///d:/SCIC%202026/SCIC-2026/scripts/pipeline.py)** tự động điều phối khép kín luồng làm việc 4 tầng:
```
[Tiền xử lý Zip -> Parquet] -> [Huấn luyện GRU-DKT] -> [Xây dựng Env Đa nguồn] -> [Huấn luyện CQL Policy] -> [Release Manifest]
```

### 1. Chạy Toàn bộ Pipeline Mặc định (Full Pipeline Run)
```bash
python scripts/pipeline.py
```
> **Cơ chế Tự động Nhận diện (Smart Auto-detection):** Nếu hệ thống phát hiện thư mục `data/shards/` đã có sẵn các file Parquet, script sẽ **tự động bỏ qua Stage 1 (Tiền xử lý)** và chuyển ngay sang huấn luyện GRU-DKT và CQL để tiết kiệm thời gian!

### 2. Chạy Nhanh để Kiểm thử trên CPU (Fast Smoke Run)
Chạy thử 2 shard dữ liệu, 50 sinh viên, 1 epoch DKT, 5 episode mô phỏng trong vòng 10 giây:
```bash
python scripts/pipeline.py --num-shards 2 --max-students 50 --estimator-epochs 1 --policy-episodes 5 --policy-steps 10
```

### 3. Huấn luyện Quy mô Lớn trên GPU (Production Run)
Huấn luyện trên toàn bộ shards Parquet và 100 phiên mô phỏng lớp học:
```bash
python scripts/pipeline.py --num-shards -1 --estimator-epochs 10 --policy-episodes 100 --policy-steps 500
```

### 4. Tùy chọn Bỏ qua hoặc Ép chạy từng Giai đoạn
* `--force-preprocess`: Ép nén lại toàn bộ Parquet shards từ file ZIP.
* `--skip-preprocess`: Bỏ qua tiền xử lý dữ liệu.
* `--skip-estimator`: Bỏ qua huấn luyện GRU-DKT (sử dụng checkpoint có sẵn).
* `--skip-policy`: Bỏ qua huấn luyện chính sách CQL.
* `--use-synthetic-twin`: Chuyển sang mô phỏng thuần tham số Beta thay vì dữ liệu đa nguồn thật.

### 5. Huấn luyện Lại Riêng Chính sách CQL & Inverse RL
Khi bạn đã có sẵn checkpoint GRU-DKT và chỉ muốn tinh chỉnh hoặc huấn luyện lại chính sách sư phạm an toàn:
```bash
# Huấn luyện CQL với hàm thưởng hiển thị cải tiến
python scripts/train_policy.py --reward-mode explicit --episodes 40 --train-steps 150

# Huấn luyện Inverse RL (MaxEnt IRL) với bộ sinh chuyên gia ngẫu nhiên mềm
python scripts/train_policy.py --reward-mode irl --episodes 40 --train-steps 150
```

### 6. Đánh giá Độc lập & Kiểm tra Reward Hacking (Pre-Deployment Benchmark)
Đo lường toàn diện chất lượng mô hình ước lượng (Calibration, Uncertainty, Dynamic Range) và chẩn đoán hiện tượng Mode Collapse / Reward Hacking của các chính sách:
```bash
python scripts/evaluate_models.py
```
> **Chỉ số Sức khỏe Sư phạm (Shannon Action Entropy):** Đo độ đa dạng hoạt động $H(\pi) = -\sum_a P(a) \log_2 P(a)$ trên môi trường Digital Twin. Tự động gắn cờ cảnh báo `⚠️ YES (Hacked)` nếu chính sách bị co cụm vào 1--2 lựa chọn hoặc entropy $< 0.40$. Xuất báo cáo JSON tại `output/artifacts/evaluation_report.json`.

---

## 💻 Cấu trúc Dự án & Chạy Cục bộ (Local Environment)

### Cài đặt Môi trường
```bash
# Cài đặt gói ở chế độ phát triển
pip install -e .
```

### Chạy Bộ Kiểm thử
```bash
python -c "
import sys, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
sys.path.insert(0, str(Path('src').resolve()))
for tf in ['tests/test_schemas.py', 'tests/test_state_estimation.py', 'tests/test_twin.py', 'tests/test_cql.py', 'tests/test_inference.py']:
    spec = importlib.util.spec_from_file_location(Path(tf).stem, tf)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for a in dir(mod):
        if a.startswith('test_'): getattr(mod, a)()
print('13/13 tests passed!')
"
```

### Khởi chạy Server API Hỗ trợ Quyết định (FastAPI Backend)
```bash
python scripts/run_api.py --dev
```
Truy cập tài liệu Swagger UI tương tác tại: [http://localhost:8000/docs](http://localhost:8000/docs)
Lệnh này tự động mở giao diện UI React ở [http://localhost:3000](http://localhost:3000).
Các API endpoints sẵn sàng phục vụ giảng đường:
* `POST /v1/events:batch`: Tiếp nhận luồng sự kiện tương tác của sinh viên.
* `GET /v1/classes/{class_id}/state`: Lấy snapshot trạng thái tổng hợp của lớp học (độ trễ $< 1\text{s}$).
* `POST /v1/recommendations`: Nhận gợi ý can thiệp sư phạm tối ưu kèm giải thích và kịch bản mô phỏng What-if.
* `POST /v1/teacher-decisions`: Ghi nhận quyết định phê duyệt/tùy biến của giảng viên (Human-in-the-Loop).
* `POST /v1/simulations`: Chạy mô phỏng phản thực Monte-Carlo cho danh sách hành động ứng viên. <br>
---

## 📄 Tài liệu Đề án (LaTeX & PDF)

* **Bản thảo Đề án Toàn diện (29 trang):** [docs/ClassTwin_Proposal_SCIC2026.pdf](file:///d:/SCIC%202026/SCIC-2026/docs/ClassTwin_Proposal_SCIC2026.pdf)
* **Mã nguồn LaTeX:** [docs/ClassTwin_Proposal_SCIC2026.tex](file:///d:/SCIC%202026/SCIC-2026/docs/ClassTwin_Proposal_SCIC2026.tex)
* **Đặc tả Thiết kế Kỹ thuật:** [docs/build_technical_design.py](file:///d:/SCIC%202026/SCIC-2026/docs/build_technical_design.py)

---

## 📜 Giấy phép (License)

Dự án được phân phối dưới giấy phép [MIT License](file:///d:/SCIC%202026/SCIC-2026/LICENSE).
