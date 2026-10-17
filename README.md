# ClassTwin: Classroom Digital Twin (SCIC 2026)

<p align="center">
  <img src="https://img.shields.io/badge/Project-SCIC%202026-007A87?style=for-the-badge" alt="SCIC 2026"/>
  <img src="https://img.shields.io/badge/Domain-AI%20in%20Education%20(AIED)-1A365D?style=for-the-badge" alt="AIED"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License MIT"/>
</p>

<p align="center">
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

## ⚙️ Hướng dẫn Điều chỉnh Tham số (Chỉnh ở đâu?)

Mọi tham số của dự án đều được tách biệt hoàn toàn trong thư mục [`configs/`](file:///d:/SCIC%202026/SCIC-2026/configs/) dưới dạng file YAML rõ ràng, không hardcode trong mã nguồn:

### 1. Tham số Dữ liệu & Tiền xử lý ([`configs/preprocess.yaml`](file:///d:/SCIC%202026/SCIC-2026/configs/preprocess.yaml))
* `raw_zip_path`: Đường dẫn tới file `EdNet-KT1.zip`.
* `shards_dir`: Thư mục lưu trữ các file Parquet phân mảnh (`train/`, `val/`, `test/`).
* `chunk_size_rows`: Kích thước mỗi shard Parquet (mặc định: `50,000` dòng/shard).
* `train_ratio`, `val_ratio`, `test_ratio`: Tỷ lệ chia tập theo Protocol A (mặc định: `0.8 / 0.1 / 0.1` theo hash `student_id`).
* `seed`: Seed ngẫu nhiên cố định để tái lập kết quả phân chia.

### 2. Tham số Mô hình Ước lượng Trạng thái ([`configs/estimator.yaml`](file:///d:/SCIC%202026/SCIC-2026/configs/estimator.yaml))
* `d_model`: Kích thước vector nhúng câu hỏi (mặc định: `64`).
* `d_hidden`: Kích thước vector ẩn của mạng GRU (mặc định: `128`).
* `dropout`: Tỷ lệ dropout chống quá khớp (mặc định: `0.1`).
* `learning_rate`: Tốc độ học tối ưu hóa AdamW (mặc định: `0.001`).
* `batch_size`: Số lượng chuỗi trong mỗi mini-batch (mặc định: `64`).
* `max_epochs`: Số epoch huấn luyện tối đa (mặc định: `20`).
* `probe_sample_size`: Số lượng câu hỏi kiểm định cố định theo kỹ năng để tính Probe Mastery.

### 3. Tham số Chính sách Offline RL & Digital Twin ([`configs/policy_sim.yaml`](file:///d:/SCIC%202026/SCIC-2026/configs/policy_sim.yaml))
* `d_state`: Chiều của vector trạng thái lớp học $\mathbf{x}_t$ (chuẩn: `12`).
* `cql_alpha`: Hệ số phạt bảo thủ CQL dìm giá trị Q ngoài phân phối (mặc định: `1.0`).
* `gamma_base`: Hệ số chiết khấu cơ sở (mặc định: `0.99`).
* `tau_seconds`: Hằng số thời gian chuẩn hóa bước chuyển Bán Markov (mặc định: `60.0` giây).
* `workload_budget`: Ngân sách công sức tối đa cho phép của giảng viên trong 1 tiết (mặc định: `15.0`).
* `default_cohort_size`: Sĩ số lớp học mô phỏng (mặc định: `60` sinh viên).
* `mc_rollout_samples`: Số mẫu Monte-Carlo khi chạy kịch bản phản thực What-if (mặc định: `20`).

### 4. Ghi đè nhanh qua Command Line (CLI Overrides)
Bạn có thể ghi đè tham số trực tiếp qua dòng lệnh mà không cần sửa file YAML:
```bash
# Thay đổi số epoch của estimator và số session của policy:
python scripts/pipeline.py --estimator-epochs 15 --policy-episodes 50 --policy-steps 200
```

---

## 🚀 Hướng dẫn Chạy Chi tiết trên Kaggle (GPU Miễn phí)

Kaggle cung cấp GPU T4 x2 hoặc P100 với 30 giờ tính toán miễn phí/tuần. Dưới đây là hướng dẫn từ A-Z:

### Bước 1: Upload Dữ liệu EdNet lên Kaggle
1. Truy cập [Kaggle Datasets](https://www.kaggle.com/datasets) và nhấn nút **New Dataset**.
2. Đặt tiêu đề Dataset là: `ednet-kt1`.
3. Kéo thả file `EdNet-KT1.zip` vào ô tải lên và nhấn **Create**.
4. Sau khi tải xong, đường dẫn dữ liệu trên Kaggle sẽ là:
   ```
   /kaggle/input/ednet-kt1/EdNet-KT1.zip
   ```

### Bước 2: Tạo Kaggle Notebook & Bật GPU
1. Nhấn nút **New Notebook** trên Kaggle.
2. Tại cột menu bên phải (**Notebook options**):
   * **Accelerator:** Chọn `GPU T4 x2` hoặc `GPU P100`.
   * **Internet:** Chọn `Always On` (để clone git và tải thư viện).
3. Nhấn **+ Add Data** ở góc phải và chọn dataset `ednet-kt1` vừa tạo ở Bước 1.

### Bước 3: Clone Mã nguồn và Kiểm tra
Trong ô code đầu tiên của Notebook, chạy:
```bash
# 1. Clone mã nguồn dự án vào thư mục làm việc /kaggle/working/classtwin
!git clone https://github.com/NguyenDucAnh-2k6/SCIC-2026.git /kaggle/working/classtwin
%cd /kaggle/working/classtwin

# 2. Chạy bộ kiểm thử tự động xác minh môi trường (13 unit tests)
!python -c "
import sys, importlib.util
from pathlib import Path
sys.path.insert(0, '/kaggle/working/classtwin')
sys.path.insert(0, '/kaggle/working/classtwin/src')
for tf in ['tests/test_schemas.py', 'tests/test_state_estimation.py', 'tests/test_twin.py', 'tests/test_cql.py', 'tests/test_inference.py']:
    spec = importlib.util.spec_from_file_location(Path(tf).stem, tf)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for a in dir(mod):
        if a.startswith('test_'): getattr(mod, a)()
print('All 13 unit tests passed successfully on Kaggle!')
"
```

### Bước 4: Chạy Luồng Pipeline Hợp nhất trên Kaggle
Chạy toàn bộ chu trình tiền xử lý, huấn luyện Estimator và huấn luyện CQL bằng 1 lệnh duy nhất:
```bash
!python scripts/pipeline.py \
    --config configs/kaggle_pipeline.yaml \
    --output-dir /kaggle/working/output \
    --estimator-epochs 15 \
    --policy-episodes 40 \
    --policy-steps 150
```

> **Ghi chú về cơ chế Resume:** Nếu phiên làm việc Kaggle bị ngắt, bạn chỉ cần chạy lại lệnh trên. File `processed_ledger.json` sẽ tự động bỏ qua các file đã xử lý và tiếp tục từ vị trí dừng.

### Bước 5: Kiểm tra và Tải về Checkpoint
Tất cả checkpoint và tài liệu phát hành sẽ được lưu tại `/kaggle/working/output`:
```bash
!ls -lh /kaggle/working/output/models
!cat /kaggle/working/output/release_manifest.json
```
Bạn có thể tải trực tiếp file `release_manifest.json`, `gru_dkt.pt` và `cql_policy.pt` từ mục **Data > Output** ở cột bên phải giao diện Kaggle.

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

### Khởi chạy Server API Hỗ trợ Quyết định (FastAPI)
```bash
python scripts/run_api.py
```
Truy cập tài liệu Swagger UI tương tác tại: [http://localhost:8000/docs](http://localhost:8000/docs)

Các API endpoints sẵn sàng phục vụ giảng đường:
* `POST /v1/events:batch`: Tiếp nhận luồng sự kiện tương tác của sinh viên.
* `GET /v1/classes/{class_id}/state`: Lấy snapshot trạng thái tổng hợp của lớp học (độ trễ $< 1\text{s}$).
* `POST /v1/recommendations`: Nhận gợi ý can thiệp sư phạm tối ưu kèm giải thích và kịch bản mô phỏng What-if.
* `POST /v1/teacher-decisions`: Ghi nhận quyết định phê duyệt/tùy biến của giảng viên (Human-in-the-Loop).
* `POST /v1/simulations`: Chạy mô phỏng phản thực Monte-Carlo cho danh sách hành động ứng viên.

---

## 📄 Tài liệu Đề án (LaTeX & PDF)

* **Bản thảo Đề án Toàn diện (19 trang):** [docs/ClassTwin_Proposal_SCIC2026.pdf](file:///d:/SCIC%202026/SCIC-2026/docs/ClassTwin_Proposal_SCIC2026.pdf)
* **Mã nguồn LaTeX:** [docs/ClassTwin_Proposal_SCIC2026.tex](file:///d:/SCIC%202026/SCIC-2026/docs/ClassTwin_Proposal_SCIC2026.tex)
* **Đặc tả Thiết kế Kỹ thuật:** [docs/build_technical_design.py](file:///d:/SCIC%202026/SCIC-2026/docs/build_technical_design.py)

---

## 📜 Giấy phép (License)

Dự án được phân phối dưới giấy phép [MIT License](file:///d:/SCIC%202026/SCIC-2026/LICENSE). Bản quyền thuộc về **Nguyen Duc Anh & Nhóm Nghiên cứu ClassTwin (SCIC 2026)**.
