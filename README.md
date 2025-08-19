# Dự án MLOps - Triển khai Mô hình Machine Learning

## 📋 Tổng quan dự án

Đây là dự án MLOps hoàn chỉnh bao gồm:
- Huấn luyện mô hình với nhiều thuật toán
- Quản lý phiên bản mô hình với MLflow
- API web với Flask
- Triển khai mô hình sẵn sàng sử dụng

## 🏗️ Cấu trúc dự án

```
Hung/
├── app.py                 # Ứng dụng web Flask
├── requirements.txt       # Thư viện Python cần thiết
├── README.md             # File này
├── src/
│   ├── train.py          # Script huấn luyện mô hình
│   └── config.yaml       # Cấu hình huấn luyện
├── templates/
│   └── index.html        # Giao diện web
└── mlruns/               # Dữ liệu theo dõi MLflow
```

## 🚀 Hướng dẫn nhanh

### Yêu cầu
- Python 3.10
- Anaconda hoặc Miniconda

### 1. Cài đặt môi trường

```bash
# Tạo môi trường conda
conda create -n mlops-project python=3.10

# Kích hoạt môi trường
conda activate mlops-project

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Khởi động MLflow Server

```bash
# Khởi động MLflow (Terminal 1)
mlflow server --host 0.0.0.0 --port 5000
```

### 3. Huấn luyện mô hình

```bash
# Vào thư mục src
cd src

# Chạy script huấn luyện
python train.py
```

Quá trình này sẽ:
- Tạo dữ liệu phân loại
- Huấn luyện nhiều mô hình (LogisticRegression, RandomForest, SVC)
- Tìm tham số tốt nhất
- Lưu kết quả vào MLflow
- Đăng ký mô hình tốt nhất

### 4. Chạy API Flask

```bash
# Về thư mục gốc
cd ..

# Khởi động ứng dụng Flask (Terminal 2)
python app.py
```

API sẽ chạy tại: `http://localhost:5001`

## 📊 Huấn luyện mô hình

### Cấu hình
Chỉnh sửa `src/config.yaml` để thay đổi:
- Thông số dữ liệu (số mẫu, số đặc trưng, số lớp)
- Tham số mô hình
- Tỷ lệ chia dữ liệu

### Các mô hình hỗ trợ
- **LogisticRegression**: Phân loại tuyến tính
- **RandomForestClassifier**: Phân loại ensemble
- **SVC**: Support Vector Classification

## 🌐 Sử dụng API

### Giao diện web
Truy cập `http://localhost:5001` để sử dụng giao diện web.

### API dự đoán
```bash
POST http://localhost:5001/predict
Content-Type: application/json

{
  "feature_0": 0.5,
  "feature_1": -0.2,
  "feature_2": 1.1,
  ...
  "feature_19": 0.8
}
```

### Ví dụ sử dụng Python
```python
import requests

# Dữ liệu mẫu (20 đặc trưng)
data = {
    "feature_0": 0.5, "feature_1": -0.2, "feature_2": 1.1,
    "feature_3": 0.3, "feature_4": -0.8, "feature_5": 0.9,
    "feature_6": -0.1, "feature_7": 0.7, "feature_8": -0.4,
    "feature_9": 0.6, "feature_10": 0.2, "feature_11": -0.9,
    "feature_12": 0.8, "feature_13": -0.3, "feature_14": 0.4,
    "feature_15": -0.6, "feature_16": 0.1, "feature_17": 0.9,
    "feature_18": -0.7, "feature_19": 0.3
}

response = requests.post(
    "http://localhost:5001/predict",
    json=data
)
print(response.json())
```

## 🔧 MLflow

### Thông tin mô hình
Ứng dụng sẽ hiển thị:
- Phiên bản mô hình
- Giai đoạn hiện tại
- ID chạy huấn luyện

### Giao diện MLflow
Truy cập `http://localhost:5000` để xem:
- Theo dõi thí nghiệm
- Đăng ký mô hình
- Lưu trữ dữ liệu

## 🛠️ Phát triển

### Thêm mô hình mới
1. Import lớp mô hình trong `src/train.py`
2. Thêm cấu hình vào `src/config.yaml`
3. Script sẽ tự động bao gồm mô hình mới

### Tùy chỉnh cấu hình
Chỉnh sửa `src/config.yaml`:
```yaml
dataset:
  n_samples: 2000      # Số mẫu
  n_features: 20       # Số đặc trưng
  n_informative: 10    # Đặc trưng có ích
  n_classes: 2         # Số lớp
  test_size: 0.2       # Tỷ lệ test
  random_state: 42     # Seed ngẫu nhiên

models:
  YourModel:
    param1: [value1, value2]
    param2: [value3, value4]
```
