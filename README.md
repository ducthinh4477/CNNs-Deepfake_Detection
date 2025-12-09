# 🧠 Deepfake Detection using Convolutional Neural Networks (CNNs)

[![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch&logoColor=white)]()
[![Status](https://img.shields.io/badge/Status-Course%20Project-green)]()
[![License](https://img.shields.io/badge/License-Educational-lightgrey)]()

> **Đồ án môn học Công nghệ Thông tin – Xây dựng mô hình CNN phát hiện ảnh giả mạo (Deepfake / AI-generated)**

---

## 👤 Thông tin chung

- **Sinh viên thực hiện:** Nguyễn Đức Thịnh  
- **Trường:** Đại học Sư phạm Kỹ thuật TP.HCM (HCMUTE)  
- **Giảng viên hướng dẫn:** TS. Lê Văn Vinh  
- **Môn học:** Công nghệ Thông tin  

---

## 📌 Giới thiệu (Introduction)

Dự án này tập trung vào việc **nghiên cứu và xây dựng một mô hình Convolutional Neural Network (CNN) thủ công** nhằm phát hiện sự khác biệt giữa:

- ✅ **REAL** – Ảnh thật
- ❌ **FAKE** – Ảnh được sinh ra bởi AI (Deepfake / Synthetic Image)

Mô hình được xây dựng bằng **PyTorch**, huấn luyện và đánh giá trên bộ dữ liệu **CIFAKE**, đạt được:

- 🎯 **Test Accuracy ≈ 94% sau 10 epochs**
- Học ổn định, giảm overfitting nhờ BatchNorm & Dropout

---

## 📂 Dataset

- **Tên dataset:** CIFAKE – Real and AI-Generated Synthetic Images  
- **Nguồn:** Kaggle  
  👉 https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images  

### 📊 Thông tin chi tiết
| Thành phần | Số lượng |
|-----------|----------|
| Training images | 100,000 |
| Testing images | 20,000 |
| Số lớp | 2 (REAL / FAKE) |
| Kích thước ảnh | 224 × 224 (resize trong code) |

---

## 🛠️ Công nghệ sử dụng (Tech Stack)

- **Ngôn ngữ:** Python
- **Deep Learning Framework:** PyTorch
- **Môi trường:** Google Colab (GPU NVIDIA T4)

### 🔧 Kỹ thuật chính
- Data Augmentation:
  - RandomHorizontalFlip
  - RandomRotation
  - ColorJitter
- Custom CNN Architecture
- Batch Normalization
- Dropout chống overfitting
- SGD Optimizer với Momentum

---

## 🧠 Kiến trúc mô hình (Model Architecture)

Mô hình **MyNet** gồm 4 khối tích chập (Conv Blocks):

Input Image (3 x 224 x 224)
│
├── Conv Block 1: Conv2d (3 → 32) → BatchNorm → ReLU → MaxPool
├── Conv Block 2: Conv2d (32 → 64) → BatchNorm → ReLU → MaxPool
├── Conv Block 3: Conv2d (64 → 128) → BatchNorm → ReLU → MaxPool
├── Conv Block 4: Conv2d (128 → 256) → BatchNorm → ReLU → MaxPool
│
├── Flatten
├── Linear (512) → ReLU → Dropout (0.5)
└── Linear (2 classes: REAL / FAKE)


### ⚙️ Cấu hình huấn luyện
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** SGD  
  - Learning rate = 0.001  
  - Momentum = 0.9  
- **Epochs:** 10  

---

## 🚀 Hướng dẫn chạy (How to Run)

### 1️⃣ Cài đặt thư viện

pip install torch torchvision matplotlib kaggle

### 2️⃣ Chuẩn bị Kaggle API

Dự án chạy tốt nhất trên Google Colab và tải dataset tự động từ Kaggle.

Các bước:

Đăng nhập Kaggle → Account → Settings

Chọn Create New Token

Tải file kaggle.json

Upload file này khi notebook yêu cầu

### 3️⃣ Training & Testing

Mở file notebook:

CNNs_Deepfake_Detection.ipynb


Chạy lần lượt các bước:

Tải & giải nén dataset

Preprocessing & DataLoader

Khởi tạo mô hình CNN

Training loop

Evaluation & Visualization

## 📊 Kết quả (Results)

Sau 10 epochs huấn luyện:

Chỉ số	Giá trị
Training Loss	~0.17
Test Accuracy	~94.6%
Overfitting	Thấp

📈 Biểu đồ Loss & Accuracy được sinh tự động trong notebook sau khi training.

## 🧪 Nhận xét & Hạn chế

### ✅ Ưu điểm:

Kiến trúc CNN tự xây dựng, dễ hiểu

Accuracy cao với dataset lớn

Huấn luyện ổn định

### ⚠️ Hạn chế:

Chỉ sử dụng CNN cơ bản

Chưa khai thác đặc trưng miền tần số (FFT/DCT)

Chưa so sánh với các mô hình SOTA (Xception, EfficientNet, ViT)

### 🔮 Hướng phát triển

So sánh CNN với Transfer Learning (ResNet, EfficientNet)

Áp dụng Frequency Domain Analysis (FFT / F3Net)

Thử nghiệm video deepfake (FaceForensics++)

Triển khai Web demo (Streamlit / Flask)

### 📝 License

Dự án được thực hiện phục vụ mục đích học tập và nghiên cứu,
không sử dụng cho mục đích thương mại.
