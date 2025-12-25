# 🛡️ DeepScan - Hệ Thống Phát Hiện Deepfake

<div align="center">

![DeepScan Banner](https://img.shields.io/badge/DeepScan-Deepfake%20Detection-blue?style=for-the-badge&logo=shield)

**Phát hiện ảnh giả mạo và ảnh được tạo bởi AI với độ tin cậy cao sử dụng phân tích pháp y nâng cao.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=flat-square&logo=next.js&logoColor=white)](https://nextjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=flat-square&logo=tailwind-css&logoColor=white)](https://tailwindcss.com)

[Tính năng](#-tính-năng) • [Kiến trúc](#-kiến-trúc) • [Cài đặt](#-cài-đặt) • [Sử dụng](#-sử-dụng) • [Phương pháp phân tích](#-phương-pháp-phân-tích-pháp-y)

</div>

---

## 🎯 Tổng quan

**DeepScan** là ứng dụng full-stack phát hiện deepfake kết hợp mô hình CNN mạnh mẽ dựa trên PyTorch với giao diện web hiện đại, trực quan. Upload bất kỳ hình ảnh nào và nhận phân tích tức thì với các công cụ trực quan hóa pháp y để hiểu *tại sao* một hình ảnh có thể là giả mạo.

### ✨ Điểm nổi bật

- 🧠 **Custom CNN Model** được huấn luyện trên dataset CIFAKE với **độ chính xác 94%**
- 🔥 **Phân tích thời gian thực** với kết quả dự đoán tức thì
- 🎨 **Trực quan hóa pháp y** bao gồm Heatmap (Grad-CAM) và Fourier Analysis
- ⚡ **Thanh trượt Confidence Threshold** để điều chỉnh ngưỡng quyết định động
- 🌙 **Giao diện tối hiện đại** với bố cục 3 cột chuyên nghiệp

---

## 🖼️ Ảnh chụp màn hình

<div align="center">

### Giao diện chính - Bố cục 3 cột

| Panel Upload | Không gian phân tích | Panel báo cáo |
|:---:|:---:|:---:|
| ![Upload](https://via.placeholder.com/250x400/1e293b/60a5fa?text=Upload+Panel) | ![Workspace](https://via.placeholder.com/400x400/1e293b/60a5fa?text=Image+Workspace) | ![Report](https://via.placeholder.com/250x400/1e293b/60a5fa?text=Analysis+Report) |
| *Upload file & thông tin model* | *Xem ảnh gốc, Heatmap, Fourier* | *Kết quả & ảnh phân tích* |

### Kết quả phân tích

| Phát hiện ảnh thật | Phát hiện ảnh giả |
|:---:|:---:|
| ![Real](https://via.placeholder.com/350x250/1e293b/10b981?text=✓+AUTHENTIC) | ![Fake](https://via.placeholder.com/350x250/1e293b/ef4444?text=✗+LIKELY+FAKE) |

</div>

> 📸 *Thay thế các ảnh placeholder bằng ảnh chụp màn hình thực tế của ứng dụng*

---

## 🏗️ Kiến trúc

DeepScan sử dụng **Bố cục 3 cột** hiện đại được thiết kế cho phân tích pháp y chuyên nghiệp:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Ứng dụng DeepScan                               │
├──────────────┬────────────────────────────────┬─────────────────────┤
│              │                                │                     │
│   📁 TRÁI    │        🖼️ GIỮA                │     📊 PHẢI         │
│   SIDEBAR    │        WORKSPACE              │     PANEL           │
│              │                                │                     │
│  • Upload    │  • Xem ảnh gốc                │  • Badge trạng thái │
│  • Thông tin │  • Lớp phủ Heatmap            │  • Gauge confidence │
│    file      │  • Phân tích Fourier          │  • Thanh ngưỡng     │
│  • Thông tin │  • Điều hướng tab             │  • Ảnh pháp y       │
│    Model     │                                │  • Chi tiết         │
│  • Hành động │                                │                     │
│              │                                │                     │
└──────────────┴────────────────────────────────┴─────────────────────┘
```

### Tech Stack

| Tầng | Công nghệ | Mục đích |
|-------|------------|---------|
| **Frontend** | Next.js 14 + React | Framework React hiện đại với SSR |
| **Styling** | Tailwind CSS | Utility-first CSS cho giao diện tối |
| **Backend** | FastAPI | Python API hiệu năng cao |
| **AI Engine** | PyTorch | Deep learning inference |
| **Model** | Custom CNN | Huấn luyện trên dataset CIFAKE |

---

## 🚀 Cài đặt

### Yêu cầu hệ thống

- Python 3.9+
- Node.js 18+
- Git

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/CNNs-Deepfake_Detection.git
cd CNNs-Deepfake_Detection
```

### 2️⃣ Cài đặt Backend (FastAPI)

```bash
# Di chuyển vào thư mục backend
cd DF_Web

# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt

# Tải trọng số model (nếu chưa có)
# Đặt file 'custom_cnn_cifake.pth' vào thư mục DF_Web

# Khởi động FastAPI server
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

✅ Backend sẽ chạy tại: `http://127.0.0.1:8000`  
📚 Tài liệu API: `http://127.0.0.1:8000/docs`

### 3️⃣ Cài đặt Frontend (Next.js)

```bash
# Mở terminal mới và di chuyển vào thư mục frontend
cd frontend

# Cài đặt dependencies
npm install

# Khởi động development server
npm run dev
```

✅ Frontend sẽ chạy tại: `http://localhost:3000`

---

## 📖 Sử dụng

1. **Mở** ứng dụng tại `http://localhost:3000`
2. **Upload** một hình ảnh sử dụng sidebar bên trái (PNG, JPG tối đa 10MB)
3. **Nhấn** "Analyze Image" để chạy phát hiện deepfake
4. **Xem** kết quả ở panel bên phải:
   - Badge trạng thái (AUTHENTIC / LIKELY FAKE)
   - Gauge phần trăm độ tin cậy
   - Mức độ đánh giá rủi ro
5. **Điều chỉnh** thanh trượt Confidence Threshold để thay đổi ngưỡng quyết định
6. **Khám phá** các công cụ trực quan hóa pháp y:
   - Chuyển đổi giữa các chế độ xem Original, Heatmap và Fourier
   - Kiểm tra các ảnh pháp y trong panel báo cáo

---

## 🔬 Phương pháp Phân tích Pháp y

### 🔥 Heatmap (Grad-CAM)

**Gradient-weighted Class Activation Mapping** trực quan hóa những vùng nào của ảnh mà CNN tập trung vào khi đưa ra dự đoán.

| Hiển thị gì | Cách diễn giải |
|---------------|------------------|
| **Vùng Đỏ/Vàng** | Kích hoạt cao - model tập trung tại đây |
| **Vùng Xanh dương/Xanh lá** | Kích hoạt thấp - ít quan trọng cho quyết định |
| **Điểm nóng tập trung** | Có thể chỉ ra các vùng bị chỉnh sửa |

> 💡 *Nếu heatmap hiển thị sự tập trung bất thường ở các khu vực cụ thể (mắt, miệng, cạnh), ảnh có thể bị chỉnh sửa cục bộ.*

### 📊 Fourier Frequency Analysis

**Fast Fourier Transform (FFT)** chuyển đổi ảnh sang miền tần số, tiết lộ các mẫu không nhìn thấy được bằng mắt thường.

| Hiển thị gì | Cách diễn giải |
|---------------|------------------|
| **Độ sáng trung tâm** | Các thành phần tần số thấp (cấu trúc tổng thể) |
| **Mẫu cạnh** | Chi tiết tần số cao (texture, cạnh) |
| **Grid artifacts** | Có thể chỉ ra ảnh được tạo bởi GAN |
| **Đối xứng bất thường** | Có thể gợi ý quá trình tạo tổng hợp |

> 💡 *Ảnh được tạo bởi AI thường có dấu vân tay tần số đặc trưng khác với ảnh chụp thật.*

---

## 📁 Cấu trúc dự án

```
CNNs-Deepfake_Detection/
├── 📂 DF_Web/                    # Backend (FastAPI)
│   ├── api.py                    # REST API endpoints
│   ├── ai_logic.py               # AI engine & phương pháp pháp y
│   ├── model.py                  # Định nghĩa kiến trúc CNN
│   ├── custom_cnn_cifake.pth     # Trọng số model đã huấn luyện
│   ├── model_config.json         # Cấu hình model
│   └── requirements.txt          # Python dependencies
│
├── 📂 frontend/                  # Frontend (Next.js)
│   ├── app/
│   │   ├── page.jsx              # Trang ứng dụng chính
│   │   ├── layout.js             # Root layout
│   │   └── globals.css           # Global styles + Tailwind
│   ├── package.json              # Node.js dependencies
│   ├── tailwind.config.js        # Cấu hình Tailwind
│   └── next.config.js            # Cấu hình Next.js
│
├── CNNs_Deepfake_Detection.ipynb # Notebook huấn luyện
├── .gitignore                    # Git ignore rules
└── README.md                     # File này
```

---

## 🔌 API Reference

### Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Trạng thái health chi tiết |
| `POST` | `/analyze` | Phân tích ảnh với công cụ trực quan hóa pháp y |
| `GET` | `/docs` | Tài liệu API Swagger |

### Ví dụ Request

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

### Ví dụ Response

```json
{
  "filename": "image.jpg",
  "is_real": true,
  "label": 1,
  "confidence_score": 0.9423,
  "risk_level": "Low",
  "trust_score": "Very High",
  "timestamp": "2025-12-25T10:30:00",
  "model_used": "CNN Custom (CIFAKE)",
  "heatmap": "base64_encoded_png...",
  "fourier": "base64_encoded_png..."
}
```

---

## 🧠 Thông tin Model

| Thuộc tính | Giá trị |
|----------|-------|
| **Architecture** | Custom CNN |
| **Training Dataset** | CIFAKE |
| **Accuracy** | 94% |
| **Input Size** | 224 × 224 pixels |
| **Output** | Binary (Real/Fake) |
| **Framework** | PyTorch |

---

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! Vui lòng tạo Pull Request.

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/TinhNangMoi`)
3. Commit các thay đổi (`git commit -m 'Thêm tính năng mới'`)
4. Push lên branch (`git push origin feature/TinhNangMoi`)
5. Mở Pull Request

---

## 📄 License

Dự án này được cấp phép theo MIT License - xem file [LICENSE](LICENSE) để biết chi tiết.

---

## 🙏 Cảm ơn

- **CIFAKE Dataset** cho dữ liệu huấn luyện
- **PyTorch** team cho deep learning framework
- **FastAPI** cho Python web framework xuất sắc
- **Vercel** cho Next.js và các công cụ frontend

---

<div align="center">

**Được xây dựng với ❤️ bởi Nhóm Đồ án Tốt nghiệp HCMUTE**

⭐ Star repo này nếu bạn thấy hữu ích! ⭐

</div>