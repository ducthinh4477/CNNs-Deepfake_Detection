# 🛡️ DeepScan - AI-Powered Deepfake Detection System

<div align="center">

![DeepScan Banner](https://img.shields.io/badge/DeepScan-Deepfake%20Detection-blue?style=for-the-badge&logo=shield)

**Detect AI-generated and manipulated images with confidence using advanced forensic analysis.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=flat-square&logo=next.js&logoColor=white)](https://nextjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=flat-square&logo=tailwind-css&logoColor=white)](https://tailwindcss.com)

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Forensic Methods](#-forensic-methods)

</div>

---

## 🎯 Overview

**DeepScan** is a full-stack deepfake detection application that combines a powerful PyTorch-based CNN model with an intuitive modern web interface. Upload any image and get instant analysis with forensic visualizations to understand *why* an image might be fake.

### ✨ Key Highlights

- 🧠 **Custom CNN Model** trained on CIFAKE dataset with **94% accuracy**
- 🔥 **Real-time Analysis** with instant prediction results
- 🎨 **Forensic Visualizations** including Heatmap (Grad-CAM) and Fourier Analysis
- ⚡ **Confidence Threshold Slider** for dynamic result interpretation
- 🌙 **Modern Dark UI** with a professional 3-pane layout

---

## 🖼️ Screenshots

<div align="center">

### Main Interface - 3-Pane Layout

| Upload Panel | Analysis Workspace | Report Panel |
|:---:|:---:|:---:|
| ![Upload](https://via.placeholder.com/250x400/1e293b/60a5fa?text=Upload+Panel) | ![Workspace](https://via.placeholder.com/400x400/1e293b/60a5fa?text=Image+Workspace) | ![Report](https://via.placeholder.com/250x400/1e293b/60a5fa?text=Analysis+Report) |
| *File upload & model info* | *Original, Heatmap, Fourier views* | *Results & forensic images* |

### Analysis Results

| Authentic Image Detected | Fake Image Detected |
|:---:|:---:|
| ![Real](https://via.placeholder.com/350x250/1e293b/10b981?text=✓+AUTHENTIC) | ![Fake](https://via.placeholder.com/350x250/1e293b/ef4444?text=✗+LIKELY+FAKE) |

</div>

> 📸 *Replace placeholder images with actual screenshots of your application*

---

## 🏗️ Architecture

DeepScan uses a modern **3-Pane Layout** designed for professional forensic analysis:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DeepScan Application                         │
├──────────────┬────────────────────────────────┬─────────────────────┤
│              │                                │                     │
│   📁 LEFT    │        🖼️ CENTER              │     📊 RIGHT        │
│   SIDEBAR    │        WORKSPACE              │     PANEL           │
│              │                                │                     │
│  • Upload    │  • Original Image View        │  • Status Badge     │
│  • File Info │  • Heatmap Overlay            │  • Confidence Gauge │
│  • Model     │  • Fourier Analysis           │  • Threshold Slider │
│    Info      │  • Tab Navigation             │  • Forensic Images  │
│  • Actions   │                                │  • Details          │
│              │                                │                     │
└──────────────┴────────────────────────────────┴─────────────────────┘
```

### Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | Next.js 14 + React | Modern React framework with SSR |
| **Styling** | Tailwind CSS | Utility-first CSS for dark theme |
| **Backend** | FastAPI | High-performance Python API |
| **AI Engine** | PyTorch | Deep learning inference |
| **Model** | Custom CNN | Trained on CIFAKE dataset |

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- Node.js 18+
- Git

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/CNNs-Deepfake_Detection.git
cd CNNs-Deepfake_Detection
```

### 2️⃣ Backend Setup (FastAPI)

```bash
# Navigate to backend directory
cd DF_Web

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download the model weights (if not included)
# Place 'custom_cnn_cifake.pth' in the DF_Web directory

# Start the FastAPI server
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

✅ Backend will be running at: `http://127.0.0.1:8000`  
📚 API Documentation: `http://127.0.0.1:8000/docs`

### 3️⃣ Frontend Setup (Next.js)

```bash
# Open a new terminal and navigate to frontend
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

✅ Frontend will be running at: `http://localhost:3000`

---

## 📖 Usage

1. **Open** the application at `http://localhost:3000`
2. **Upload** an image using the left sidebar (PNG, JPG up to 10MB)
3. **Click** "Analyze Image" to run the deepfake detection
4. **View** results in the right panel:
   - Status badge (AUTHENTIC / LIKELY FAKE)
   - Confidence percentage gauge
   - Risk assessment level
5. **Adjust** the Confidence Threshold slider to change the decision boundary
6. **Explore** forensic visualizations:
   - Switch between Original, Heatmap, and Fourier views
   - Check the forensic images in the report panel

---

## 🔬 Forensic Methods

### 🔥 Heatmap (Grad-CAM)

**Gradient-weighted Class Activation Mapping** visualizes which regions of the image the CNN focuses on when making its prediction.

| What it shows | How to interpret |
|---------------|------------------|
| **Red/Yellow areas** | High activation - model focuses here |
| **Blue/Green areas** | Low activation - less important for decision |
| **Concentrated hot spots** | May indicate manipulated regions |

> 💡 *If heatmap shows unusual concentration on specific areas (eyes, mouth, edges), the image may have localized manipulation.*

### 📊 Fourier Frequency Analysis

**Fast Fourier Transform (FFT)** converts the image to the frequency domain, revealing patterns invisible to the human eye.

| What it shows | How to interpret |
|---------------|------------------|
| **Center brightness** | Low-frequency components (overall structure) |
| **Edge patterns** | High-frequency details (textures, edges) |
| **Grid artifacts** | May indicate GAN-generated images |
| **Unusual symmetry** | Could suggest synthetic generation |

> 💡 *AI-generated images often have distinctive frequency fingerprints that differ from real photographs.*

---

## 📁 Project Structure

```
CNNs-Deepfake_Detection/
├── 📂 DF_Web/                    # Backend (FastAPI)
│   ├── api.py                    # REST API endpoints
│   ├── ai_logic.py               # AI engine & forensic methods
│   ├── model.py                  # CNN architecture definition
│   ├── custom_cnn_cifake.pth     # Trained model weights
│   ├── model_config.json         # Model configuration
│   └── requirements.txt          # Python dependencies
│
├── 📂 frontend/                  # Frontend (Next.js)
│   ├── app/
│   │   ├── page.jsx              # Main application page
│   │   ├── layout.js             # Root layout
│   │   └── globals.css           # Global styles + Tailwind
│   ├── package.json              # Node.js dependencies
│   ├── tailwind.config.js        # Tailwind configuration
│   └── next.config.js            # Next.js configuration
│
├── CNNs_Deepfake_Detection.ipynb # Training notebook
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## 🔌 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Detailed health status |
| `POST` | `/analyze` | Analyze image with forensic visualizations |
| `GET` | `/docs` | Swagger API documentation |

### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

### Example Response

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

## 🧠 Model Information

| Property | Value |
|----------|-------|
| **Architecture** | Custom CNN |
| **Training Dataset** | CIFAKE |
| **Accuracy** | 94% |
| **Input Size** | 224 × 224 pixels |
| **Output** | Binary (Real/Fake) |
| **Framework** | PyTorch |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **CIFAKE Dataset** for training data
- **PyTorch** team for the deep learning framework
- **FastAPI** for the excellent Python web framework
- **Vercel** for Next.js and frontend tooling

---

<div align="center">

**Built with ❤️ by HCMUTE Senior Design Team**

⭐ Star this repo if you find it useful! ⭐

</div>