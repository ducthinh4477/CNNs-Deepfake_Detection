# DeepScan - AI Deepfake Detection

**Tech Stack:** FastAPI (Backend) + Next.js (Frontend)

---

## 🚀 Quick Start

### Backend (FastAPI)
```bash
cd DF_Web
pip install -r requirements.txt
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev
```

---

## 📁 Project Structure

```
CNNs-Deepfake_Detection/
├── DF_Web/                   # FastAPI Backend
│   ├── api.py                # Main API entry point
│   ├── ai_logic.py           # AI/ML logic
│   ├── model.py              # MyNet architecture
│   ├── model2.py             # CombinedModel architecture
│   ├── models/
│   │   ├── config.json       # Model registry (add new models here)
│   │   ├── custom_cnn_cifake.pth
│   │   └── best_model.pth
│   ├── requirements.txt
│   └── vercel.json           # Vercel deployment config
│
└── frontend/                 # Next.js Frontend
    ├── app/
    │   ├── page.jsx          # Main UI with model selector
    │   ├── layout.js
    │   └── globals.css
    ├── package.json
    └── next.config.js
```

---

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/models` | GET | List all available models |
| `/models/{model_id}/select` | POST | Switch active model |
| `/models/current` | GET | Get current model info |
| `/analyze` | POST | Analyze image with current model |

---

## ➕ Adding New Models

### 1. Add model file
Copy your `.pth` file to `DF_Web/models/`:
```bash
cp your_model.pth DF_Web/models/
```

### 2. Register in config
Edit `DF_Web/models/config.json`:
```json
{
  "models": {
    "your_model_id": {
      "name": "Your Model Name",
      "description": "Model description",
      "file": "your_model.pth",
      "architecture": "MyNet",  // or "CombinedModel"
      "input_size": [224, 224],
      "accuracy": 95.0,
      "dataset": "Your Dataset",
      "version": "1.0",
      "color": "#FF6B6B"
    }
  }
}
```

### 3. Restart API server
The new model will automatically appear in the dropdown!

---

## 🌐 Deployment

### Backend (Vercel/Render/Railway)

**Note:** Vercel has 50MB function limit. For large models (>50MB):
- Use Render or Railway instead
- Or upload models to cloud storage (Google Drive/S3) and download at startup

**Vercel:**
```bash
cd DF_Web
vercel --prod
```

**Render:**
- Create Web Service
- Connect GitHub repo
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`

### Frontend (Vercel/Netlify/Render)

**Vercel:**
```bash
cd frontend
vercel --prod
```

**Netlify:**
```bash
cd frontend
npm run build
netlify deploy --prod --dir=.next
```

---

## 🔧 Environment Variables

### Backend (.env)
```env
FRONTEND_URL=https://your-frontend.vercel.app
MODEL_DOWNLOAD_URL=https://drive.google.com/...  # Optional: for model auto-download
```

### Frontend (.env.local)
```env
NEXT_PUBLIC_API_URL=https://your-backend.vercel.app
```

---

## 📊 Model Architectures

### MyNet (Custom CNN)
- Input: 224×224 RGB
- Architecture: 4 Conv blocks + FC layers
- Use for: CIFAKE dataset

### CombinedModel (EfficientNet + FFT)
- Input: 160×160 RGB
- Architecture: EfficientNet-B0 + Frequency Analysis
- Use for: DIRE dataset

---

## 🛠️ Troubleshooting

### Model dropdown not showing
✅ Fixed with z-index: 9999 and overflow-visible

### CORS errors
Update `FRONTEND_URL` in backend environment variables

### Model loading fails
Check file paths in `models/config.json`

---

## 👨‍💻 Author
**Nguyễn Đức Thịnh - 23110156**
HCMUTE Senior Design Team
