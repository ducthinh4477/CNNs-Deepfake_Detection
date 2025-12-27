"""
DeepScan FastAPI Backend
========================
REST API for DeepScan Deepfake Detection service.
Wraps the existing AI logic and exposes it via HTTP endpoints.

Author: HCMUTE Senior Design Team
Run: uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

import os
import io
import json
import base64
import uuid
import tempfile
import requests
from typing import Optional, List, Dict
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2

from ai_logic import DeepfakeAI


# =============================================================================
# MODEL CONFIGURATION - Central Model Registry
# =============================================================================

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODELS_CONFIG_PATH = os.path.join(MODELS_DIR, "config.json")

def load_models_config() -> Dict:
    """Load models configuration from config.json"""
    try:
        with open(MODELS_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load models config: {e}")
        # Return default config
        return {
            "default_model": "cnn_cifake",
            "models": {
                "cnn_cifake": {
                    "name": "CNN Custom (CIFAKE)",
                    "description": "Custom CNN for CIFAKE dataset",
                    "file": "custom_cnn_cifake.pth",
                    "architecture": "MyNet",
                    "input_size": [224, 224],
                    "accuracy": 92.0,
                    "dataset": "CIFAKE",
                    "version": "1.0",
                    "color": "#10B981"
                }
            }
        }

# Load config at module level
MODELS_CONFIG = load_models_config()


# =============================================================================
# MODEL AUTO-DOWNLOAD (for Cloud Deployment)
# =============================================================================

# Google Drive File IDs for each model
MODEL_DOWNLOAD_URLS = {
    "custom_cnn_cifake.pth": "https://drive.google.com/uc?export=download&id=1a0kszqbVthsrEnCSV8mzk4vrupSUmDeB&confirm=t",
    "best_model.pth": "https://drive.google.com/uc?export=download&id=16v-yryyAM-ynJ0qHNy4EeZFkFsniBhJX&confirm=t"
}

def download_model_if_not_exists(model_path: str) -> bool:
    """
    Check if model file exists. If not, download from configured URL.
    
    Returns:
        True if model exists or was downloaded successfully, False otherwise.
    """
    if os.path.exists(model_path):
        print(f"✅ Model file found: {model_path}")
        return True
    
    # Get model filename from path
    model_filename = os.path.basename(model_path)
    
    # Try environment variable first (MODEL_DOWNLOAD_URL_<filename>)
    env_var_name = f"MODEL_DOWNLOAD_URL_{model_filename.replace('.', '_').upper()}"
    model_url = os.environ.get(env_var_name)
    
    if not model_url:
        # Fallback to configured URLs
        model_url = MODEL_DOWNLOAD_URLS.get(model_filename)
        if model_url:
            print(f"📥 Using configured download URL for {model_filename}")
    
    if not model_url:
        print(f"❌ No download URL configured for {model_filename}")
        return False
    
    print(f"📥 Model not found locally. Downloading {model_filename}...")
    
    try:
        # For Google Drive links, convert to direct download URL
        if "drive.google.com" in model_url:
            # Extract file ID from Google Drive URL
            if "/file/d/" in model_url:
                file_id = model_url.split("/file/d/")[1].split("/")[0]
                model_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
        
        response = requests.get(model_url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Get total file size for progress
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"   Progress: {progress:.1f}%", end='\r')
        
        print(f"\n✅ Model downloaded successfully: {model_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download model: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error downloading model: {e}")
        # Clean up partial download
        if os.path.exists(model_path):
            os.remove(model_path)
        return False


# =============================================================================
# APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="DeepScan API",
    description="AI-powered Deepfake Detection REST API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# =============================================================================
# CORS Configuration - Secure origins for production
# =============================================================================

def get_allowed_origins() -> List[str]:
    """
    Get allowed CORS origins from environment variable.
    FRONTEND_URL can be a single URL or comma-separated list.
    Falls back to '*' for development/testing.
    """
    frontend_url = os.environ.get("FRONTEND_URL", "*")
    
    if frontend_url == "*":
        return ["*"]
    
    # Support comma-separated URLs
    origins = [url.strip() for url in frontend_url.split(",")]
    
    # Always include localhost for development
    dev_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    
    return list(set(origins + dev_origins))

allowed_origins = get_allowed_origins()
print(f"🌐 CORS allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# MODEL INITIALIZATION (Multi-model support)
# =============================================================================

# Cache loaded models to avoid reloading
loaded_models: Dict[str, DeepfakeAI] = {}
current_model_id: str = MODELS_CONFIG.get("default_model", "cnn_cifake")


def get_model_path(model_id: str) -> Optional[str]:
    """Get the full path to a model file"""
    model_info = MODELS_CONFIG.get("models", {}).get(model_id)
    if not model_info:
        return None
    return os.path.join(MODELS_DIR, model_info["file"])


def load_model(model_id: str) -> Optional[DeepfakeAI]:
    """Load a model by ID, using cache if available"""
    global loaded_models
    
    # Return cached model if available
    if model_id in loaded_models:
        print(f"📦 Using cached model: {model_id}")
        return loaded_models[model_id]
    
    model_info = MODELS_CONFIG.get("models", {}).get(model_id)
    if not model_info:
        print(f"❌ Model not found in config: {model_id}")
        return None
    
    model_path = get_model_path(model_id)
    if not model_path or not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        # Try to download if URL is configured
        if not download_model_if_not_exists(model_path):
            return None
    
    try:
        architecture = model_info.get("architecture", "MyNet")
        print(f"🔄 Loading model: {model_id} ({architecture})")
        ai_engine = DeepfakeAI(model_path=model_path, model_type=architecture)
        loaded_models[model_id] = ai_engine
        print(f"✅ Model '{model_id}' loaded successfully!")
        return ai_engine
    except Exception as e:
        print(f"❌ Failed to load model '{model_id}': {e}")
        return None


def get_current_model() -> Optional[DeepfakeAI]:
    """Get the currently selected model"""
    return load_model(current_model_id)


@app.on_event("startup")
async def startup_event():
    """Initialize default AI model on server startup"""
    global current_model_id
    print("🚀 Initializing DeepScan AI Engine...")
    print(f"📁 Models directory: {MODELS_DIR}")
    print(f"📋 Available models: {list(MODELS_CONFIG.get('models', {}).keys())}")
    
    # Load default model
    default_model = MODELS_CONFIG.get("default_model", "cnn_cifake")
    current_model_id = default_model
    
    model = load_model(default_model)
    if model:
        print(f"✅ Default model '{default_model}' ready on device: {model.device}")
    else:
        print("⚠️ Default model not available. API will run in degraded mode.")


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class AnalysisResult(BaseModel):
    """Response model for analysis endpoint"""
    filename: str
    is_real: bool
    label: int  # 0 = Fake, 1 = Real
    confidence_score: float
    risk_level: str
    trust_score: str
    timestamp: str
    model_used: str
    model_id: str  # Add model_id for frontend
    model_used: str
    # Hybrid Analysis Scores
    final_score: Optional[float] = None  # Weighted combination score
    heatmap_score: Optional[float] = None  # Heat intensity score (0-1)
    fourier_score: Optional[float] = None  # High-frequency artifact score (0-1)
    analysis_reason: Optional[str] = None  # Human-readable explanation
    # Forensic visualizations as base64-encoded PNG images
    heatmap: Optional[str] = None
    fourier: Optional[str] = None
    fourier_highpass: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    device: str
    version: str
    current_model: str


class ModelInfo(BaseModel):
    """Response model for model info"""
    id: str
    name: str
    description: str
    architecture: str
    accuracy: float
    dataset: str
    version: str
    color: str
    is_available: bool
    is_current: bool


class ModelsListResponse(BaseModel):
    """Response model for models list"""
    models: List[ModelInfo]
    current_model: str


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_risk_level(is_real: bool, confidence: float) -> str:
    """Calculate risk level based on prediction"""
    if is_real:
        return "Low"
    else:
        if confidence >= 0.9:
            return "Critical"
        elif confidence >= 0.75:
            return "High"
        else:
            return "Medium"


def calculate_trust_score(confidence: float) -> str:
    """Calculate trust score based on confidence"""
    if confidence >= 0.9:
        return "Very High"
    elif confidence >= 0.8:
        return "High"
    elif confidence >= 0.6:
        return "Medium"
    else:
        return "Low"


def numpy_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy array to base64 string for JSON response"""
    img = Image.fromarray(img_array.astype('uint8'))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# =============================================================================
# HYBRID ANALYSIS FUNCTIONS
# =============================================================================

def calculate_heatmap_score(heatmap_array: np.ndarray) -> float:
    """
    Calculate suspicion score from Grad-CAM heatmap.
    
    Analyzes the intensity of "hot" regions (red/yellow colors) in the heatmap.
    High intensity in localized areas suggests the model found suspicious regions.
    
    Args:
        heatmap_array: RGB numpy array of the heatmap overlay
        
    Returns:
        float: Score from 0-1 (higher = more suspicious regions detected)
    """
    try:
        # Convert to HSV to better isolate red/orange/yellow colors
        heatmap_hsv = cv2.cvtColor(heatmap_array, cv2.COLOR_RGB2HSV)
        
        # Red color has hue near 0 or 180 in HSV
        # Yellow-Orange is around hue 15-30
        # We look for "hot" colors: Red, Orange, Yellow
        
        # Mask for red (hue 0-10 and 170-180)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Mask for orange/yellow (hue 10-35)
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([35, 255, 255])
        
        # Create masks
        mask_red1 = cv2.inRange(heatmap_hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(heatmap_hsv, lower_red2, upper_red2)
        mask_orange = cv2.inRange(heatmap_hsv, lower_orange, upper_orange)
        
        # Combine masks
        hot_mask = mask_red1 | mask_red2 | mask_orange
        
        # Calculate the percentage of "hot" pixels
        total_pixels = hot_mask.size
        hot_pixels = np.count_nonzero(hot_mask)
        hot_ratio = hot_pixels / total_pixels
        
        # Also consider the intensity (Value channel) of hot regions
        hot_values = heatmap_hsv[:, :, 2][hot_mask > 0]
        avg_intensity = np.mean(hot_values) / 255.0 if len(hot_values) > 0 else 0
        
        # Combine ratio and intensity for final score
        # More hot pixels + higher intensity = higher suspicion score
        score = 0.6 * hot_ratio * 10 + 0.4 * avg_intensity  # Scale ratio for better range
        
        return min(1.0, max(0.0, score))  # Clamp to 0-1
        
    except Exception as e:
        print(f"⚠️ Heatmap score calculation failed: {e}")
        return 0.0


def calculate_fourier_score(fourier_highpass_array: np.ndarray) -> float:
    """
    Calculate artifact score from Fourier high-pass filtered image.
    
    High-frequency artifacts (common in deepfakes) appear as bright regions
    in the high-pass filtered image. Higher average brightness indicates
    more high-frequency noise/artifacts.
    
    Args:
        fourier_highpass_array: RGB numpy array of high-pass filtered image
        
    Returns:
        float: Score from 0-1 (higher = more high-frequency artifacts)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(fourier_highpass_array, cv2.COLOR_RGB2GRAY)
        
        # Calculate mean brightness (normalized to 0-1)
        mean_brightness = np.mean(gray) / 255.0
        
        # Calculate standard deviation (indicates artifact distribution)
        std_brightness = np.std(gray) / 255.0
        
        # Look at the percentage of very bright pixels (potential artifacts)
        bright_threshold = 150  # Pixels brighter than this are considered "artifact indicators"
        bright_pixels = np.count_nonzero(gray > bright_threshold)
        bright_ratio = bright_pixels / gray.size
        
        # Combine metrics for final score
        # High mean brightness + high variation + many bright spots = more artifacts
        score = 0.4 * mean_brightness + 0.3 * std_brightness * 4 + 0.3 * bright_ratio * 5
        
        return min(1.0, max(0.0, score))  # Clamp to 0-1
        
    except Exception as e:
        print(f"⚠️ Fourier score calculation failed: {e}")
        return 0.0


def calculate_hybrid_analysis(
    cnn_confidence: float,
    cnn_is_real: bool,
    heatmap_score: float,
    fourier_score: float
) -> dict:
    """
    Perform hybrid analysis combining CNN prediction with forensic scores.
    
    Formula: Final_Score = 0.4 * CNN_Conf + 0.3 * (1 - Heatmap_Score) + 0.3 * (1 - Fourier_Score)
    
    Veto Logic: If heatmap_score > 0.6 OR fourier_score > 0.7, flag as suspicious
    regardless of CNN prediction.
    
    Args:
        cnn_confidence: CNN model confidence score (0-1)
        cnn_is_real: CNN prediction (True = Real, False = Fake)
        heatmap_score: Heatmap suspicion score (0-1, higher = more suspicious)
        fourier_score: Fourier artifact score (0-1, higher = more artifacts)
        
    Returns:
        dict with final_score, adjusted_is_real, risk_level, analysis_reason
    """
    # Veto thresholds
    HEATMAP_VETO_THRESHOLD = 0.55  # If heatmap shows >55% suspicious regions
    FOURIER_VETO_THRESHOLD = 0.60  # If fourier shows >60% artifacts
    
    # Convert scores: Higher forensic score = less likely to be real
    # So we invert them for the "realness" calculation
    heatmap_realness = 1.0 - heatmap_score
    fourier_realness = 1.0 - fourier_score
    
    # CNN confidence for "Real" class
    cnn_real_conf = cnn_confidence if cnn_is_real else (1.0 - cnn_confidence)
    
    # Weighted final score (higher = more likely Real)
    final_score = (
        0.4 * cnn_real_conf +
        0.3 * heatmap_realness +
        0.3 * fourier_realness
    )
    
    # Initialize analysis reason components
    reasons = []
    veto_triggered = False
    
    # Check Veto conditions
    if heatmap_score >= HEATMAP_VETO_THRESHOLD:
        veto_triggered = True
        reasons.append(f"Heatmap phát hiện vùng nghi ngờ cao ({heatmap_score*100:.1f}%)")
    
    if fourier_score >= FOURIER_VETO_THRESHOLD:
        veto_triggered = True
        reasons.append(f"Fourier phát hiện nhiều artifact tần số cao ({fourier_score*100:.1f}%)")
    
    # Determine final classification
    if veto_triggered:
        adjusted_is_real = False
        risk_level = "High" if (heatmap_score >= 0.7 or fourier_score >= 0.75) else "Medium"
        
        if cnn_is_real:
            reasons.insert(0, "CNN dự đoán ẢNH THẬT")
            reasons.append("⚠️ Kết luận: NGHI NGỜ GIẢ MẠO (forensic analysis phủ quyết)")
        else:
            reasons.insert(0, "CNN dự đoán ẢNH GIẢ")
            reasons.append("✓ Forensic analysis xác nhận nghi ngờ")
    else:
        # No veto - use final_score to determine
        adjusted_is_real = final_score >= 0.5
        
        if adjusted_is_real:
            if final_score >= 0.8:
                risk_level = "Low"
                reasons.append("✓ Ảnh có độ tin cậy cao - không phát hiện dấu hiệu giả mạo")
            elif final_score >= 0.6:
                risk_level = "Low"
                reasons.append("✓ Ảnh có vẻ thật - một số chỉ số cần lưu ý nhưng không đáng ngờ")
            else:
                risk_level = "Medium"
                reasons.append("⚠️ Ảnh ở ranh giới - cần kiểm tra thêm để xác nhận")
        else:
            if final_score <= 0.3:
                risk_level = "Critical"
                reasons.append("❌ Nhiều dấu hiệu cho thấy ảnh đã bị chỉnh sửa hoặc tạo bởi AI")
            elif final_score <= 0.4:
                risk_level = "High"
                reasons.append("⚠️ Phát hiện dấu hiệu đáng ngờ - có thể là deepfake")
            else:
                risk_level = "Medium"
                reasons.append("⚠️ Một số chỉ số bất thường - cần xem xét kỹ hơn")
    
    # Add score breakdown
    score_breakdown = f"[CNN: {cnn_real_conf*100:.1f}% | Heatmap: {heatmap_realness*100:.1f}% | Fourier: {fourier_realness*100:.1f}%]"
    reasons.append(score_breakdown)
    
    return {
        "final_score": round(final_score, 4),
        "adjusted_is_real": adjusted_is_real,
        "risk_level": risk_level,
        "analysis_reason": " | ".join(reasons)
    }


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    ai_engine = get_current_model()
    model_info = MODELS_CONFIG.get("models", {}).get(current_model_id, {})
    return HealthResponse(
        status="online",
        model_loaded=ai_engine is not None,
        device=str(ai_engine.device) if ai_engine else "N/A",
        version="2.0.0",
        current_model=model_info.get("name", "Unknown")
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    ai_engine = get_current_model()
    model_info = MODELS_CONFIG.get("models", {}).get(current_model_id, {})
    return HealthResponse(
        status="healthy" if ai_engine else "degraded",
        model_loaded=ai_engine is not None,
        device=str(ai_engine.device) if ai_engine else "N/A",
        version="2.0.0",
        current_model=model_info.get("name", "Unknown")
    )


# =============================================================================
# MODEL MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/models", response_model=ModelsListResponse)
async def list_models():
    """
    List all available models.
    
    Returns:
        List of models with their info and availability status
    """
    models_list = []
    
    for model_id, info in MODELS_CONFIG.get("models", {}).items():
        model_path = get_model_path(model_id)
        is_available = model_path and os.path.exists(model_path)
        
        models_list.append(ModelInfo(
            id=model_id,
            name=info.get("name", model_id),
            description=info.get("description", ""),
            architecture=info.get("architecture", "Unknown"),
            accuracy=info.get("accuracy", 0.0),
            dataset=info.get("dataset", "Unknown"),
            version=info.get("version", "1.0"),
            color=info.get("color", "#3B82F6"),
            is_available=is_available,
            is_current=(model_id == current_model_id)
        ))
    
    return ModelsListResponse(
        models=models_list,
        current_model=current_model_id
    )


@app.post("/models/{model_id}/select")
async def select_model(model_id: str):
    """
    Select a model as the current active model.
    
    Args:
        model_id: The ID of the model to select (e.g., 'cnn_cifake', 'combined_model')
    
    Returns:
        Success message with selected model info
    """
    global current_model_id
    
    # Check if model exists in config
    if model_id not in MODELS_CONFIG.get("models", {}):
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. Available: {list(MODELS_CONFIG.get('models', {}).keys())}"
        )
    
    # Try to load the model
    model = load_model(model_id)
    if not model:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_id}' could not be loaded. Check if the model file exists."
        )
    
    # Update current model
    current_model_id = model_id
    model_info = MODELS_CONFIG.get("models", {}).get(model_id, {})
    
    print(f"✅ Switched to model: {model_id}")
    
    return JSONResponse({
        "success": True,
        "message": f"Model switched to '{model_info.get('name', model_id)}'",
        "current_model": {
            "id": model_id,
            "name": model_info.get("name", model_id),
            "accuracy": model_info.get("accuracy", 0.0),
            "architecture": model_info.get("architecture", "Unknown")
        }
    })


@app.get("/models/current")
async def get_current_model_info():
    """Get info about the currently selected model"""
    model_info = MODELS_CONFIG.get("models", {}).get(current_model_id, {})
    ai_engine = get_current_model()
    
    return JSONResponse({
        "id": current_model_id,
        "name": model_info.get("name", "Unknown"),
        "description": model_info.get("description", ""),
        "accuracy": model_info.get("accuracy", 0.0),
        "dataset": model_info.get("dataset", "Unknown"),
        "architecture": model_info.get("architecture", "Unknown"),
        "version": model_info.get("version", "1.0"),
        "color": model_info.get("color", "#3B82F6"),
        "is_loaded": ai_engine is not None,
        "device": str(ai_engine.device) if ai_engine else "N/A"
    })


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_image(
    file: UploadFile = File(...),
    model_id: Optional[str] = Query(None, description="Model ID to use for analysis. If not provided, uses current model.")
):
    """
    Analyze an image for deepfake detection.
    
    Args:
        file: Image file (JPG, PNG, WEBP)
        model_id: Optional model ID to use (defaults to current model)
    
    Returns:
        AnalysisResult with prediction details
    """
    global current_model_id
    
    # Use specified model or current model
    use_model_id = model_id if model_id else current_model_id
    ai_engine = load_model(use_model_id)
    model_info = MODELS_CONFIG.get("models", {}).get(use_model_id, {})
    
    # Validate AI engine is loaded
    if ai_engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"AI model '{use_model_id}' not loaded. Please check if model file exists."
        )
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: JPG, PNG, WEBP"
        )
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Run CNN prediction
        label, confidence, _ = ai_engine.predict(image)
        cnn_is_real = label == 1
        
        # Generate forensic visualizations
        heatmap_array = ai_engine.generate_heatmap(image)
        fourier_array = ai_engine.generate_fourier_analysis(image)
        fourier_highpass_array = ai_engine.generate_fourier_highpass(image)
        
        # ========== HYBRID ANALYSIS ==========
        # Calculate forensic scores
        heatmap_score = calculate_heatmap_score(heatmap_array) if heatmap_array is not None else 0.0
        fourier_score = calculate_fourier_score(fourier_highpass_array) if fourier_highpass_array is not None else 0.0
        
        # Perform hybrid analysis with veto logic
        hybrid_result = calculate_hybrid_analysis(
            cnn_confidence=confidence,
            cnn_is_real=cnn_is_real,
            heatmap_score=heatmap_score,
            fourier_score=fourier_score
        )
        
        # Use hybrid results for final determination
        final_is_real = hybrid_result["adjusted_is_real"]
        final_score = hybrid_result["final_score"]
        risk_level = hybrid_result["risk_level"]
        analysis_reason = hybrid_result["analysis_reason"]
        
        # Calculate trust score based on final score
        trust_score = calculate_trust_score(final_score)
        
        # Convert numpy arrays to base64 strings
        heatmap_base64 = numpy_to_base64(heatmap_array) if heatmap_array is not None else None
        fourier_base64 = numpy_to_base64(fourier_array) if fourier_array is not None else None
        fourier_highpass_base64 = numpy_to_base64(fourier_highpass_array) if fourier_highpass_array is not None else None
        
        return AnalysisResult(
            filename=file.filename or "unknown",
            is_real=final_is_real,
            label=1 if final_is_real else 0,
            confidence_score=round(final_score, 4),  # Use final hybrid score
            risk_level=risk_level,
            trust_score=trust_score,
            timestamp=datetime.now().isoformat(),
            model_used=model_info.get("name", "Unknown Model"),
            model_id=use_model_id,
            # Hybrid analysis details
            final_score=round(final_score, 4),
            heatmap_score=round(heatmap_score, 4),
            fourier_score=round(fourier_score, 4),
            analysis_reason=analysis_reason,
            # Visualizations
            heatmap=heatmap_base64,
            fourier=fourier_base64,
            fourier_highpass=fourier_highpass_base64
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/analyze/full")
async def analyze_image_full(file: UploadFile = File(...)):
    """
    Full analysis with heatmap and Fourier visualizations.
    Returns base64-encoded images for frontend display.
    
    Args:
        file: Image file (JPG, PNG, WEBP)
    
    Returns:
        JSON with prediction + visualization images as base64
    """
    ai_engine = get_current_model()
    model_info = MODELS_CONFIG.get("models", {}).get(current_model_id, {})
    
    if ai_engine is None:
        raise HTTPException(
            status_code=503,
            detail="AI model not loaded."
        )
    
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}"
        )
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Run prediction
        label, confidence, _ = ai_engine.predict(image)
        is_real = label == 1
        
        # Generate visualizations
        heatmap = ai_engine.generate_heatmap(image)
        fourier_magnitude = ai_engine.generate_fourier_analysis(image)
        fourier_highpass = ai_engine.generate_fourier_highpass(image)
        
        return JSONResponse({
            "prediction": {
                "filename": file.filename or "unknown",
                "is_real": is_real,
                "label": label,
                "confidence_score": round(confidence, 4),
                "risk_level": calculate_risk_level(is_real, confidence),
                "trust_score": calculate_trust_score(confidence),
                "timestamp": datetime.now().isoformat(),
                "model_used": model_info.get("name", "Unknown"),
                "model_id": current_model_id
            },
            "visualizations": {
                "heatmap": numpy_to_base64(heatmap),
                "fourier_magnitude": numpy_to_base64(fourier_magnitude),
                "fourier_highpass": numpy_to_base64(fourier_highpass)
            }
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Full analysis failed: {str(e)}"
        )


@app.post("/heatmap")
async def generate_heatmap(file: UploadFile = File(...)):
    """Generate heatmap visualization only"""
    ai_engine = get_current_model()
    if ai_engine is None:
        raise HTTPException(status_code=503, detail="AI model not loaded.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        heatmap = ai_engine.generate_heatmap(image)
        
        return JSONResponse({
            "image": numpy_to_base64(heatmap),
            "type": "heatmap"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fourier")
async def generate_fourier(file: UploadFile = File(...), mode: str = "magnitude"):
    """
    Generate Fourier analysis visualization.
    
    Args:
        file: Image file
        mode: 'magnitude' or 'highpass'
    """
    ai_engine = get_current_model()
    if ai_engine is None:
        raise HTTPException(status_code=503, detail="AI model not loaded.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        if mode == "highpass":
            result = ai_engine.generate_fourier_highpass(image)
        else:
            result = ai_engine.generate_fourier_analysis(image)
        
        return JSONResponse({
            "image": numpy_to_base64(result),
            "type": f"fourier_{mode}"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
