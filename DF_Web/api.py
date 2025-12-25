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
import base64
import uuid
import tempfile
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np

from ai_logic import DeepfakeAI


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

# CORS Configuration - Allow requests from Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# MODEL INITIALIZATION (Singleton - loaded once at startup)
# =============================================================================

MODEL_PATH = "custom_cnn_cifake.pth"
ai_engine: Optional[DeepfakeAI] = None


@app.on_event("startup")
async def startup_event():
    """Initialize AI model on server startup"""
    global ai_engine
    print("🚀 Initializing DeepScan AI Engine...")
    try:
        ai_engine = DeepfakeAI(model_path=MODEL_PATH)
        print(f"Model loaded successfully on device: {ai_engine.device}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        ai_engine = None


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
# API ENDPOINTS
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="online",
        model_loaded=ai_engine is not None,
        device=str(ai_engine.device) if ai_engine else "N/A",
        version="2.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy" if ai_engine else "degraded",
        model_loaded=ai_engine is not None,
        device=str(ai_engine.device) if ai_engine else "N/A",
        version="2.0.0"
    )


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an image for deepfake detection.
    
    Args:
        file: Image file (JPG, PNG, WEBP)
    
    Returns:
        AnalysisResult with prediction details
    """
    # Validate AI engine is loaded
    if ai_engine is None:
        raise HTTPException(
            status_code=503,
            detail="AI model not loaded. Please restart the server."
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
        
        # Run prediction
        label, confidence, _ = ai_engine.predict(image)
        
        # Calculate metrics
        is_real = label == 1
        risk_level = calculate_risk_level(is_real, confidence)
        trust_score = calculate_trust_score(confidence)
        
        # Generate forensic visualizations
        heatmap_array = ai_engine.generate_heatmap(image)
        fourier_array = ai_engine.generate_fourier_analysis(image)
        fourier_highpass_array = ai_engine.generate_fourier_highpass(image)
        
        # Convert numpy arrays to base64 strings
        heatmap_base64 = numpy_to_base64(heatmap_array) if heatmap_array is not None else None
        fourier_base64 = numpy_to_base64(fourier_array) if fourier_array is not None else None
        fourier_highpass_base64 = numpy_to_base64(fourier_highpass_array) if fourier_highpass_array is not None else None
        
        return AnalysisResult(
            filename=file.filename or "unknown",
            is_real=is_real,
            label=label,
            confidence_score=round(confidence, 4),
            risk_level=risk_level,
            trust_score=trust_score,
            timestamp=datetime.now().isoformat(),
            model_used="CNN Custom (CIFAKE)",
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
                "model_used": "CNN Custom (CIFAKE)"
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
