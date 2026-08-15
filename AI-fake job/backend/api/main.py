"""
FastAPI Main Application
Main API endpoints for fake job detection
"""

import logging
from pathlib import Path
import sys
import os
from typing import Optional, Dict, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Optional FastAPI imports
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Optional imports
try:
    from backend.core.config import settings, setup_logging
except ImportError:
    # Fallback configuration
    class Settings:
        MODEL_DIR = "data/models"
        MAX_UPLOAD_SIZE = 10485760
    settings = Settings()
    def setup_logging():
        logging.basicConfig(level=logging.INFO)

try:
    from models.hybrid.hybrid_model import HybridModel
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

try:
    from explainability.models.shap_explainer import SHAPExplainer
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app if available
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="AI Fake Job Detector API",
        description="API for detecting fake job postings using explainable multimodal NLP",
        version="1.0.0"
    )
else:
    app = None
    logger.warning("FastAPI not available, API endpoints will not be functional")

# Add CORS middleware if FastAPI is available
if FASTAPI_AVAILABLE and app:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Initialize models
hybrid_model = None
shap_explainer = None

# In-memory storage for predictions (in production, use database)
prediction_history = []


# Pydantic models for request/response (only if FastAPI available)
if FASTAPI_AVAILABLE:
    class PredictionRequest(BaseModel):
        job_description: str = Field(..., description="Job description text")
        company_name: Optional[str] = Field(None, description="Company name")
        salary: Optional[str] = Field(None, description="Salary information")
        job_title: Optional[str] = Field(None, description="Job title")
        has_company_logo: Optional[bool] = Field(False, description="Whether company has logo")
        company_profile: Optional[str] = Field(None, description="Company profile text")

    class PredictionResponse(BaseModel):
        prediction: str = Field(..., description="Prediction: 'fake' or 'real'")
        confidence: float = Field(..., description="Confidence score (0-1)")
        fraud_probability: float = Field(..., description="Probability of being fake (0-1)")
        risk_level: str = Field(..., description="Risk level: 'High', 'Medium', or 'Low'")
        explanation: Optional[str] = Field(None, description="Explanation of prediction")
        suspicious_phrases: List[str] = Field(default_factory=list, description="Suspicious phrases detected")
        highlighted_text: Optional[str] = Field(None, description="Text with suspicious phrases highlighted")
        components: Dict = Field(default_factory=dict, description="Component scores")

    class HealthResponse(BaseModel):
        status: str = Field(..., description="API status")
        models_loaded: bool = Field(..., description="Whether models are loaded")
        version: str = Field(..., description="API version")

    class HistoryResponse(BaseModel):
        total_predictions: int = Field(..., description="Total number of predictions")
        recent_predictions: List[Dict] = Field(default_factory=list, description="Recent predictions")


# Define endpoints only if FastAPI is available
if FASTAPI_AVAILABLE and app:
    @app.on_event("startup")
    async def startup_event():
        """Initialize models on startup"""
        global hybrid_model, shap_explainer
        
        logger.info("Starting up AI Fake Job Detector API...")
        
        try:
            # Initialize hybrid model
            if HYBRID_AVAILABLE:
                hybrid_model = HybridModel(model_dir=settings.MODEL_DIR)
                logger.info("Hybrid model loaded successfully")
            
            # Initialize SHAP explainer
            if SHAP_AVAILABLE:
                shap_explainer = SHAPExplainer(model_dir=settings.MODEL_DIR)
                shap_explainer.initialize_explainer()
                logger.info("SHAP explainer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.warning("API will start with limited functionality")


    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        return HealthResponse(
            status="healthy",
            models_loaded=hybrid_model is not None,
            version="1.0.0"
        )


    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """
        Predict whether a job posting is fake or real
        
        Args:
            request: Prediction request with job details
            
        Returns:
            Prediction response with analysis
        """
        if hybrid_model is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        try:
            # Prepare company data
            company_data = {
                'has_company_logo': 1 if request.has_company_logo else 0,
                'company_profile': request.company_profile or '',
                'email': '',  # Could extract from description
                'has_questions': 0  # Could be added as parameter
            }
            
            # Make prediction
            result = hybrid_model.predict(
                text=request.job_description,
                salary_str=request.salary or "",
                company_data=company_data,
                job_title=request.job_title or ""
            )
            
            # Generate explanation
            explanation_text = None
            suspicious_phrases = []
            highlighted_text = None
            
            if shap_explainer:
                try:
                    shap_result = shap_explainer.explain_with_shap(request.job_description)
                    suspicious_phrases = shap_result.get('top_suspicious_phrases', [])
                    highlighted_text = shap_explainer.highlight_suspicious_phrases(request.job_description, shap_result)
                    explanation_text = hybrid_model.explain_prediction(result)
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {e}")
                    explanation_text = hybrid_model.explain_prediction(result)
            else:
                explanation_text = hybrid_model.explain_prediction(result)
            
            # Store prediction in history
            import pandas as pd
            prediction_entry = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'job_description': request.job_description[:200],  # Truncate for storage
                'prediction': 'fake' if result['is_fake'] else 'real',
                'confidence': result['confidence'],
                'fraud_probability': result['fraud_probability']
            }
            prediction_history.append(prediction_entry)
            
            # Keep only last 100 predictions
            if len(prediction_history) > 100:
                prediction_history.pop(0)
            
            return PredictionResponse(
                prediction='fake' if result['is_fake'] else 'real',
                confidence=result['confidence'],
                fraud_probability=result['fraud_probability'],
                risk_level=result['risk_level'],
                explanation=explanation_text,
                suspicious_phrases=suspicious_phrases,
                highlighted_text=highlighted_text,
                components=result['components']
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


    @app.post("/image-predict", response_model=PredictionResponse)
    async def image_predict(file: UploadFile = File(...)):
        """
        Predict from uploaded job posting image (OCR + prediction)
        
        Args:
            file: Uploaded image file
            
        Returns:
            Prediction response with analysis
        """
        if hybrid_model is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Check file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
            raise HTTPException(status_code=400, detail="Invalid file type. Allowed: png, jpg, jpeg, pdf")
        
        # Check file size
        file_size = 0
        for chunk in file.file:
            file_size += len(chunk)
            if file_size > settings.MAX_UPLOAD_SIZE:
                raise HTTPException(status_code=400, detail="File too large")
        
        try:
            # Reset file pointer
            await file.seek(0)
            
            # For now, return a placeholder response
            # In production, this would use OCR to extract text
            # TODO: Implement OCR pipeline
            
            return PredictionResponse(
                prediction='real',
                confidence=0.5,
                fraud_probability=0.5,
                risk_level='Medium',
                explanation="OCR processing not yet implemented. Please use text prediction endpoint.",
                suspicious_phrases=[],
                highlighted_text=None,
                components={}
            )
            
        except Exception as e:
            logger.error(f"Image prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Image prediction failed: {str(e)}")


    @app.get("/history", response_model=HistoryResponse)
    async def get_history(limit: int = 10):
        """
        Get prediction history
        
        Args:
            limit: Number of recent predictions to return
            
        Returns:
            Prediction history
        """
        recent_predictions = prediction_history[-limit:] if prediction_history else []
        
        return HistoryResponse(
            total_predictions=len(prediction_history),
            recent_predictions=recent_predictions
        )


    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "AI Fake Job Detector API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
                "image_predict": "/image-predict",
                "history": "/history",
                "docs": "/docs"
            }
        }


if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        import uvicorn
        
        uvicorn.run(
            "backend.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )
    else:
        logger.warning("FastAPI not available. Please install dependencies: pip install fastapi uvicorn pydantic")
        logger.info("Testing prediction functionality directly...")
        
        # Test prediction functionality
        if HYBRID_AVAILABLE:
            try:
                hybrid_model = HybridModel(model_dir=settings.MODEL_DIR)
                test_text = "URGENT! Earn $5000 weekly working from home. No experience needed."
                result = hybrid_model.predict(text=test_text)
                print(f"Test prediction: {result}")
            except Exception as e:
                logger.error(f"Test prediction failed: {e}")
        else:
            logger.warning("Hybrid model not available")
