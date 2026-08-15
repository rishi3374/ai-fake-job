"""
Configuration Module
Handles application configuration and environment variables
"""

from pydantic_settings import BaseSettings
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    DEBUG: bool = True
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./fake_job_detector.db"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "fake_job_detector"
    DB_USER: str = "user"
    DB_PASSWORD: str = "password"
    
    # Model Configuration
    MODEL_DIR: str = "data/models"
    ROBERTA_MODEL_NAME: str = "roberta-base"
    MAX_LENGTH: int = 512
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 2e-5
    EPOCHS: int = 3
    WARMUP_STEPS: int = 500
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "fake_job_detection"
    
    # OCR Configuration
    OCR_MODEL_PATH: str = "ocr/models"
    OCR_CONFIDENCE_THRESHOLD: float = 0.7
    
    # Company Verification
    VERIFICATION_TIMEOUT: int = 10
    LINKEDIN_API_KEY: Optional[str] = None
    
    # Security
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 10485760  # 10MB
    ALLOWED_EXTENSIONS: list = ["png", "jpg", "jpeg", "pdf"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Configure logging
def setup_logging():
    """Setup application logging"""
    import os
    from pathlib import Path
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(settings.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging configured at {settings.LOG_LEVEL} level")
