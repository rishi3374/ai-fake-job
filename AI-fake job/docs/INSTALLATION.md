# Installation Guide

This guide provides step-by-step instructions for installing and setting up the AI Fake Job Detector system.

## Prerequisites

- Python 3.12 or higher
- Node.js 18 or higher
- Docker and Docker Compose (optional, for containerized deployment)
- Git

## System Requirements

- **Minimum**: 4GB RAM, 10GB disk space
- **Recommended**: 8GB RAM, 20GB disk space
- **GPU**: Optional, for RoBERTa training (CUDA-compatible GPU recommended)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-fake-job-detector.git
cd ai-fake-job-detector
```

### 2. Backend Setup

#### Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Install Optional Dependencies

Some features require additional packages:

```bash
# For RoBERTa model training
pip install torch transformers

# For OCR functionality
pip install easyocr

# For SHAP explainability
pip install shap

# For database functionality
pip install sqlalchemy psycopg2-binary

# For FastAPI backend
pip install fastapi uvicorn pydantic pydantic-settings
```

#### Set Up Environment Variables

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Required variables:
# - DATABASE_URL: Database connection string
# - API_HOST: API server host
# - API_PORT: API server port
# - SECRET_KEY: Secret key for security
```

### 3. Frontend Setup

#### Install Node Dependencies

```bash
cd frontend
npm install
```

#### Configure Frontend

```bash
# Create environment file
echo "VITE_API_URL=http://localhost:8000" > .env
```

### 4. Database Setup

#### Using SQLite (Default)

SQLite is used by default and requires no additional setup.

#### Using PostgreSQL

```bash
# Install PostgreSQL
# Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib
# macOS: brew install postgresql
# Windows: Download from postgresql.org

# Create database
createdb fake_job_detector

# Update DATABASE_URL in .env
# DATABASE_URL=postgresql://user:password@localhost:5432/fake_job_detector
```

#### Initialize Database

```bash
# Run database initialization
python3 backend/database/connection.py
```

### 5. Data Preparation

#### Download Dataset

```bash
# Using Kaggle API (requires Kaggle credentials)
python3 scripts/prepare_dataset.py

# Or manually download from Kaggle and place in data/raw/fake_job_postings.csv
```

#### Process Data

```bash
# Clean and preprocess data
python3 scripts/clean_dataset.py

# This will create:
# - data/processed/cleaned_dataset.csv
# - Various feature files
```

### 6. Model Training

#### Train Baseline Models

```bash
python3 models/baseline/baseline_models.py
```

#### Train RoBERTa Model (Optional)

```bash
python3 models/roberta/roberta_trainer.py
```

#### Train Hybrid Model

```bash
python3 models/hybrid/hybrid_model.py
```

### 7. Start the Application

#### Development Mode

**Backend:**
```bash
python3 backend/api/main.py
```

**Frontend:**
```bash
cd frontend
npm run dev
```

#### Production Mode with Docker

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Verification

### Test Backend

```bash
# Health check
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "Senior Software Engineer position with Python experience",
    "company_name": "TechCorp",
    "salary": "$80,000 - $95,000"
  }'
```

### Test Frontend

Open your browser and navigate to `http://localhost:3000`

## Troubleshooting

### Common Issues

**Issue: ModuleNotFoundError**
```bash
# Solution: Install missing dependencies
pip install <missing_module>
```

**Issue: Database connection error**
```bash
# Solution: Check DATABASE_URL in .env
# Ensure PostgreSQL is running if using PostgreSQL
```

**Issue: GPU not detected for PyTorch**
```bash
# Solution: Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Issue: Frontend build fails**
```bash
# Solution: Clear node_modules and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Getting Help

- Check the [Documentation](README.md)
- Review [API Documentation](API.md)
- Open an issue on GitHub

## Next Steps

- Read the [User Manual](USER_MANUAL.md)
- Explore the [API Documentation](API.md)
- Review the [Developer Guide](DEVELOPER.md)

## Security Notes

- Change default SECRET_KEY in production
- Use strong database passwords
- Enable HTTPS in production
- Implement authentication for production use
- Regular security updates for dependencies
