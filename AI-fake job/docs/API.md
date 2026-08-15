# API Documentation

This document provides comprehensive documentation for the AI Fake Job Detector API endpoints.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com/api`

## Authentication

Currently, the API does not require authentication. For production use, implement authentication using JWT tokens or API keys.

## Endpoints

### Health Check

Check if the API is running and models are loaded.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "version": "1.0.0"
}
```

### Predict Job Posting

Analyze a job posting to determine if it's fake or real.

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "job_description": "Senior Software Engineer position requiring 5+ years Python experience. Competitive salary and benefits.",
  "company_name": "TechCorp",
  "salary": "$80,000 - $95,000",
  "job_title": "Senior Software Engineer",
  "has_company_logo": true,
  "company_profile": "TechCorp is a leading technology company with 500 employees worldwide."
}
```

**Parameters:**
- `job_description` (string, required): Full job description text
- `company_name` (string, optional): Company name
- `salary` (string, optional): Salary information
- `job_title` (string, optional): Job title
- `has_company_logo` (boolean, optional): Whether company has logo
- `company_profile` (string, optional): Company profile text

**Response:**
```json
{
  "prediction": "real",
  "confidence": 0.92,
  "fraud_probability": 0.08,
  "risk_level": "Low",
  "explanation": "This job posting is classified as REAL with 92.0% confidence. The company profile shows typical legitimacy indicators.",
  "suspicious_phrases": [],
  "highlighted_text": null,
  "components": {
    "roberta": {
      "prediction": 0,
      "probability": 0.08,
      "is_fake": false,
      "confidence": 0.92
    },
    "salary_anomaly": 0.15,
    "company_legitimacy": 0.85,
    "ocr_confidence": 0.5,
    "heuristic_score": 0.1
  }
}
```

**Response Fields:**
- `prediction`: "fake" or "real"
- `confidence`: Confidence score (0-1)
- `fraud_probability`: Probability of being fake (0-1)
- `risk_level`: "High", "Medium", or "Low"
- `explanation`: Human-readable explanation
- `suspicious_phrases`: List of suspicious phrases detected
- `highlighted_text`: Text with suspicious phrases highlighted (HTML)
- `components`: Individual component scores

### Image Prediction

Analyze a job posting image using OCR + prediction.

**Endpoint:** `POST /image-predict`

**Request:** Multipart form data with file upload

**Parameters:**
- `file` (file, required): Image file (PNG, JPG, JPEG, PDF)

**Response:** Same format as `/predict` endpoint

**Example using curl:**
```bash
curl -X POST http://localhost:8000/image-predict \
  -F "file=@job_posting.png"
```

### Get Prediction History

Retrieve recent prediction history.

**Endpoint:** `GET /history`

**Query Parameters:**
- `limit` (integer, optional): Number of recent predictions to return (default: 10)

**Response:**
```json
{
  "total_predictions": 25,
  "recent_predictions": [
    {
      "timestamp": "2024-01-15T10:30:00",
      "job_description": "Senior Software Engineer...",
      "prediction": "real",
      "confidence": 0.92,
      "fraud_probability": 0.08
    }
  ]
}
```

### Root Endpoint

Get API information and available endpoints.

**Endpoint:** `GET /`

**Response:**
```json
{
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
```

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Invalid file type. Allowed: png, jpg, jpeg, pdf"
}
```

### 503 Service Unavailable

```json
{
  "detail": "Models not loaded"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Prediction failed: error message"
}
```

## Rate Limiting

Currently, there are no rate limits. For production use, implement rate limiting to prevent abuse.

## Example Usage

### Python

```python
import requests

# Text prediction
response = requests.post('http://localhost:8000/predict', json={
    'job_description': 'URGENT! Earn $5000 weekly working from home. No experience needed.',
    'company_name': 'QuickCash Inc',
    'salary': '$5000/week'
})

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")

# Image prediction
with open('job_posting.png', 'rb') as f:
    response = requests.post('http://localhost:8000/image-predict', files={'file': f})
    
result = response.json()
print(f"Prediction: {result['prediction']}")
```

### JavaScript

```javascript
// Text prediction
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    job_description: 'Senior Software Engineer position...',
    company_name: 'TechCorp',
    salary: '$80,000 - $95,000'
  })
});

const result = await response.json();
console.log('Prediction:', result.prediction);
console.log('Confidence:', (result.confidence * 100).toFixed(1) + '%');

// Image prediction
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/image-predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log('Prediction:', result.prediction);
```

### cURL

```bash
# Text prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "Senior Software Engineer position...",
    "company_name": "TechCorp",
    "salary": "$80,000 - $95,000"
  }'

# Image prediction
curl -X POST http://localhost:8000/image-predict \
  -F "file=@job_posting.png"

# Health check
curl http://localhost:8000/health
```

## Interactive API Documentation

When running the API, access the interactive documentation at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## WebSocket Support (Future)

Real-time prediction streaming will be supported via WebSocket in future versions.

## Webhook Support (Future)

Webhook notifications for batch processing results will be supported in future versions.
