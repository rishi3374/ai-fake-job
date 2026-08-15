# Developer Guide

This guide provides comprehensive information for developers working on the AI Fake Job Detector project.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Development Setup](#development-setup)
3. [Code Architecture](#code-architecture)
4. [Testing](#testing)
5. [Model Training](#model-training)
6. [API Development](#api-development)
7. [Frontend Development](#frontend-development)
8. [Deployment](#deployment)
9. [Contributing](#contributing)

## Project Structure

```
ai-fake-job-detector/
├── backend/
│   ├── api/
│   │   └── main.py              # FastAPI application
│   ├── core/
│   │   └── config.py            # Configuration management
│   └── database/
│       ├── models.py            # SQLAlchemy models
│       └── connection.py        # Database connection
├── models/
│   ├── baseline/
│   │   └── baseline_models.py   # Traditional ML models
│   ├── roberta/
│   │   └── roberta_trainer.py   # RoBERTa fine-tuning
│   └── hybrid/
│       └── hybrid_model.py      # Hybrid ensemble model
├── preprocessing/
│   └── models/
│       ├── text_preprocessor.py # Text preprocessing
│       └── feature_engineering.py # Feature engineering
├── explainability/
│   └── models/
│       └── shap_explainer.py    # SHAP explanations
├── ocr/
│   └── models/
│       └── ocr_processor.py     # OCR processing
├── verification/
│   └── models/
│       └── company_verifier.py  # Company verification
├── anomaly/
│   └── models/
│       └── salary_anomaly_detector.py # Salary anomaly detection
├── frontend/
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── pages/              # React pages
│   │   ├── App.tsx             # Main app component
│   │   └── main.tsx            # Entry point
│   ├── package.json            # Node dependencies
│   └── vite.config.ts          # Vite configuration
├── scripts/
│   ├── prepare_dataset.py      # Dataset preparation
│   ├── clean_dataset.py        # Data cleaning
│   └── evaluate_models.py     # Model evaluation
├── data/
│   ├── raw/                    # Raw data
│   ├── processed/              # Processed data
│   └── models/                 # Trained models
├── docs/                       # Documentation
├── tests/                      # Test files
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
└── docker-compose.yml          # Docker Compose configuration
```

## Development Setup

### Prerequisites

- Python 3.12+
- Node.js 18+
- Git
- Docker (optional)

### Backend Development

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Run backend
python3 backend/api/main.py
```

### Frontend Development

```bash
cd frontend
npm install
npm run dev
```

### Code Quality

```bash
# Python linting
flake8 backend/ models/ preprocessing/

# Python formatting
black backend/ models/ preprocessing/

# Type checking
mypy backend/ models/

# JavaScript linting
cd frontend
npm run lint
```

## Code Architecture

### Backend Architecture

**FastAPI Application:**
- RESTful API design
- Async/await support
- Automatic API documentation
- Request validation with Pydantic

**Model Architecture:**
- Modular design with separate model types
- Common interface for predictions
- Ensemble/hybrid model support

**Data Pipeline:**
- Text preprocessing pipeline
- Feature engineering pipeline
- Model inference pipeline

### Frontend Architecture

**React Components:**
- Functional components with hooks
- Component composition
- State management with React hooks
- Routing with React Router

**Styling:**
- Tailwind CSS for styling
- Responsive design
- Dark mode support (future)

### Database Architecture

**SQLAlchemy Models:**
- Declarative model definitions
- Relationship management
- Migration support (Alembic)

**Connection Management:**
- Connection pooling
- Session management
- Transaction handling

## Testing

### Backend Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=backend --cov=models --cov=preprocessing

# Run specific test file
pytest tests/test_api.py

# Run with verbose output
pytest tests/ -v
```

### Frontend Testing

```bash
cd frontend

# Run unit tests
npm test

# Run with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e
```

### Integration Testing

```bash
# Test API endpoints
pytest tests/integration/test_api.py

# Test database operations
pytest tests/integration/test_database.py

# Test model predictions
pytest tests/integration/test_models.py
```

## Model Training

### Baseline Models

```bash
# Train baseline models
python3 models/baseline/baseline_models.py

# This will:
# - Load processed data
# - Train Logistic Regression
# - Train Random Forest
# - Train XGBoost (if available)
# - Save models to data/models/
```

### RoBERTa Model

```bash
# Train RoBERTa model
python3 models/roberta/roberta_trainer.py

# This will:
# - Fine-tune RoBERTa on job data
# - Use attention masks
# - Implement learning rate scheduling
# - Save model to data/models/roberta_fake_job_detector/
```

### Hybrid Model

```bash
# Train hybrid model
python3 models/hybrid/hybrid_model.py

# This will:
# - Combine multiple model predictions
# - Weight ensemble components
# - Integrate with OCR and verification
# - Save configuration to data/models/
```

### Model Evaluation

```bash
# Evaluate all models
python3 scripts/evaluate_models.py

# This will:
# - Load all trained models
# - Calculate comprehensive metrics
# - Generate comparison plots
# - Save evaluation report
```

## API Development

### Adding New Endpoints

1. **Define Pydantic models** in `backend/api/main.py`:
```python
class NewRequest(BaseModel):
    field1: str
    field2: int

class NewResponse(BaseModel):
    result: str
    status: str
```

2. **Create endpoint function**:
```python
@app.post("/new-endpoint", response_model=NewResponse)
async def new_endpoint(request: NewRequest):
    # Process request
    result = process_data(request)
    return NewResponse(result=result, status="success")
```

3. **Test endpoint**:
```bash
curl -X POST http://localhost:8000/new-endpoint \
  -H "Content-Type: application/json" \
  -d '{"field1": "value", "field2": 123}'
```

### API Documentation

- Automatic documentation at `/docs` (Swagger UI)
- Alternative documentation at `/redoc` (ReDoc)
- Update docstrings for better documentation

## Frontend Development

### Adding New Components

1. **Create component file** in `frontend/src/components/`:
```typescript
import React from 'react'

function NewComponent({ prop1, prop2 }: { prop1: string; prop2: number }) {
  return (
    <div className="card">
      <h3>{prop1}</h3>
      <p>{prop2}</p>
    </div>
  )
}

export default NewComponent
```

2. **Use component in page**:
```typescript
import NewComponent from '../components/NewComponent'

function Page() {
  return (
    <div>
      <NewComponent prop1="Hello" prop2={42} />
    </div>
  )
}
```

### Adding New Pages

1. **Create page file** in `frontend/src/pages/`:
```typescript
import React from 'react'

function NewPage() {
  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold">New Page</h1>
    </div>
  )
}

export default NewPage
```

2. **Add route** in `frontend/src/App.tsx`:
```typescript
<Route path="/new-page" element={<NewPage />} />
```

3. **Add navigation link** in `frontend/src/components/Navbar.tsx`:
```typescript
<Link to="/new-page" className="text-gray-700 hover:text-primary-600">
  New Page
</Link>
```

## Deployment

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop services
docker-compose down

# Rebuild specific service
docker-compose up -d --build backend
```

### Production Deployment

**Backend:**
```bash
# Set environment variables
export DATABASE_URL=postgresql://...
export SECRET_KEY=your-secret-key
export API_HOST=0.0.0.0
export API_PORT=8000

# Run with gunicorn (production WSGI server)
pip install gunicorn uvicorn
gunicorn backend.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

**Frontend:**
```bash
cd frontend

# Build for production
npm run build

# Serve with nginx or similar
# Or use node: npm run preview
```

**Database:**
```bash
# Use PostgreSQL for production
# Set up connection pooling
# Configure backups
# Enable SSL
```

## Contributing

### Code Style

**Python:**
- Follow PEP 8
- Use type hints
- Write docstrings
- Maximum line length: 100 characters

**TypeScript/React:**
- Follow ESLint rules
- Use functional components
- Use TypeScript types
- Follow React best practices

### Commit Messages

Follow conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Maintenance tasks

Example:
```
feat: add company verification module

Implement company legitimacy verification using
multiple data sources and heuristic checks.
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit pull request
7. Address review feedback

### Testing Requirements

- Unit tests for new functions
- Integration tests for new endpoints
- Update test coverage
- All tests must pass before merging

## Performance Optimization

### Backend Optimization

- Use async/await for I/O operations
- Implement caching for expensive operations
- Use connection pooling for database
- Optimize database queries
- Use batch processing for bulk operations

### Frontend Optimization

- Code splitting with React.lazy
- Image optimization
- Lazy loading components
- Minimize bundle size
- Use CDN for static assets

### Model Optimization

- Model quantization for deployment
- Batch inference for multiple predictions
- Model pruning for smaller size
- Use ONNX for cross-platform deployment

## Security Considerations

### Backend Security

- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CSRF protection
- Rate limiting
- Authentication and authorization

### Frontend Security

- Content Security Policy
- HTTPS only in production
- Secure cookies
- Input validation
- XSS prevention

### Data Security

- Encrypt sensitive data
- Secure database connections
- Regular security audits
- Dependency updates
- Environment variable management

## Monitoring and Logging

### Application Monitoring

- Use application performance monitoring (APM)
- Track API response times
- Monitor error rates
- Set up alerts for critical issues

### Logging

- Structured logging
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Log aggregation
- Log retention policy

### Metrics

- Track prediction accuracy
- Monitor model performance
- Track API usage
- Monitor system resources

## Troubleshooting

### Common Development Issues

**Import errors:**
```bash
# Solution: Check PYTHONPATH
export PYTHONPATH=/path/to/project
```

**Database connection errors:**
```bash
# Solution: Check DATABASE_URL
# Ensure database is running
```

**Frontend build errors:**
```bash
# Solution: Clear cache and rebuild
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## Support

For development questions:
- GitHub Issues
- Community Forum
- Developer Documentation
- Code Review
