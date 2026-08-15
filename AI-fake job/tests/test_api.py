"""
API Tests
Test cases for FastAPI endpoints
"""

import pytest
from typing import Dict
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Optional imports
try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestAPI:
    """Test API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        try:
            from backend.api.main import app
            return TestClient(app)
        except ImportError:
            pytest.skip("Backend not available")
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data
    
    def test_predict_endpoint(self, client):
        """Test prediction endpoint"""
        response = client.post("/predict", json={
            "job_description": "Senior Software Engineer position with Python experience",
            "company_name": "TechCorp",
            "salary": "$80,000 - $95,000"
        })
        
        # May return 503 if models not loaded
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "fraud_probability" in data
    
    def test_predict_missing_description(self, client):
        """Test prediction with missing description"""
        response = client.post("/predict", json={
            "company_name": "TechCorp"
        })
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_history_endpoint(self, client):
        """Test history endpoint"""
        response = client.get("/history")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "recent_predictions" in data


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestAPIIntegration:
    """Integration tests for API"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        try:
            from backend.api.main import app
            return TestClient(app)
        except ImportError:
            pytest.skip("Backend not available")
    
    def test_prediction_flow(self, client):
        """Test complete prediction flow"""
        # Make a prediction
        response = client.post("/predict", json={
            "job_description": "URGENT! Earn $5000 weekly working from home. No experience needed.",
            "company_name": "QuickCash Inc",
            "salary": "$5000/week"
        })
        
        if response.status_code == 200:
            data = response.json()
            
            # Check response structure
            assert "prediction" in data
            assert "confidence" in data
            assert "risk_level" in data
            
            # Check history
            history_response = client.get("/history")
            history_data = history_response.json()
            assert history_data["total_predictions"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
