"""
Model Tests
Test cases for model components
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class TestHybridModel:
    """Test hybrid model"""
    
    @pytest.fixture
    def hybrid_model(self):
        """Create hybrid model instance"""
        try:
            from models.hybrid.hybrid_model import HybridModel
            return HybridModel()
        except ImportError:
            pytest.skip("Hybrid model not available")
    
    def test_model_initialization(self, hybrid_model):
        """Test model initialization"""
        assert hybrid_model is not None
        assert hybrid_model.weights is not None
        assert len(hybrid_model.weights) > 0
    
    def test_prediction(self, hybrid_model):
        """Test prediction functionality"""
        test_text = "Senior Software Engineer position with Python experience"
        result = hybrid_model.predict(text=test_text)
        
        assert "prediction" in result
        assert "confidence" in result
        assert "fraud_probability" in result
        assert "risk_level" in result
    
    def test_prediction_with_company_data(self, hybrid_model):
        """Test prediction with company data"""
        test_text = "Senior Software Engineer position"
        company_data = {
            'has_company_logo': 1,
            'company_profile': 'TechCorp is a leading technology company',
            'email': 'careers@techcorp.com'
        }
        
        result = hybrid_model.predict(
            text=test_text,
            company_data=company_data,
            salary="$80,000 - $95,000"
        )
        
        assert "prediction" in result
        assert "components" in result


class TestFeatureEngineering:
    """Test feature engineering"""
    
    @pytest.fixture
    def feature_engineer(self):
        """Create feature engineer instance"""
        try:
            from preprocessing.models.feature_engineering import FeatureEngineer
            return FeatureEngineer()
        except ImportError:
            pytest.skip("Feature engineer not available")
    
    def test_scam_keyword_frequency(self, feature_engineer):
        """Test scam keyword frequency calculation"""
        text = "URGENT! Earn money quickly with no experience needed"
        frequency = feature_engineer.calculate_scam_keyword_frequency(text)
        
        assert frequency >= 0
        assert isinstance(frequency, float)
    
    def test_urgency_score(self, feature_engineer):
        """Test urgency score calculation"""
        text = "URGENT! Immediate hiring! Apply now!"
        score = feature_engineer.calculate_urgency_score(text)
        
        assert score >= 0
        assert isinstance(score, float)
    
    def test_salary_realism(self, feature_engineer):
        """Test salary realism calculation"""
        salary = "$80,000 - $95,000"
        score = feature_engineer.calculate_salary_realism_score(salary)
        
        assert score >= 0
        assert score <= 1


class TestTextPreprocessor:
    """Test text preprocessor"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance"""
        try:
            from preprocessing.models.text_preprocessor import TextPreprocessor
            return TextPreprocessor()
        except ImportError:
            pytest.skip("Text preprocessor not available")
    
    def test_preprocessing(self, preprocessor):
        """Test text preprocessing"""
        text = "Visit http://example.com for more info! URGENT hiring!"
        processed = preprocessor.preprocess(text)
        
        assert processed is not None
        assert len(processed) > 0
    
    def test_url_removal(self, preprocessor):
        """Test URL removal"""
        text = "Visit http://example.com and https://test.com"
        processed = preprocessor.remove_urls(text)
        
        assert "http://" not in processed
        assert "https://" not in processed
    
    def test_email_removal(self, preprocessor):
        """Test email removal"""
        text = "Contact us at test@example.com"
        processed = preprocessor.remove_emails(text)
        
        assert "@" not in processed


class TestSHAPExplainer:
    """Test SHAP explainer"""
    
    @pytest.fixture
    def explainer(self):
        """Create explainer instance"""
        try:
            from explainability.models.shap_explainer import SHAPExplainer
            return SHAPExplainer()
        except ImportError:
            pytest.skip("SHAP explainer not available")
    
    def test_explainer_initialization(self, explainer):
        """Test explainer initialization"""
        assert explainer is not None
        assert explainer.suspicious_patterns is not None
    
    def test_rule_based_explanation(self, explainer):
        """Test rule-based explanation"""
        text = "URGENT! Earn $5000 weekly working from home"
        explanation = explainer.explain_with_rules(text)
        
        assert "method" in explanation
        assert "explanations" in explanation
        assert "top_suspicious_phrases" in explanation
    
    def test_highlight_suspicious_phrases(self, explainer):
        """Test suspicious phrase highlighting"""
        text = "URGENT! Earn $5000 weekly"
        explanation = explainer.explain_with_rules(text)
        highlighted = explainer.highlight_suspicious_phrases(text, explanation)
        
        assert highlighted is not None
        assert "<mark" in highlighted or len(highlighted) > 0


class TestCompanyVerifier:
    """Test company verifier"""
    
    @pytest.fixture
    def verifier(self):
        """Create verifier instance"""
        try:
            from verification.models.company_verifier import CompanyVerifier
            return CompanyVerifier()
        except ImportError:
            pytest.skip("Company verifier not available")
    
    def test_company_verification(self, verifier):
        """Test company verification"""
        company_data = {
            'name': 'TechCorp',
            'website': 'https://techcorp.com',
            'email': 'careers@techcorp.com',
            'profile': 'TechCorp is a leading technology company'
        }
        
        result = verifier.verify_company(company_data)
        
        assert "legitimacy_score" in result
        assert "risk_level" in result
        assert "checks" in result


class TestSalaryAnomalyDetector:
    """Test salary anomaly detector"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance"""
        try:
            from anomaly.models.salary_anomaly_detector import SalaryAnomalyDetector
            return SalaryAnomalyDetector()
        except ImportError:
            pytest.skip("Salary anomaly detector not available")
    
    def test_salary_feature_extraction(self, detector):
        """Test salary feature extraction"""
        salary = "$80,000 - $95,000"
        features = detector.extract_salary_features(salary)
        
        assert "has_salary" in features
        assert "salary_min" in features
        assert "salary_max" in features
    
    def test_heuristic_detection(self, detector):
        """Test heuristic anomaly detection"""
        salary = "$5000/week"
        result = detector._heuristic_detection(salary)
        
        assert "is_anomaly" in result
        assert "anomaly_probability" in result
        assert "method" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
