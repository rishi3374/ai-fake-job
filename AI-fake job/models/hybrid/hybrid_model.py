"""
Hybrid Model Module
Combines RoBERTa predictions with salary anomaly detection, company legitimacy, and OCR confidence
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, List
import json
import joblib

# Import feature engineering
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from preprocessing.models.feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HybridModel:
    """
    Hybrid ensemble model combining multiple prediction sources:
    - RoBERTa text classification
    - Salary anomaly detection
    - Company legitimacy verification
    - OCR confidence (for image inputs)
    """
    
    def __init__(self, 
                 model_dir: str = "data/models",
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize hybrid model
        
        Args:
            model_dir: Directory containing trained models
            weights: Custom weights for each component
        """
        self.model_dir = Path(model_dir)
        self.feature_engineer = FeatureEngineer()
        
        # Default weights for each component
        self.weights = weights or {
            'roberta': 0.4,           # RoBERTa prediction
            'salary_anomaly': 0.25,   # Salary anomaly score
            'company_legitimacy': 0.2, # Company legitimacy
            'ocr_confidence': 0.15,   # OCR confidence (when applicable)
            'heuristic_features': 0.3 # Feature engineering scores
        }
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        # Load models
        self.models = {}
        self._load_models()
        
        logger.info(f"Hybrid model initialized with weights: {self.weights}")
    
    def _load_models(self):
        """Load individual component models"""
        # Load baseline models
        try:
            lr_path = self.model_dir / "logistic_regression.joblib"
            if lr_path.exists():
                self.models['logistic_regression'] = joblib.load(lr_path)
                logger.info("Loaded Logistic Regression model")
            
            rf_path = self.model_dir / "random_forest.joblib"
            if rf_path.exists():
                self.models['random_forest'] = joblib.load(rf_path)
                logger.info("Loaded Random Forest model")
            
            # Load TF-IDF vectorizer
            tfidf_path = self.model_dir / "tfidf_vectorizer.joblib"
            if tfidf_path.exists():
                self.models['tfidf_vectorizer'] = joblib.load(tfidf_path)
                logger.info("Loaded TF-IDF vectorizer")
            
        except Exception as e:
            logger.warning(f"Error loading baseline models: {e}")
        
        # RoBERTa model loading (if available)
        try:
            from models.roberta.roberta_trainer import RoBERTaTrainer
            roberta_path = self.model_dir / "roberta_fake_job_detector"
            if roberta_path.exists():
                self.roberta_trainer = RoBERTaTrainer()
                self.roberta_trainer.load_model(str(roberta_path))
                self.models['roberta'] = self.roberta_trainer
                logger.info("Loaded RoBERTa model")
        except Exception as e:
            logger.warning(f"RoBERTa model not available: {e}")
    
    def calculate_salary_anomaly_score(self, salary_str: str, job_title: str = "") -> float:
        """
        Calculate salary anomaly score (0-1, where 1 indicates anomaly)
        
        Args:
            salary_str: Salary string or range
            job_title: Job title for context
            
        Returns:
            Anomaly score between 0 and 1
        """
        realism_score = self.feature_engineer.calculate_salary_realism_score(salary_str, job_title)
        anomaly_score = 1.0 - realism_score
        return anomaly_score
    
    def calculate_company_legitimacy_score(self, company_data: Dict) -> float:
        """
        Calculate company legitimacy score (0-1, where 1 indicates legitimate)
        
        Args:
            company_data: Dictionary containing company information
            
        Returns:
            Legitimacy score between 0 and 1
        """
        score = 0.0
        
        # Check for company logo
        if company_data.get('has_company_logo', 0) == 1:
            score += 0.3
        
        # Check for company profile
        profile = company_data.get('company_profile', '')
        if profile and len(str(profile)) > 50:
            score += 0.3
        elif profile and len(str(profile)) > 20:
            score += 0.15
        
        # Check for questions
        if company_data.get('has_questions', 0) == 1:
            score += 0.2
        
        # Check for professional email
        email = company_data.get('email', '')
        if email:
            free_providers = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
            if not any(provider in email.lower() for provider in free_providers):
                score += 0.2
        
        return min(score, 1.0)
    
    def calculate_ocr_confidence(self, ocr_result: Dict) -> float:
        """
        Calculate OCR confidence score (0-1)
        
        Args:
            ocr_result: Dictionary containing OCR results
            
        Returns:
            Confidence score between 0 and 1
        """
        if not ocr_result:
            return 0.5
        
        # Use average confidence from OCR
        confidences = ocr_result.get('confidences', [])
        if confidences:
            return np.mean(confidences) / 100.0  # Assuming confidence is 0-100
        
        # Fallback to text length and quality
        extracted_text = ocr_result.get('text', '')
        if extracted_text:
            # Basic quality check
            text_length = len(extracted_text)
            if text_length > 100:  # Good amount of text extracted
                return 0.8
            elif text_length > 50:
                return 0.6
            else:
                return 0.4
        
        return 0.5
    
    def get_roberta_prediction(self, text: str) -> Dict:
        """
        Get RoBERTa prediction for text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with prediction results
        """
        if 'roberta' not in self.models:
            # Fallback to baseline models
            return self.get_baseline_prediction(text)
        
        try:
            return self.models['roberta'].predict(text)
        except Exception as e:
            logger.warning(f"RoBERTa prediction failed: {e}, using baseline")
            return self.get_baseline_prediction(text)
    
    def get_baseline_prediction(self, text: str) -> Dict:
        """
        Get baseline model prediction (fallback)
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with prediction results
        """
        if 'logistic_regression' not in self.models or 'tfidf_vectorizer' not in self.models:
            # Use heuristic features
            return self.get_heuristic_prediction(text)
        
        try:
            vectorizer = self.models['tfidf_vectorizer']
            model = self.models['logistic_regression']
            
            # Transform text
            text_tfidf = vectorizer.transform([text])
            
            # Make prediction
            prediction = model.predict(text_tfidf)[0]
            probability = model.predict_proba(text_tfidf)[0, 1]
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'is_fake': bool(prediction == 1),
                'confidence': float(probability if prediction == 1 else 1 - probability)
            }
        except Exception as e:
            logger.warning(f"Baseline prediction failed: {e}, using heuristic")
            return self.get_heuristic_prediction(text)
    
    def get_heuristic_prediction(self, text: str) -> Dict:
        """
        Get heuristic-based prediction (final fallback)
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with prediction results
        """
        # Calculate heuristic features
        scam_keyword_freq = self.feature_engineer.calculate_scam_keyword_frequency(text)
        urgency_score = self.feature_engineer.calculate_urgency_score(text)
        suspicious_patterns = self.feature_engineer.detect_suspicious_patterns(text)
        
        # Combine into a simple score
        heuristic_score = (
            scam_keyword_freq * 0.4 +
            min(urgency_score / 5.0, 1.0) * 0.3 +
            min(suspicious_patterns / 3.0, 1.0) * 0.3
        )
        
        prediction = 1 if heuristic_score > 0.5 else 0
        confidence = heuristic_score if prediction == 1 else 1 - heuristic_score
        
        return {
            'prediction': prediction,
            'probability': heuristic_score,
            'is_fake': bool(prediction == 1),
            'confidence': confidence,
            'method': 'heuristic'
        }
    
    def predict(self, 
                text: str,
                salary_str: str = "",
                company_data: Optional[Dict] = None,
                ocr_result: Optional[Dict] = None,
                job_title: str = "") -> Dict:
        """
        Make hybrid prediction combining all components
        
        Args:
            text: Job description text
            salary_str: Salary information
            company_data: Company information dictionary
            ocr_result: OCR results (if applicable)
            job_title: Job title
            
        Returns:
            Dictionary with comprehensive prediction results
        """
        logger.info("Making hybrid prediction...")
        
        # Initialize company data if not provided
        if company_data is None:
            company_data = {}
        
        # Get individual component predictions
        roberta_result = self.get_roberta_prediction(text)
        salary_anomaly_score = self.calculate_salary_anomaly_score(salary_str, job_title)
        company_legitimacy_score = self.calculate_company_legitimacy_score(company_data)
        ocr_confidence = self.calculate_ocr_confidence(ocr_result) if ocr_result else 0.5
        
        # Calculate heuristic features from text
        scam_keyword_freq = self.feature_engineer.calculate_scam_keyword_frequency(text)
        urgency_score = self.feature_engineer.calculate_urgency_score(text)
        heuristic_score = min(scam_keyword_freq * 2 + urgency_score * 0.2, 1.0)
        
        # Combine predictions using weighted average
        weighted_fraud_probability = (
            roberta_result['probability'] * self.weights['roberta'] +
            salary_anomaly_score * self.weights['salary_anomaly'] +
            (1 - company_legitimacy_score) * self.weights['company_legitimacy'] +
            (1 - ocr_confidence) * self.weights['ocr_confidence'] +
            heuristic_score * self.weights['heuristic_features']
        )
        
        # Final prediction
        final_prediction = 1 if weighted_fraud_probability > 0.5 else 0
        final_confidence = weighted_fraud_probability if final_prediction == 1 else 1 - weighted_fraud_probability
        
        # Determine risk level
        if final_confidence > 0.8:
            risk_level = "High"
        elif final_confidence > 0.5:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        result = {
            'prediction': int(final_prediction),
            'is_fake': bool(final_prediction == 1),
            'confidence': float(final_confidence),
            'fraud_probability': float(weighted_fraud_probability),
            'risk_level': risk_level,
            'components': {
                'roberta': roberta_result,
                'salary_anomaly': float(salary_anomaly_score),
                'company_legitimacy': float(company_legitimacy_score),
                'ocr_confidence': float(ocr_confidence),
                'heuristic_score': float(heuristic_score)
            },
            'weights': self.weights
        }
        
        return result
    
    def predict_batch(self, 
                     texts: List[str],
                     salary_strs: Optional[List[str]] = None,
                     company_data_list: Optional[List[Dict]] = None,
                     ocr_results: Optional[List[Dict]] = None,
                     job_titles: Optional[List[str]] = None) -> List[Dict]:
        """
        Make predictions on batch of job postings
        
        Args:
            texts: List of job descriptions
            salary_strs: List of salary strings
            company_data_list: List of company data dictionaries
            ocr_results: List of OCR results
            job_titles: List of job titles
            
        Returns:
            List of prediction results
        """
        results = []
        n_samples = len(texts)
        
        # Handle optional parameters
        salary_strs = salary_strs or [""] * n_samples
        company_data_list = company_data_list or [{}] * n_samples
        ocr_results = ocr_results or [None] * n_samples
        job_titles = job_titles or [""] * n_samples
        
        for i in range(n_samples):
            result = self.predict(
                text=texts[i],
                salary_str=salary_strs[i],
                company_data=company_data_list[i],
                ocr_result=ocr_results[i],
                job_title=job_titles[i]
            )
            results.append(result)
        
        return results
    
    def explain_prediction(self, prediction_result: Dict) -> str:
        """
        Generate human-readable explanation of prediction
        
        Args:
            prediction_result: Result from predict method
            
        Returns:
            Explanation string
        """
        components = prediction_result['components']
        
        explanation_parts = []
        
        # Main prediction
        if prediction_result['is_fake']:
            explanation_parts.append(f"This job posting is classified as FAKE with {prediction_result['confidence']:.1%} confidence.")
        else:
            explanation_parts.append(f"This job posting is classified as REAL with {prediction_result['confidence']:.1%} confidence.")
        
        # Component explanations
        roberta_prob = components['roberta']['probability']
        if roberta_prob > 0.7:
            explanation_parts.append(f"Text analysis strongly suggests this is fake ({roberta_prob:.1%}).")
        elif roberta_prob < 0.3:
            explanation_parts.append(f"Text analysis suggests this is legitimate ({1-roberta_prob:.1%}).")
        
        if components['salary_anomaly'] > 0.6:
            explanation_parts.append("The salary information appears unrealistic or suspicious.")
        
        if components['company_legitimacy'] < 0.4:
            explanation_parts.append("The company profile lacks typical legitimacy indicators.")
        
        if components['heuristic_score'] > 0.5:
            explanation_parts.append("The text contains multiple scam indicators or urgency language.")
        
        return " ".join(explanation_parts)
    
    def save_model(self, model_name: str = "hybrid_model") -> Path:
        """
        Save hybrid model configuration
        
        Args:
            model_name: Name for saved model
            
        Returns:
            Path to saved model
        """
        model_path = self.model_dir / f"{model_name}.json"
        
        config = {
            'weights': self.weights,
            'model_name': model_name,
            'components': list(self.weights.keys())
        }
        
        with open(model_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Hybrid model configuration saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path: str):
        """
        Load hybrid model configuration
        
        Args:
            model_path: Path to saved model configuration
        """
        model_path = Path(model_path)
        
        with open(model_path, 'r') as f:
            config = json.load(f)
        
        self.weights = config['weights']
        logger.info(f"Hybrid model configuration loaded from {model_path}")


def main():
    """Main execution function for testing"""
    # Create hybrid model
    hybrid_model = HybridModel()
    
    # Test prediction
    test_text = "URGENT! Earn $5000 weekly working from home. No experience needed. Start today!"
    test_salary = "$5000/week"
    test_company = {
        'has_company_logo': 0,
        'company_profile': '',
        'has_questions': 0,
        'email': 'gethiredquick@gmail.com'
    }
    
    result = hybrid_model.predict(
        text=test_text,
        salary_str=test_salary,
        company_data=test_company,
        job_title="Easy Money Maker"
    )
    
    print("=== Hybrid Model Prediction ===")
    print(f"Prediction: {'FAKE' if result['is_fake'] else 'REAL'}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Fraud Probability: {result['fraud_probability']:.1%}")
    print(f"\nComponent Scores:")
    for component, score in result['components'].items():
        if isinstance(score, dict):
            print(f"  {component}: {score}")
        else:
            print(f"  {component}: {score:.3f}")
    
    print(f"\nExplanation:")
    print(hybrid_model.explain_prediction(result))
    
    # Save model
    hybrid_model.save_model()


if __name__ == "__main__":
    main()
