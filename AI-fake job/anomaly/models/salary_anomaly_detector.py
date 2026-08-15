"""
Salary Anomaly Detection Module
Uses Isolation Forest to detect anomalous salary claims in job postings
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import re
import joblib
import json

# Optional sklearn imports
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SalaryAnomalyDetector:
    """Salary anomaly detection using Isolation Forest"""
    
    def __init__(self, model_dir: str = "data/models"):
        """
        Initialize salary anomaly detector
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        # Salary patterns for extraction
        self.salary_patterns = [
            r'\$?\d{1,3}(?:,\d{3})*(?:\s*(?:to|-|–)\s*\$?\d{1,3}(?:,\d{3})*)?',
            r'\$?\d+(?:\.\d+)?',
        ]
        
        logger.info("Salary Anomaly Detector initialized")
    
    def extract_salary_features(self, salary_str: str, job_title: str = "") -> Dict:
        """
        Extract numerical features from salary string
        
        Args:
            salary_str: Salary string
            job_title: Job title for context
            
        Returns:
            Dictionary of salary features
        """
        features = {
            'has_salary': 0,
            'salary_min': 0,
            'salary_max': 0,
            'salary_avg': 0,
            'salary_range': 0,
            'is_hourly': 0,
            'is_daily': 0,
            'is_weekly': 0,
            'is_monthly': 0,
            'is_annual': 0,
            'has_range': 0,
            'per_1000': 0,
            'round_number': 0
        }
        
        if pd.isna(salary_str) or salary_str == '':
            return features
        
        features['has_salary'] = 1
        
        # Extract numeric values
        numbers = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', str(salary_str))
        if numbers:
            # Clean numbers (remove commas)
            clean_numbers = [float(num.replace(',', '')) for num in numbers]
            
            if len(clean_numbers) >= 1:
                features['salary_min'] = min(clean_numbers)
                features['salary_max'] = max(clean_numbers)
                features['salary_avg'] = np.mean(clean_numbers)
                features['salary_range'] = features['salary_max'] - features['salary_min']
                
                # Check if it's a range
                features['has_range'] = 1 if len(clean_numbers) > 1 else 0
        
        # Detect time period
        salary_lower = str(salary_str).lower()
        if 'hour' in salary_lower or 'hr' in salary_lower:
            features['is_hourly'] = 1
        elif 'day' in salary_lower:
            features['is_daily'] = 1
        elif 'week' in salary_lower or 'wk' in salary_lower:
            features['is_weekly'] = 1
        elif 'month' in salary_lower or 'mo' in salary_lower:
            features['is_monthly'] = 1
        elif 'year' in salary_lower or 'yr' in salary_lower or 'annual' in salary_lower:
            features['is_annual'] = 1
        
        # Check for suspicious patterns
        if features['salary_avg'] > 0:
            # Check if salary is a round number (suspicious)
            if features['salary_avg'] % 1000 == 0:
                features['round_number'] = 1
            
            # Check if salary is per 1000 (suspicious)
            if 'k' in salary_lower or '000' in salary_lower:
                features['per_1000'] = 1
        
        return features
    
    def normalize_salary(self, salary: float, time_period: str) -> float:
        """
        Normalize salary to annual equivalent
        
        Args:
            salary: Salary amount
            time_period: Time period ('hourly', 'daily', 'weekly', 'monthly', 'annual')
            
        Returns:
            Annualized salary
        """
        multipliers = {
            'hourly': 2080,  # 40 hours * 52 weeks
            'daily': 260,    # 5 days * 52 weeks
            'weekly': 52,
            'monthly': 12,
            'annual': 1
        }
        
        return salary * multipliers.get(time_period, 1)
    
    def create_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create feature matrix for anomaly detection
        
        Args:
            df: Input dataframe with salary information
            
        Returns:
            Feature matrix
        """
        features_list = []
        
        for _, row in df.iterrows():
            salary_features = self.extract_salary_features(
                row.get('salary_range', ''),
                row.get('title', '')
            )
            features_list.append(salary_features)
        
        feature_df = pd.DataFrame(features_list)
        
        # Add contextual features
        feature_df['title_length'] = df['title'].str.len().fillna(0)
        feature_df['description_length'] = df['description'].str.len().fillna(0)
        
        # Store feature columns
        self.feature_columns = feature_df.columns.tolist()
        
        return feature_df
    
    def train(self, df: pd.DataFrame, contamination: float = 0.1) -> Dict:
        """
        Train Isolation Forest model
        
        Args:
            df: Training dataframe
            contamination: Expected proportion of outliers
            
        Returns:
            Training results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for salary anomaly detection")
        
        logger.info("Training salary anomaly detector...")
        
        # Create feature matrix
        feature_df = self.create_feature_matrix(df)
        
        # Scale features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(feature_df)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        
        # Fit model
        self.model.fit(scaled_features)
        
        # Get predictions
        predictions = self.model.predict(scaled_features)
        anomaly_scores = self.model.score_samples(scaled_features)
        
        # Calculate statistics
        anomaly_count = sum(1 for pred in predictions if pred == -1)
        anomaly_rate = anomaly_count / len(predictions)
        
        results = {
            'total_samples': len(predictions),
            'anomaly_count': anomaly_count,
            'anomaly_rate': anomaly_rate,
            'contamination': contamination,
            'feature_columns': self.feature_columns
        }
        
        logger.info(f"Training completed. Anomaly rate: {anomaly_rate:.2%}")
        
        # Save model
        self.save_model()
        
        return results
    
    def detect_anomaly(self, salary_str: str, job_title: str = "", description: str = "") -> Dict:
        """
        Detect if a salary is anomalous
        
        Args:
            salary_str: Salary string
            job_title: Job title for context
            description: Job description for context
            
        Returns:
            Anomaly detection result
        """
        if self.model is None or self.scaler is None:
            logger.warning("Model not trained, using heuristic detection")
            return self._heuristic_detection(salary_str, job_title)
        
        # Extract features
        features = self.extract_salary_features(salary_str, job_title)
        
        # Add contextual features if they were used during training
        if self.feature_columns and 'title_length' in self.feature_columns:
            features['title_length'] = len(job_title)
            features['description_length'] = len(description)
        
        # Ensure all expected features are present
        if self.feature_columns:
            feature_vector = []
            for col in self.feature_columns:
                feature_vector.append(features.get(col, 0))
            feature_vector = np.array([feature_vector])
        else:
            feature_vector = np.array([list(features.values())])
        
        # Scale features
        scaled_features = self.scaler.transform(feature_vector)
        
        # Predict
        prediction = self.model.predict(scaled_features)[0]
        anomaly_score = self.model.score_samples(scaled_features)[0]
        
        # Convert to probability-like score
        anomaly_probability = 1 - (anomaly_score + 0.5)  # Normalize to 0-1
        
        return {
            'is_anomaly': prediction == -1,
            'anomaly_score': float(anomaly_score),
            'anomaly_probability': float(anomaly_probability),
            'method': 'isolation_forest',
            'features': features
        }
    
    def _heuristic_detection(self, salary_str: str, job_title: str = "") -> Dict:
        """
        Heuristic-based anomaly detection (fallback)
        
        Args:
            salary_str: Salary string
            job_title: Job title for context
            
        Returns:
            Anomaly detection result
        """
        features = self.extract_salary_features(salary_str, job_title)
        
        anomaly_indicators = 0
        reasons = []
        
        # Check for suspicious patterns
        if features['salary_avg'] > 0:
            # Very high weekly salary
            if features['is_weekly'] and features['salary_avg'] > 2000:
                anomaly_indicators += 1
                reasons.append("Very high weekly salary")
            
            # Very high daily salary
            if features['is_daily'] and features['salary_avg'] > 500:
                anomaly_indicators += 1
                reasons.append("Very high daily salary")
            
            # Very high hourly rate without senior title
            if features['is_hourly'] and features['salary_avg'] > 100:
                if 'senior' not in job_title.lower():
                    anomaly_indicators += 1
                    reasons.append("High hourly rate for non-senior position")
            
            # Very low annual salary
            if features['is_annual'] and features['salary_avg'] < 15000:
                anomaly_indicators += 1
                reasons.append("Very low annual salary")
            
            # Extremely high annual salary
            if features['is_annual'] and features['salary_avg'] > 500000:
                anomaly_indicators += 1
                reasons.append("Extremely high annual salary")
            
            # Round number pattern
            if features['round_number']:
                anomaly_indicators += 0.5
                reasons.append("Round salary number (suspicious)")
        
        # Calculate anomaly probability
        anomaly_probability = min(anomaly_indicators / 3.0, 1.0)
        
        return {
            'is_anomaly': anomaly_indicators >= 1,
            'anomaly_score': -anomaly_probability if anomaly_indicators >= 1 else 0.5,
            'anomaly_probability': anomaly_probability,
            'method': 'heuristic',
            'features': features,
            'reasons': reasons
        }
    
    def batch_detect(self, salaries: List[str], job_titles: List[str] = None) -> List[Dict]:
        """
        Detect anomalies in batch
        
        Args:
            salaries: List of salary strings
            job_titles: List of job titles
            
        Returns:
            List of anomaly detection results
        """
        if job_titles is None:
            job_titles = [""] * len(salaries)
        
        results = []
        for salary, title in zip(salaries, job_titles):
            result = self.detect_anomaly(salary, title)
            results.append(result)
        
        return results
    
    def save_model(self, model_name: str = "salary_anomaly_detector"):
        """Save trained model"""
        if self.model is not None:
            model_path = self.model_dir / f"{model_name}.joblib"
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
        
        if self.scaler is not None:
            scaler_path = self.model_dir / f"{model_name}_scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
        
        # Save feature columns
        if self.feature_columns is not None:
            config_path = self.model_dir / f"{model_name}_config.json"
            with open(config_path, 'w') as f:
                json.dump({'feature_columns': self.feature_columns}, f)
            logger.info(f"Config saved to {config_path}")
    
    def load_model(self, model_name: str = "salary_anomaly_detector"):
        """Load trained model"""
        model_path = self.model_dir / f"{model_name}.joblib"
        scaler_path = self.model_dir / f"{model_name}_scaler.joblib"
        config_path = self.model_dir / f"{model_name}_config.json"
        
        if model_path.exists():
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.feature_columns = config['feature_columns']
            logger.info(f"Config loaded from {config_path}")


def main():
    """Main execution function for testing"""
    # Create detector
    detector = SalaryAnomalyDetector()
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'title': [
            'Senior Software Engineer',
            'Junior Developer',
            'Data Scientist',
            'Easy Money Maker',
            'Marketing Manager',
            'Quick Cash Job'
        ],
        'salary_range': [
            '$120,000 - $150,000',
            '$60,000 - $75,000',
            '$90,000 - $110,000',
            '$5000/week',
            '$80,000 - $95,000',
            '$3000/day'
        ],
        'description': [
            'Senior position with 5+ years experience',
            'Entry level position',
            'Data science role with ML experience',
            'Work from home easy money',
            'Marketing role at established company',
            'Quick cash no experience needed'
        ]
    })
    
    print("=== Salary Anomaly Detection Test ===")
    
    # Train model
    if SKLEARN_AVAILABLE:
        results = detector.train(sample_data)
        print(f"Training results: {results}")
    else:
        print("sklearn not available, using heuristic detection")
    
    # Test individual detection
    test_salaries = [
        "$120,000 - $150,000",
        "$5000/week",
        "$3000/day",
        "$25/hour",
        "$15/hour"
    ]
    
    print("\n=== Individual Detection Tests ===")
    for salary in test_salaries:
        result = detector.detect_anomaly(salary, description="Test job description")
        print(f"Salary: {salary}")
        print(f"  Is Anomaly: {result['is_anomaly']}")
        print(f"  Anomaly Probability: {result['anomaly_probability']:.2f}")
        print(f"  Method: {result['method']}")
        if 'reasons' in result:
            print(f"  Reasons: {result['reasons']}")
        print()


if __name__ == "__main__":
    main()
