"""
Baseline Models Module
Implements traditional ML models: Logistic Regression, Random Forest, XGBoost
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, List
import joblib
import json

# ML imports
try:
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                f1_score, roc_auc_score, confusion_matrix, 
                                classification_report, roc_curve, auc)
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaselineModelTrainer:
    """Train and evaluate baseline ML models"""
    
    def __init__(self, model_dir: str = "data/models"):
        """
        Initialize the baseline model trainer
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self.results = {}
        
    def prepare_data(self, df: pd.DataFrame, text_column: str = 'description') -> Tuple:
        """
        Prepare data for training
        
        Args:
            df: Input dataframe
            text_column: Name of text column to use for TF-IDF
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, tfidf_vectorizer)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for baseline models")
        
        logger.info("Preparing data for training...")
        
        # Extract text and labels
        texts = df[text_column].fillna('').values
        labels = df['fraudulent'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        # Fit and transform
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        
        logger.info(f"Data prepared: Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"TF-IDF features: {X_train_tfidf.shape[1]}")
        
        return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test) -> Dict:
        """
        Train Logistic Regression model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with model and results
        """
        logger.info("Training Logistic Regression...")
        
        # Create and train model
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            C=1.0
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        logger.info(f"Logistic Regression - Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")
        
        # Store model
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = results
        
        return {
            'model': model,
            'results': results,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def train_random_forest(self, X_train, y_train, X_test, y_test) -> Dict:
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with model and results
        """
        logger.info("Training Random Forest...")
        
        # Create and train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Get feature importance
        feature_importance = model.feature_importances_
        
        logger.info(f"Random Forest - Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")
        
        # Store model
        self.models['random_forest'] = model
        self.results['random_forest'] = results
        
        return {
            'model': model,
            'results': results,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'feature_importance': feature_importance
        }
    
    def train_xgboost(self, X_train, y_train, X_test, y_test) -> Dict:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with model and results
        """
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, skipping...")
            return None
        
        logger.info("Training XGBoost...")
        
        # Convert to DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Parameters
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'eta': 0.1,
            'scale_pos_weight': 1,
            'eval_metric': 'logloss',
            'random_state': 42
        }
        
        # Train model
        model = xgb.train(params, dtrain, num_boost_round=100)
        
        # Make predictions
        y_pred_proba = model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        results = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        logger.info(f"XGBoost - Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")
        
        # Store model
        self.models['xgboost'] = model
        self.results['xgboost'] = results
        
        return {
            'model': model,
            'results': results,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba) -> Dict:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        return metrics
    
    def train_all_models(self, df: pd.DataFrame, text_column: str = 'description') -> Dict:
        """
        Train all baseline models and compare results
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            
        Returns:
            Dictionary with all model results
        """
        logger.info("Starting baseline model training...")
        
        # Prepare data
        X_train, X_test, y_train, y_test, tfidf_vectorizer = self.prepare_data(df, text_column)
        self.vectorizers['tfidf'] = tfidf_vectorizer
        
        # Train models
        results = {}
        
        # Logistic Regression
        lr_results = self.train_logistic_regression(X_train, y_train, X_test, y_test)
        results['logistic_regression'] = lr_results
        
        # Random Forest
        rf_results = self.train_random_forest(X_train, y_train, X_test, y_test)
        results['random_forest'] = rf_results
        
        # XGBoost
        xgb_results = self.train_xgboost(X_train, y_train, X_test, y_test)
        if xgb_results:
            results['xgboost'] = xgb_results
        
        # Compare results
        comparison = self.compare_models(results)
        
        logger.info("Baseline model training completed!")
        logger.info(f"Model comparison:\n{comparison}")
        
        return {
            'results': results,
            'comparison': comparison,
            'vectorizer': tfidf_vectorizer
        }
    
    def compare_models(self, results: Dict) -> pd.DataFrame:
        """
        Compare performance of all models
        
        Args:
            results: Dictionary of model results
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for model_name, model_results in results.items():
            if model_results is None:
                continue
                
            metrics = model_results['results']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'ROC-AUC': metrics['roc_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        return comparison_df
    
    def save_models(self, model_names: Optional[List[str]] = None) -> Dict[str, Path]:
        """
        Save trained models to disk
        
        Args:
            model_names: List of model names to save (None = save all)
            
        Returns:
            Dictionary mapping model names to saved paths
        """
        saved_paths = {}
        
        models_to_save = model_names if model_names else list(self.models.keys())
        
        for model_name in models_to_save:
            if model_name not in self.models:
                continue
                
            model_path = self.model_dir / f"{model_name}.joblib"
            joblib.dump(self.models[model_name], model_path)
            saved_paths[model_name] = model_path
            
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save vectorizer
        if 'tfidf' in self.vectorizers:
            vectorizer_path = self.model_dir / "tfidf_vectorizer.joblib"
            joblib.dump(self.vectorizers['tfidf'], vectorizer_path)
            saved_paths['vectorizer'] = vectorizer_path
        
        # Save results
        if self.results:
            results_path = self.model_dir / "baseline_results.json"
            # Convert numpy types for JSON serialization
            json_results = {}
            for model_name, metrics in self.results.items():
                json_results[model_name] = self._convert_to_json_serializable(metrics)
            
            with open(results_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            saved_paths['results'] = results_path
        
        return saved_paths
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def load_model(self, model_name: str):
        """
        Load a trained model from disk
        
        Args:
            model_name: Name of model to load
            
        Returns:
            Loaded model
        """
        model_path = self.model_dir / f"{model_name}.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"Loaded {model_name} from {model_path}")
        
        return model
    
    def predict(self, model_name: str, text: str) -> Dict:
        """
        Make prediction using a trained model
        
        Args:
            model_name: Name of model to use
            text: Input text to classify
            
        Returns:
            Dictionary with prediction and probability
        """
        # Load model
        model = self.load_model(model_name)
        
        # Load vectorizer
        vectorizer_path = self.model_dir / "tfidf_vectorizer.joblib"
        vectorizer = joblib.load(vectorizer_path)
        
        # Transform text
        text_tfidf = vectorizer.transform([text])
        
        # Make prediction
        prediction = model.predict(text_tfidf)[0]
        
        # Get probability
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(text_tfidf)[0, 1]
        else:
            # For XGBoost
            import xgboost as xgb
            dmatrix = xgb.DMatrix(text_tfidf)
            probability = model.predict(dmatrix)[0]
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'is_fake': bool(prediction == 1),
            'confidence': float(probability if prediction == 1 else 1 - probability)
        }


def main():
    """Main execution function for testing"""
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'description': [
            'Senior Software Engineer position requiring 5+ years Python experience. Competitive salary and benefits.',
            'URGENT! Earn $5000 weekly working from home. No experience needed. Start today!',
            'Marketing Manager at TechCorp. MBA required. $80,000-$95,000 salary.',
            'Easy money! Work from anywhere and earn $10000 per month with zero investment.',
            'Data Analyst position with SQL, Python skills. 2+ years experience. $70,000 annually.',
            'Immediate hiring! High paying job with no qualifications required.',
            'DevOps Engineer with AWS, Kubernetes experience. $100,000-$125,000.',
            'Get rich overnight! Investment opportunity with 100% returns guaranteed.',
            'UX Designer position. Portfolio required. $75,000-$90,000.',
            'Work from home and earn $8000 weekly. Limited spots available!'
        ],
        'fraudulent': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    
    # Train models
    trainer = BaselineModelTrainer()
    results = trainer.train_all_models(sample_data)
    
    # Print comparison
    print("\n=== Model Comparison ===")
    print(results['comparison'].to_string())
    
    # Save models
    saved_paths = trainer.save_models()
    print(f"\nModels saved to: {saved_paths}")
    
    # Test prediction
    test_text = "Earn $3000 daily working from home. No experience needed. Apply now!"
    prediction = trainer.predict('logistic_regression', test_text)
    print(f"\nTest prediction: {prediction}")


if __name__ == "__main__":
    main()
