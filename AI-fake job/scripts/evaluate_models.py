"""
Model Evaluation Script
Comprehensive evaluation and comparison of all models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Optional sklearn imports
try:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                f1_score, roc_auc_score, confusion_matrix, 
                                classification_report, roc_curve, auc,
                                precision_recall_curve, average_precision_score)
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and comparison"""
    
    def __init__(self, model_dir: str = "data/models", output_dir: str = "docs"):
        """
        Initialize model evaluator
        
        Args:
            model_dir: Directory containing trained models
            output_dir: Directory to save evaluation results
        """
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.vectorizer = None
        self.evaluation_results = {}
        
        logger.info("Model Evaluator initialized")
    
    def load_models(self):
        """Load all trained models"""
        logger.info("Loading trained models...")
        
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
            
            # Load vectorizer
            tfidf_path = self.model_dir / "tfidf_vectorizer.joblib"
            if tfidf_path.exists():
                self.vectorizer = joblib.load(tfidf_path)
                logger.info("Loaded TF-IDF vectorizer")
                
        except Exception as e:
            logger.warning(f"Error loading models: {e}")
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict:
        """
        Evaluate a single model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1': f1_score(y_test, y_pred, average='binary'),
        }
        
        # ROC-AUC if probabilities available
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            metrics['avg_precision'] = average_precision_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def evaluate_all_models(self, test_df: pd.DataFrame) -> Dict:
        """
        Evaluate all loaded models
        
        Args:
            test_df: Test dataframe
            
        Returns:
            Dictionary of all evaluation results
        """
        if not SKLEARN_AVAILABLE:
            logger.error("sklearn not available for evaluation")
            return {}
        
        logger.info("Starting comprehensive model evaluation...")
        
        # Prepare test data
        texts = test_df['description'].fillna('').values
        labels = test_df['fraudulent'].values
        
        # Transform text
        if self.vectorizer:
            X_test = self.vectorizer.transform(texts)
        else:
            logger.error("Vectorizer not available")
            return {}
        
        # Evaluate each model
        for model_name, model in self.models.items():
            if model_name == 'tfidf_vectorizer':
                continue
                
            try:
                metrics = self.evaluate_model(model, X_test, labels, model_name)
                self.evaluation_results[model_name] = metrics
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
        
        # Generate comparison
        comparison = self.generate_comparison()
        
        return {
            'individual_results': self.evaluation_results,
            'comparison': comparison
        }
    
    def generate_comparison(self) -> pd.DataFrame:
        """Generate comparison dataframe"""
        comparison_data = []
        
        for model_name, metrics in self.evaluation_results.items():
            if 'error' in metrics:
                continue
                
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'ROC-AUC': metrics.get('roc_auc', 0),
                'Avg Precision': metrics.get('avg_precision', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        return comparison_df
    
    def plot_comparison(self, comparison_df: pd.DataFrame):
        """Plot model comparison charts"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Cannot plot - matplotlib/sklearn not available")
            return
        
        logger.info("Generating comparison plots...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Bar chart of metrics
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        comparison_df.set_index('Model')[metrics_to_plot].plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Performance Metrics Comparison')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: ROC-AUC comparison
        if 'ROC-AUC' in comparison_df.columns:
            comparison_df.plot(x='Model', y='ROC-AUC', kind='bar', ax=axes[0, 1], color='skyblue')
            axes[0, 1].set_title('ROC-AUC Comparison')
            axes[0, 1].set_ylabel('ROC-AUC Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Precision-Recall comparison
        if 'Avg Precision' in comparison_df.columns:
            comparison_df.plot(x='Model', y='Avg Precision', kind='bar', ax=axes[1, 0], color='lightcoral')
            axes[1, 0].set_title('Average Precision Comparison')
            axes[1, 0].set_ylabel('Average Precision')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Overall score (F1-Score)
        comparison_df.plot(x='Model', y='F1-Score', kind='bar', ax=axes[1, 1], color='lightgreen')
        axes[1, 1].set_title('F1-Score Comparison')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "model_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {plot_path}")
        
        plt.close()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Cannot plot confusion matrices - matplotlib/sklearn not available")
            return
        
        logger.info("Generating confusion matrices...")
        
        num_models = len(self.evaluation_results)
        fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5))
        
        if num_models == 1:
            axes = [axes]
        
        for idx, (model_name, metrics) in enumerate(self.evaluation_results.items()):
            if 'error' in metrics or 'confusion_matrix' not in metrics:
                continue
            
            cm = np.array(metrics['confusion_matrix'])
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name} Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "confusion_matrices.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrices saved to {plot_path}")
        
        plt.close()
    
    def generate_evaluation_report(self) -> str:
        """Generate comprehensive evaluation report"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("AI FAKE JOB DETECTOR - MODEL EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary
        report_lines.append("SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Total models evaluated: {len(self.evaluation_results)}")
        report_lines.append("")
        
        # Comparison table
        comparison = self.generate_comparison()
        report_lines.append("MODEL COMPARISON")
        report_lines.append("-" * 40)
        report_lines.append(comparison.to_string(index=False))
        report_lines.append("")
        
        # Detailed results for each model
        report_lines.append("DETAILED RESULTS")
        report_lines.append("-" * 40)
        
        for model_name, metrics in self.evaluation_results.items():
            if 'error' in metrics:
                continue
                
            report_lines.append(f"\n{model_name.upper()}")
            report_lines.append("-" * 30)
            report_lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
            report_lines.append(f"Precision: {metrics['precision']:.4f}")
            report_lines.append(f"Recall: {metrics['recall']:.4f}")
            report_lines.append(f"F1-Score: {metrics['f1']:.4f}")
            
            if 'roc_auc' in metrics:
                report_lines.append(f"ROC-AUC: {metrics['roc_auc']:.4f}")
            
            if 'avg_precision' in metrics:
                report_lines.append(f"Average Precision: {metrics['avg_precision']:.4f}")
            
            # Confusion matrix
            cm = metrics['confusion_matrix']
            report_lines.append(f"\nConfusion Matrix:")
            report_lines.append(f"  TN: {cm[0][0]}  FP: {cm[0][1]}")
            report_lines.append(f"  FN: {cm[1][0]}  TP: {cm[1][1]}")
        
        # Recommendations
        report_lines.append("\n" + "=" * 60)
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("=" * 60)
        
        if not comparison.empty:
            best_model = comparison.iloc[0]['Model']
            best_f1 = comparison.iloc[0]['F1-Score']
            
            report_lines.append(f"Best performing model: {best_model}")
            report_lines.append(f"Best F1-Score: {best_f1:.4f}")
            report_lines.append("")
            report_lines.append("Recommendation: Use the best performing model for production")
            report_lines.append("deployment. Consider ensemble methods for improved robustness.")
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_path = self.output_dir / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Evaluation report saved to {report_path}")
        
        return report_text
    
    def save_results(self):
        """Save evaluation results to JSON"""
        # Convert numpy types for JSON serialization
        json_results = {}
        for model_name, metrics in self.evaluation_results.items():
            json_results[model_name] = self._convert_to_json_serializable(metrics)
        
        # Save to JSON
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_path}")
    
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


def main():
    """Main execution function"""
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Load models
    evaluator.load_models()
    
    # Create test data (using cleaned dataset)
    try:
        test_df = pd.read_csv("data/processed/cleaned_dataset.csv")
        logger.info(f"Loaded test data with {len(test_df)} samples")
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        # Create sample test data
        test_df = pd.DataFrame({
            'description': [
                'Senior Software Engineer position with Python experience',
                'URGENT! Earn $5000 weekly working from home. No experience needed.',
                'Marketing Manager at TechCorp. MBA required.',
                'Easy money! Work from anywhere and earn $10000 per month.',
                'Data Analyst position with SQL skills. 2+ years experience.',
            ],
            'fraudulent': [0, 1, 0, 1, 0]
        })
        logger.info("Using sample test data")
    
    # Evaluate models
    if SKLEARN_AVAILABLE and evaluator.models:
        results = evaluator.evaluate_all_models(test_df)
        
        # Generate plots
        if results['comparison'] is not None and not results['comparison'].empty:
            evaluator.plot_comparison(results['comparison'])
            evaluator.plot_confusion_matrices()
        
        # Generate report
        report = evaluator.generate_evaluation_report()
        print(report)
        
        # Save results
        evaluator.save_results()
        
        logger.info("Model evaluation completed successfully!")
    else:
        logger.warning("Cannot evaluate - sklearn not available or no models loaded")


if __name__ == "__main__":
    main()
