"""
SHAP Explainable AI Module
Implements SHAP-based explanations for highlighting suspicious phrases in job descriptions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import json
import re
import joblib

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP-based explainer for fake job detection"""
    
    def __init__(self, model_dir: str = "data/models"):
        """
        Initialize SHAP explainer
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        self.explainer = None
        self.model = None
        self.vectorizer = None
        
        # Load model and vectorizer
        self._load_components()
        
        # Suspicious phrase patterns
        self.suspicious_patterns = [
            r'earn\s+\$?\d+',
            r'\$?\d+\s*(weekly|daily|hourly)',
            r'no experience',
            r'no qualification',
            r'work from home',
            r'easy money',
            r'quick cash',
            r'urgent',
            r'immediate',
            r'asap',
            r'limited spots',
            r'no interview',
            r'get rich',
            r'overnight',
            r'guaranteed',
            r'risk.?free',
            r'investment',
            r'telegram',
            r'whatsapp',
            r'gmail\.com',
            r'yahoo\.com',
            r'hotmail\.com'
        ]
        
        logger.info("SHAP Explainer initialized")
    
    def _load_components(self):
        """Load model and vectorizer for SHAP explanation"""
        try:
            # Load baseline model
            lr_path = self.model_dir / "logistic_regression.joblib"
            if lr_path.exists():
                self.model = joblib.load(lr_path)
                logger.info("Loaded model for SHAP explanation")
            
            # Load vectorizer
            tfidf_path = self.model_dir / "tfidf_vectorizer.joblib"
            if tfidf_path.exists():
                self.vectorizer = joblib.load(tfidf_path)
                logger.info("Loaded vectorizer for SHAP explanation")
                
        except Exception as e:
            logger.warning(f"Error loading components for SHAP: {e}")
    
    def initialize_explainer(self, background_data: Optional[pd.DataFrame] = None):
        """
        Initialize SHAP explainer with background data
        
        Args:
            background_data: Background data for SHAP explainer
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, using rule-based explanation instead")
            return
        
        if self.model is None or self.vectorizer is None:
            logger.warning("Model or vectorizer not loaded, cannot initialize SHAP")
            return
        
        try:
            # Create background data if not provided
            if background_data is None:
                # Use sample data as background
                sample_texts = [
                    "Senior software engineer position requiring Python experience",
                    "Urgent hiring earn money working from home no experience needed",
                    "Marketing manager at tech company MBA required competitive salary",
                    "Easy money work from anywhere earn weekly with zero investment"
                ]
                background_data = self.vectorizer.transform(sample_texts)
            else:
                background_data = self.vectorizer.transform(background_data['description'].fillna(''))
            
            # Initialize explainer
            self.explainer = shap.LinearExplainer(self.model, background_data)
            logger.info("SHAP explainer initialized successfully")
            
        except Exception as e:
            logger.warning(f"Error initializing SHAP explainer: {e}")
    
    def explain_with_shap(self, text: str, num_features: int = 10) -> Dict:
        """
        Explain prediction using SHAP values
        
        Args:
            text: Input text to explain
            num_features: Number of top features to return
            
        Returns:
            Dictionary with SHAP explanation
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return self.explain_with_rules(text)
        
        try:
            # Transform text
            text_tfidf = self.vectorizer.transform([text])
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(text_tfidf)
            
            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get top features
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # For binary classification
            
            # Get absolute SHAP values
            abs_shap = np.abs(shap_values[0])
            
            # Get top indices
            top_indices = np.argsort(abs_shap)[-num_features:][::-1]
            
            # Extract phrases
            explanations = []
            for idx in top_indices:
                feature_name = feature_names[idx]
                shap_value = shap_values[0][idx]
                
                # Find the phrase in text
                phrase = self._find_phrase_in_text(text, feature_name)
                
                explanations.append({
                    'phrase': phrase,
                    'feature': feature_name,
                    'shap_value': float(shap_value),
                    'importance': float(abs_shap[idx]),
                    'contribution': 'increases_fake_probability' if shap_value > 0 else 'decreases_fake_probability'
                })
            
            return {
                'method': 'shap',
                'explanations': explanations,
                'top_suspicious_phrases': [e['phrase'] for e in explanations if e['shap_value'] > 0]
            }
            
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}, using rule-based")
            return self.explain_with_rules(text)
    
    def explain_with_rules(self, text: str) -> Dict:
        """
        Explain prediction using rule-based approach (fallback)
        
        Args:
            text: Input text to explain
            
        Returns:
            Dictionary with rule-based explanation
        """
        text_lower = text.lower()
        
        explanations = []
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                phrase = match.group()
                start_pos = match.start()
                end_pos = match.end()
                
                # Calculate importance based on pattern type
                importance = self._calculate_pattern_importance(phrase)
                
                explanations.append({
                    'phrase': phrase,
                    'pattern': pattern,
                    'position': (start_pos, end_pos),
                    'importance': importance,
                    'contribution': 'increases_fake_probability'
                })
        
        # Sort by importance
        explanations.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            'method': 'rule_based',
            'explanations': explanations[:15],  # Top 15 explanations
            'top_suspicious_phrases': [e['phrase'] for e in explanations[:10]]
        }
    
    def _find_phrase_in_text(self, text: str, feature: str) -> str:
        """
        Find the phrase corresponding to a feature in the text
        
        Args:
            text: Original text
            feature: Feature name from TF-IDF
            
        Returns:
            Phrase from text
        """
        # Simple approach: find the feature in text
        text_lower = text.lower()
        feature_lower = feature.lower()
        
        if feature_lower in text_lower:
            # Find the phrase with some context
            start = text_lower.find(feature_lower)
            end = start + len(feature)
            
            # Add some context (up to 20 chars on each side)
            context_start = max(0, start - 20)
            context_end = min(len(text), end + 20)
            
            return text[context_start:context_end]
        
        return feature
    
    def _calculate_pattern_importance(self, phrase: str) -> float:
        """
        Calculate importance score for a suspicious phrase
        
        Args:
            phrase: Suspicious phrase
            
        Returns:
            Importance score between 0 and 1
        """
        phrase_lower = phrase.lower()
        
        # High importance patterns
        high_importance = ['earn', '$', 'urgent', 'immediate', 'no experience', 'guaranteed']
        medium_importance = ['work from home', 'easy money', 'telegram', 'whatsapp', 'gmail.com']
        
        for word in high_importance:
            if word in phrase_lower:
                return 0.9
        
        for word in medium_importance:
            if word in phrase_lower:
                return 0.7
        
        return 0.5
    
    def highlight_suspicious_phrases(self, text: str, explanation: Optional[Dict] = None) -> str:
        """
        Highlight suspicious phrases in text with HTML markup
        
        Args:
            text: Original text
            explanation: Explanation dictionary (optional)
            
        Returns:
            Text with HTML highlights
        """
        if explanation is None:
            explanation = self.explain_with_shap(text)
        
        if explanation['method'] == 'shap':
            return self._highlight_shap_explanations(text, explanation)
        else:
            return self._highlight_rule_explanations(text, explanation)
    
    def _highlight_shap_explanations(self, text: str, explanation: Dict) -> str:
        """Highlight text based on SHAP explanations"""
        highlighted_text = text
        
        # Sort explanations by position (reverse to avoid offset issues)
        explanations = sorted(explanation['explanations'], 
                            key=lambda x: text.lower().find(x['phrase'].lower()), 
                            reverse=True)
        
        for exp in explanations:
            phrase = exp['phrase']
            shap_value = exp['shap_value']
            
            # Determine color based on SHAP value
            if shap_value > 0:
                # Red for increasing fake probability
                color = '#ff6b6b'
                intensity = min(abs(shap_value) * 2, 1)  # Cap at 1
            else:
                # Green for decreasing fake probability
                color = '#51cf66'
                intensity = min(abs(shap_value) * 2, 1)
            
            # Create highlight
            highlight = f'<mark style="background-color: {color}; opacity: {intensity}; padding: 2px 4px; border-radius: 3px;">{phrase}</mark>'
            
            # Replace in text (case-insensitive)
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            highlighted_text = pattern.sub(highlight, highlighted_text, count=1)
        
        return highlighted_text
    
    def _highlight_rule_explanations(self, text: str, explanation: Dict) -> str:
        """Highlight text based on rule-based explanations"""
        highlighted_text = text
        
        # Sort explanations by position (reverse to avoid offset issues)
        explanations = sorted(explanation['explanations'], 
                            key=lambda x: x['position'][0], 
                            reverse=True)
        
        for exp in explanations:
            phrase = exp['phrase']
            importance = exp['importance']
            start, end = exp['position']
            
            # Determine color based on importance
            color = '#ff6b6b'  # Red for suspicious
            intensity = importance
            
            # Create highlight
            highlight = f'<mark style="background-color: {color}; opacity: {intensity}; padding: 2px 4px; border-radius: 3px;">{phrase}</mark>'
            
            # Replace in text
            highlighted_text = highlighted_text[:start] + highlight + highlighted_text[end:]
        
        return highlighted_text
    
    def get_feature_importance(self, background_data: pd.DataFrame) -> Dict:
        """
        Get overall feature importance from SHAP
        
        Args:
            background_data: Background data for analysis
            
        Returns:
            Dictionary with feature importance
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return self._get_rule_based_importance()
        
        try:
            # Transform background data
            background_tfidf = self.vectorizer.transform(background_data['description'].fillna(''))
            
            # Get SHAP values for all samples
            shap_values = self.explainer.shap_values(background_tfidf)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Create importance dictionary
            importance = {}
            for idx, feature_name in enumerate(feature_names):
                importance[feature_name] = float(mean_abs_shap[idx])
            
            # Sort by importance
            sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            return {
                'method': 'shap',
                'feature_importance': sorted_importance
            }
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
            return self._get_rule_based_importance()
    
    def _get_rule_based_importance(self) -> Dict:
        """Get rule-based feature importance (fallback)"""
        importance = {
            'earn': 0.9,
            'money': 0.8,
            'urgent': 0.85,
            'immediate': 0.8,
            'experience': 0.7,
            'weekly': 0.75,
            'daily': 0.75,
            'investment': 0.8,
            'guaranteed': 0.85,
            'telegram': 0.9,
            'whatsapp': 0.85,
            'gmail.com': 0.7,
            'work from home': 0.8,
            'no experience': 0.9,
            'easy money': 0.85
        }
        
        return {
            'method': 'rule_based',
            'feature_importance': importance
        }
    
    def generate_explanation_report(self, text: str, prediction_result: Dict) -> Dict:
        """
        Generate comprehensive explanation report
        
        Args:
            text: Input text
            prediction_result: Prediction result from model
            
        Returns:
            Comprehensive explanation report
        """
        # Get SHAP explanation
        shap_explanation = self.explain_with_shap(text)
        
        # Highlight suspicious phrases
        highlighted_text = self.highlight_suspicious_phrases(text, shap_explanation)
        
        # Generate summary
        summary_parts = []
        
        if prediction_result.get('is_fake'):
            summary_parts.append("This job posting is classified as FAKE.")
        else:
            summary_parts.append("This job posting is classified as REAL.")
        
        # Add suspicious phrases
        suspicious_phrases = shap_explanation.get('top_suspicious_phrases', [])
        if suspicious_phrases:
            summary_parts.append(f"Suspicious phrases detected: {', '.join(suspicious_phrases[:5])}")
        
        # Add confidence
        confidence = prediction_result.get('confidence', 0)
        summary_parts.append(f"Prediction confidence: {confidence:.1%}")
        
        return {
            'prediction': prediction_result,
            'explanation_method': shap_explanation['method'],
            'suspicious_phrases': suspicious_phrases,
            'highlighted_text': highlighted_text,
            'detailed_explanations': shap_explanation['explanations'],
            'summary': ' '.join(summary_parts)
        }


def main():
    """Main execution function for testing"""
    # Create explainer
    explainer = SHAPExplainer()
    
    # Initialize explainer
    explainer.initialize_explainer()
    
    # Test explanation
    test_text = "URGENT! Earn $5000 weekly working from home. No experience needed. Start today! Limited spots available. Telegram: @quickmoney"
    
    print("=== SHAP Explanation Test ===")
    print(f"Original text: {test_text}\n")
    
    # Get explanation
    explanation = explainer.explain_with_shap(test_text)
    
    print(f"Explanation method: {explanation['method']}")
    print(f"\nTop suspicious phrases:")
    for phrase in explanation['top_suspicious_phrases']:
        print(f"  - {phrase}")
    
    print(f"\nDetailed explanations:")
    for exp in explanation['explanations'][:5]:
        print(f"  Phrase: {exp['phrase']}")
        print(f"  Importance: {exp.get('importance', exp.get('shap_value', 0)):.3f}")
        print(f"  Contribution: {exp.get('contribution', 'N/A')}")
        print()
    
    # Highlight text
    highlighted = explainer.highlight_suspicious_phrases(test_text, explanation)
    print(f"\nHighlighted text:\n{highlighted}")
    
    # Generate full report
    prediction_result = {
        'is_fake': True,
        'confidence': 0.85,
        'fraud_probability': 0.85
    }
    
    report = explainer.generate_explanation_report(test_text, prediction_result)
    print(f"\n=== Full Report ===")
    print(f"Summary: {report['summary']}")


if __name__ == "__main__":
    main()
