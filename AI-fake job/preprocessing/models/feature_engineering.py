"""
Feature Engineering Module
Creates advanced features for fake job detection
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced feature engineering for fake job detection"""
    
    def __init__(self):
        """Initialize feature engineer with scam indicators"""
        
        # Scam keyword lists
        self.urgency_keywords = [
            'urgent', 'immediate', 'asap', 'today', 'now', 'hurry',
            'limited time', 'ending soon', 'last chance', 'don\'t miss',
            'immediately', 'instant', 'quick', 'fast', 'speed'
        ]
        
        self.scam_phrases = [
            'no experience', 'no qualification', 'work from home', 'easy money',
            'quick cash', 'earn weekly', 'daily income', 'weekly income',
            'investment opportunity', 'guaranteed income', 'risk free',
            'limited spots', 'no interview', 'hire now', 'apply now',
            'click here', 'telegram', 'whatsapp', 'get rich', 'overnight'
        ]
        
        self.suspicious_patterns = [
            r'\$\d+,?\d*',  # Dollar amounts
            r'earn\s+\$\d+',  # Earn money patterns
            r'\d+\s*(dollars|usd|eur|gbp)',  # Currency patterns
            r'bitcoins?',  # Cryptocurrency
            r'crypto',  # Cryptocurrency
            r'western union',  # Money transfer
            r'moneygram',  # Money transfer
            r'wire transfer',  # Money transfer
            r'gift card',  # Payment method
            r'itunes card',  # Payment method
            r'amazon card',  # Payment method
        ]
        
        # Free email providers
        self.free_email_providers = [
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
            'aol.com', 'mail.com', 'protonmail.com', 'zoho.com'
        ]
        
        # Legitimate company indicators
        self.legitimate_indicators = [
            'linkedin', 'indeed', 'glassdoor', 'career', 'careers',
            'recruitment', 'hr', 'human resources', 'benefits',
            'insurance', '401k', 'stock options', 'equity'
        ]
        
        # Technical skills (indicates real jobs)
        self.technical_skills = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'sql',
            'nosql', 'mongodb', 'postgresql', 'mysql', 'machine learning',
            'deep learning', 'ai', 'artificial intelligence', 'data science'
        ]
        
        # Educational requirements
        self.education_keywords = [
            'bachelor', 'master', 'phd', 'degree', 'university',
            'college', 'certification', 'certified', 'diploma'
        ]
        
    def calculate_scam_keyword_frequency(self, text: str) -> float:
        """Calculate frequency of scam keywords in text"""
        text_lower = text.lower()
        scam_count = sum(1 for phrase in self.scam_phrases if phrase in text_lower)
        word_count = len(text.split())
        
        if word_count == 0:
            return 0.0
        
        return scam_count / word_count
    
    def calculate_urgency_score(self, text: str) -> int:
        """Calculate urgency score based on urgency keywords"""
        text_lower = text.lower()
        urgency_count = sum(1 for word in self.urgency_keywords if word in text_lower)
        return urgency_count
    
    def detect_suspicious_patterns(self, text: str) -> int:
        """Detect suspicious patterns in text"""
        pattern_count = 0
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text.lower()):
                pattern_count += 1
        return pattern_count
    
    def calculate_salary_realism_score(self, salary_str: str, job_title: str = "") -> float:
        """
        Calculate salary realism score (0-1, where 1 is realistic)
        
        Args:
            salary_str: Salary string or range
            job_title: Job title for context
            
        Returns:
            Realism score between 0 and 1
        """
        if pd.isna(salary_str) or salary_str == '':
            return 0.5  # Neutral score for missing salary
        
        # Extract numeric values
        numbers = re.findall(r'\d+', str(salary_str))
        if not numbers:
            return 0.5
        
        salary_values = [int(num) for num in numbers]
        avg_salary = sum(salary_values) / len(salary_values)
        
        # Define realistic ranges (simplified)
        # Very low salaries (<$10) are suspicious
        # Very high weekly salaries (>$2000) are suspicious
        # Very high hourly rates (>$100) are suspicious unless senior role
        
        realism_score = 1.0
        
        # Check for unrealistically low salaries
        if avg_salary < 10:
            realism_score -= 0.5
        
        # Check for unrealistically high weekly amounts
        if 'week' in str(salary_str).lower() and avg_salary > 2000:
            realism_score -= 0.6
        
        # Check for unrealistically high daily amounts
        if 'day' in str(salary_str).lower() and avg_salary > 500:
            realism_score -= 0.6
        
        # Check for hourly rates
        if 'hour' in str(salary_str).lower():
            if avg_salary > 100 and 'senior' not in job_title.lower():
                realism_score -= 0.4
            elif avg_salary < 5:
                realism_score -= 0.3
        
        # Check for annual salaries
        if 'year' in str(salary_str).lower() or 'annual' in str(salary_str).lower():
            if avg_salary > 500000:  # Extremely high
                realism_score -= 0.3
            elif avg_salary < 15000:  # Extremely low
                realism_score -= 0.3
        
        return max(0.0, min(1.0, realism_score))
    
    def calculate_company_profile_completeness(self, row: pd.Series) -> float:
        """
        Calculate company profile completeness score (0-1)
        
        Args:
            row: DataFrame row with company information
            
        Returns:
            Completeness score
        """
        score = 0.0
        total_features = 0
        
        # Check for company logo
        if 'has_company_logo' in row:
            total_features += 1
            if row['has_company_logo'] == 1:
                score += 0.3
        
        # Check for company profile
        if 'company_profile' in row:
            total_features += 1
            if pd.notna(row['company_profile']) and row['company_profile'] != '':
                profile_length = len(str(row['company_profile']))
                if profile_length > 50:
                    score += 0.4
                elif profile_length > 20:
                    score += 0.2
        
        # Check for questions
        if 'has_questions' in row:
            total_features += 1
            if row['has_questions'] == 1:
                score += 0.3
        
        if total_features > 0:
            return score / total_features
        return 0.0
    
    def calculate_requirement_complexity(self, text: str) -> float:
        """
        Calculate requirement complexity score (0-1)
        Higher complexity suggests legitimate job
        """
        if pd.isna(text) or text == '':
            return 0.0
        
        text_lower = text.lower()
        
        # Count technical skills mentioned
        tech_skill_count = sum(1 for skill in self.technical_skills if skill in text_lower)
        
        # Count education requirements
        edu_count = sum(1 for edu in self.education_keywords if edu in text_lower)
        
        # Count years of experience mentioned
        experience_pattern = r'\d+\+?\s*(years?|yrs?)'
        experience_matches = len(re.findall(experience_pattern, text_lower))
        
        # Calculate complexity score
        complexity_score = 0.0
        complexity_score += min(tech_skill_count * 0.2, 0.4)  # Max 0.4 for skills
        complexity_score += min(edu_count * 0.15, 0.3)  # Max 0.3 for education
        complexity_score += min(experience_matches * 0.15, 0.3)  # Max 0.3 for experience
        
        return min(complexity_score, 1.0)
    
    def calculate_benefit_richness(self, text: str) -> float:
        """
        Calculate benefit richness score (0-1)
        More benefits suggest legitimate job
        """
        if pd.isna(text) or text == '':
            return 0.0
        
        text_lower = text.lower()
        
        benefit_keywords = [
            'insurance', 'health', 'dental', 'vision', '401k', 'retirement',
            'bonus', 'stock', 'equity', 'vacation', 'paid time off',
            'sick leave', 'parental leave', 'remote', 'flexible',
            'training', 'education', 'gym', 'wellness'
        ]
        
        benefit_count = sum(1 for keyword in benefit_keywords if keyword in text_lower)
        
        return min(benefit_count * 0.15, 1.0)
    
    def calculate_contact_quality(self, text: str) -> float:
        """
        Calculate contact quality score (0-1)
        Professional contact methods suggest legitimacy
        """
        if pd.isna(text) or text == '':
            return 0.0
        
        text_lower = text.lower()
        
        score = 0.0
        
        # Check for professional email domains
        professional_domains = ['.com', '.org', '.net', '.io', '.co']
        has_professional_email = any(domain in text_lower for domain in professional_domains)
        
        # Check for free email providers (suspicious)
        has_free_email = any(provider in text_lower for provider in self.free_email_providers)
        
        # Check for LinkedIn presence (legitimate)
        has_linkedin = 'linkedin' in text_lower
        
        # Check for company website pattern
        has_website = re.search(r'www\.[a-zA-Z0-9-]+\.[a-zA-Z]{2,}', text_lower) is not None
        
        if has_professional_email and not has_free_email:
            score += 0.4
        if has_linkedin:
            score += 0.3
        if has_website:
            score += 0.3
        
        return min(score, 1.0)
    
    def calculate_grammar_quality(self, text: str) -> float:
        """
        Calculate grammar quality score (0-1)
        Poor grammar often indicates fake jobs
        """
        if pd.isna(text) or text == '':
            return 0.5
        
        # Simple grammar checks
        errors = 0
        total_checks = 0
        
        # Check for multiple consecutive spaces
        if re.search(r'\s{3,}', text):
            errors += 1
        total_checks += 1
        
        # Check for multiple consecutive exclamation marks
        if re.search(r'!{2,}', text):
            errors += 1
        total_checks += 1
        
        # Check for all caps words (except acronyms)
        all_caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
        if len(all_caps_words) > 3:  # More than 3 all-caps words is suspicious
            errors += 1
        total_checks += 1
        
        # Check for sentence structure (basic)
        sentences = re.split(r'[.!?]+', text)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sentence_length < 3 or avg_sentence_length > 50:
                errors += 1
            total_checks += 1
        
        if total_checks == 0:
            return 0.5
        
        quality_score = 1.0 - (errors / total_checks)
        return max(0.0, quality_score)
    
    def extract_salary_features(self, salary_str: str) -> Dict[str, float]:
        """
        Extract various salary-related features
        
        Args:
            salary_str: Salary string
            
        Returns:
            Dictionary of salary features
        """
        features = {
            'has_salary': 0.0,
            'salary_min': 0.0,
            'salary_max': 0.0,
            'salary_avg': 0.0,
            'is_hourly': 0.0,
            'is_weekly': 0.0,
            'is_monthly': 0.0,
            'is_annual': 0.0,
            'salary_range_width': 0.0
        }
        
        if pd.isna(salary_str) or salary_str == '':
            return features
        
        features['has_salary'] = 1.0
        
        # Extract numeric values
        numbers = re.findall(r'\d+', str(salary_str))
        if numbers:
            salary_values = [int(num) for num in numbers]
            features['salary_min'] = min(salary_values)
            features['salary_max'] = max(salary_values)
            features['salary_avg'] = sum(salary_values) / len(salary_values)
            features['salary_range_width'] = features['salary_max'] - features['salary_min']
        
        # Detect time period
        salary_lower = str(salary_str).lower()
        if 'hour' in salary_lower:
            features['is_hourly'] = 1.0
        elif 'week' in salary_lower:
            features['is_weekly'] = 1.0
        elif 'month' in salary_lower:
            features['is_monthly'] = 1.0
        elif 'year' in salary_lower or 'annual' in salary_lower:
            features['is_annual'] = 1.0
        
        return features
    
    def calculate_text_readability(self, text: str) -> Dict[str, float]:
        """
        Calculate text readability features
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of readability features
        """
        if pd.isna(text) or text == '':
            return {
                'avg_word_length': 0.0,
                'avg_sentence_length': 0.0,
                'unique_word_ratio': 0.0,
                'punctuation_ratio': 0.0
            }
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        if not words:
            return {
                'avg_word_length': 0.0,
                'avg_sentence_length': 0.0,
                'unique_word_ratio': 0.0,
                'punctuation_ratio': 0.0
            }
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Average sentence length
        valid_sentences = [s for s in sentences if s.strip()]
        if valid_sentences:
            avg_sentence_length = sum(len(s.split()) for s in valid_sentences) / len(valid_sentences)
        else:
            avg_sentence_length = len(words)
        
        # Unique word ratio
        unique_words = set(word.lower() for word in words)
        unique_word_ratio = len(unique_words) / len(words)
        
        # Punctuation ratio
        punctuation_count = sum(1 for char in text if char in '.,!?;:')
        punctuation_ratio = punctuation_count / len(text) if text else 0
        
        return {
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'unique_word_ratio': unique_word_ratio,
            'punctuation_ratio': punctuation_ratio
        }
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering to dataframe
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with engineered features
        """
        logger.info("Starting feature engineering...")
        
        # Text-based features
        df['scam_keyword_freq'] = df['description'].apply(self.calculate_scam_keyword_frequency)
        df['urgency_score'] = df['description'].apply(self.calculate_urgency_score)
        df['suspicious_patterns'] = df['description'].apply(self.detect_suspicious_patterns)
        
        # Salary features
        df['salary_realism_score'] = df.apply(
            lambda row: self.calculate_salary_realism_score(row['salary_range'], row['title']),
            axis=1
        )
        
        # Company features
        df['company_profile_completeness'] = df.apply(
            self.calculate_company_profile_completeness,
            axis=1
        )
        
        # Requirement features
        if 'requirements' in df.columns:
            df['requirement_complexity'] = df['requirements'].apply(self.calculate_requirement_complexity)
        else:
            df['requirement_complexity'] = df['description'].apply(self.calculate_requirement_complexity)
        
        # Benefit features
        if 'benefits' in df.columns:
            df['benefit_richness'] = df['benefits'].apply(self.calculate_benefit_richness)
        else:
            df['benefit_richness'] = 0.0
        
        # Contact quality
        df['contact_quality'] = df['description'].apply(self.calculate_contact_quality)
        
        # Grammar quality
        df['grammar_quality'] = df['description'].apply(self.calculate_grammar_quality)
        
        # Readability features
        readability_features = df['description'].apply(self.calculate_text_readability).apply(pd.Series)
        df = pd.concat([df, readability_features], axis=1)
        
        # Detailed salary features
        salary_features = df['salary_range'].apply(self.extract_salary_features).apply(pd.Series)
        df = pd.concat([df, salary_features], axis=1)
        
        # Combined fraud probability (simple heuristic)
        df['heuristic_fraud_score'] = (
            df['scam_keyword_freq'] * 0.3 +
            df['urgency_score'] * 0.2 +
            (1 - df['salary_realism_score']) * 0.2 +
            (1 - df['company_profile_completeness']) * 0.15 +
            (1 - df['contact_quality']) * 0.15
        )
        
        logger.info(f"Feature engineering completed. Total features: {len(df.columns)}")
        
        return df
    
    def get_feature_importance_heuristic(self) -> Dict[str, float]:
        """
        Return heuristic feature importance weights
        
        Returns:
            Dictionary mapping feature names to importance weights
        """
        return {
            'scam_keyword_freq': 0.25,
            'urgency_score': 0.20,
            'salary_realism_score': 0.15,
            'company_profile_completeness': 0.15,
            'contact_quality': 0.10,
            'grammar_quality': 0.05,
            'requirement_complexity': 0.05,
            'benefit_richness': 0.05
        }


if __name__ == "__main__":
    # Test the feature engineer
    engineer = FeatureEngineer()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'title': ['Senior Software Engineer', 'Easy Money Maker'],
        'description': [
            'We are looking for a senior software engineer with 5+ years of experience in Python, AWS, and Docker. Competitive salary and benefits package.',
            'URGENT! Earn $5000 weekly working from home. No experience needed. Start today! Limited spots available.'
        ],
        'salary_range': ['$120,000 - $150,000', '$5000/week'],
        'requirements': ['Python, AWS, Docker, 5+ years experience', 'None'],
        'benefits': ['Health insurance, 401k, stock options', 'None'],
        'company_profile': ['Tech company with 500 employees', ''],
        'has_company_logo': [1, 0],
        'has_questions': [1, 0]
    })
    
    # Engineer features
    engineered_df = engineer.engineer_features(sample_data)
    
    print("Engineered Features:")
    print(engineered_df[['title', 'scam_keyword_freq', 'urgency_score', 'salary_realism_score', 
                        'company_profile_completeness', 'heuristic_fraud_score']].to_string())
