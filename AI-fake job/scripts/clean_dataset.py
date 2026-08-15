"""
Data Cleaning Script
Handles missing values, removes duplicates, and balances classes
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple

# Optional imblearn imports
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCleaner:
    """Handles data cleaning and preprocessing"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self, filename: str = "initial_dataset.csv") -> pd.DataFrame:
        """Load dataset from processed directory"""
        filepath = self.processed_dir / filename
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        logger.info("Handling missing values...")
        
        initial_rows = len(df)
        
        # For text columns, fill with empty string
        text_columns = ['title', 'location', 'department', 'salary_range', 
                       'company_profile', 'description', 'requirements', 'benefits']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        # For categorical columns, fill with 'Not specified'
        categorical_columns = ['employment_type', 'required_experience', 
                              'required_education', 'industry', 'function']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Not specified')
        
        # For numeric columns, fill with 0 or median
        numeric_columns = ['telecommuting', 'has_company_logo', 'has_questions']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Remove rows where description is empty (critical feature)
        df = df[df['description'].str.len() > 0]
        
        logger.info(f"Removed {initial_rows - len(df)} rows with missing critical values")
        logger.info(f"Remaining rows: {len(df)}")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate job postings"""
        logger.info("Removing duplicates...")
        
        initial_rows = len(df)
        
        # Remove exact duplicates
        df = df.drop_duplicates()
        
        # Remove duplicates based on description (most important feature)
        df = df.drop_duplicates(subset=['description'], keep='first')
        
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        logger.info(f"Remaining rows: {len(df)}")
        
        return df
    
    def balance_classes(self, df: pd.DataFrame, method: str = 'smote') -> pd.DataFrame:
        """Balance the classes using various techniques"""
        logger.info(f"Balancing classes using {method}...")
        
        initial_counts = df['fraudulent'].value_counts()
        logger.info(f"Initial class distribution: {initial_counts.to_dict()}")
        
        if method == 'oversample':
            # Simple random oversampling
            fake_jobs = df[df['fraudulent'] == 1]
            real_jobs = df[df['fraudulent'] == 0]
            
            # Oversample minority class
            max_count = max(len(fake_jobs), len(real_jobs))
            
            if len(fake_jobs) < len(real_jobs):
                fake_jobs = fake_jobs.sample(max_count, replace=True, random_state=42)
            else:
                real_jobs = real_jobs.sample(max_count, replace=True, random_state=42)
            
            df_balanced = pd.concat([fake_jobs, real_jobs])
            
        elif method == 'undersample':
            # Simple random undersampling
            fake_jobs = df[df['fraudulent'] == 1]
            real_jobs = df[df['fraudulent'] == 0]
            
            # Undersample majority class
            min_count = min(len(fake_jobs), len(real_jobs))
            
            fake_jobs = fake_jobs.sample(min_count, random_state=42)
            real_jobs = real_jobs.sample(min_count, random_state=42)
            
            df_balanced = pd.concat([fake_jobs, real_jobs])
            
        elif method == 'smote':
            # SMOTE oversampling (requires numeric features)
            # For text data, we'll use a simpler approach
            fake_jobs = df[df['fraudulent'] == 1]
            real_jobs = df[df['fraudulent'] == 0]
            
            # Calculate target count (average of both)
            target_count = (len(fake_jobs) + len(real_jobs)) // 2
            
            # Oversample minority class to target
            if len(fake_jobs) < len(real_jobs):
                fake_jobs = fake_jobs.sample(target_count, replace=True, random_state=42)
                real_jobs = real_jobs.sample(target_count, random_state=42)
            else:
                real_jobs = real_jobs.sample(target_count, replace=True, random_state=42)
                fake_jobs = fake_jobs.sample(target_count, random_state=42)
            
            df_balanced = pd.concat([fake_jobs, real_jobs])
            
        else:
            # No balancing
            df_balanced = df.copy()
        
        # Shuffle the dataset
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        final_counts = df_balanced['fraudulent'].value_counts()
        logger.info(f"Final class distribution: {final_counts.to_dict()}")
        
        return df_balanced
    
    def add_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add text-based features for analysis"""
        logger.info("Adding text features...")
        
        # Text length features
        df['description_length'] = df['description'].str.len()
        df['title_length'] = df['title'].str.len()
        
        # Word count features
        df['description_word_count'] = df['description'].str.split().str.len()
        df['title_word_count'] = df['title'].str.split().str.len()
        
        # Character-level features
        df['description_avg_word_length'] = df['description'].str.len() / (df['description'].str.split().str.len() + 1)
        
        # Special character counts (potential indicators of fake jobs)
        df['exclamation_count'] = df['description'].str.count(r'\!')
        df['question_count'] = df['description'].str.count(r'\?')
        df['dollar_sign_count'] = df['description'].str.count(r'\$')
        
        # URL presence
        df['has_url'] = df['description'].str.contains(r'http', case=False, na=False).astype(int)
        
        # Email presence
        df['has_email'] = df['description'].str.contains(r'@', na=False).astype(int)
        
        # Phone number presence
        df['has_phone'] = df['description'].str.contains(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', na=False).astype(int)
        
        logger.info("Text features added successfully")
        
        return df
    
    def clean_text_basic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic text cleaning"""
        logger.info("Performing basic text cleaning...")
        
        # Remove extra whitespace
        df['description'] = df['description'].str.strip().str.replace(r'\s+', ' ', regex=True)
        df['title'] = df['title'].str.strip().str.replace(r'\s+', ' ', regex=True)
        
        # Remove special characters (keep basic punctuation)
        df['description'] = df['description'].str.replace(r'[^\w\s\.\,\!\?\-]', '', regex=True)
        df['title'] = df['title'].str.replace(r'[^\w\s\.\,\!\?\-]', '', regex=True)
        
        logger.info("Basic text cleaning completed")
        
        return df
    
    def save_cleaned_data(self, df: pd.DataFrame, filename: str = "cleaned_dataset.csv") -> Path:
        """Save cleaned dataset"""
        output_path = self.processed_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to {output_path}")
        return output_path
    
    def generate_cleaning_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict:
        """Generate a report of the cleaning process"""
        report = {
            'original_rows': len(original_df),
            'cleaned_rows': len(cleaned_df),
            'rows_removed': len(original_df) - len(cleaned_df),
            'removal_percentage': ((len(original_df) - len(cleaned_df)) / len(original_df)) * 100,
            'original_class_distribution': original_df['fraudulent'].value_counts().to_dict(),
            'cleaned_class_distribution': cleaned_df['fraudulent'].value_counts().to_dict(),
            'original_missing_values': original_df.isnull().sum().to_dict(),
            'cleaned_missing_values': cleaned_df.isnull().sum().to_dict()
        }
        
        logger.info("=== Cleaning Report ===")
        logger.info(f"Original rows: {report['original_rows']}")
        logger.info(f"Cleaned rows: {report['cleaned_rows']}")
        logger.info(f"Rows removed: {report['rows_removed']} ({report['removal_percentage']:.2f}%)")
        logger.info(f"Original class distribution: {report['original_class_distribution']}")
        logger.info(f"Cleaned class distribution: {report['cleaned_class_distribution']}")
        
        return report

def main():
    """Main execution function"""
    cleaner = DataCleaner()
    
    # Load initial data
    df = cleaner.load_data("initial_dataset.csv")
    original_df = df.copy()
    
    # Handle missing values
    df = cleaner.handle_missing_values(df)
    
    # Remove duplicates
    df = cleaner.remove_duplicates(df)
    
    # Balance classes
    df = cleaner.balance_classes(df, method='smote')
    
    # Add text features
    df = cleaner.add_text_features(df)
    
    # Basic text cleaning
    df = cleaner.clean_text_basic(df)
    
    # Save cleaned data
    cleaner.save_cleaned_data(df, "cleaned_dataset.csv")
    
    # Generate report
    report = cleaner.generate_cleaning_report(original_df, df)
    
    logger.info("Data cleaning completed successfully!")

if __name__ == "__main__":
    main()
