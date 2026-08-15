"""
Dataset Preparation Script
Downloads and prepares the Kaggle Fake Job Postings Dataset
"""

import os
import csv
import random
from pathlib import Path
import logging
from typing import Tuple

# Optional imports
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Optional kaggle import
try:
    import kaggle
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetPreparer:
    """Handles dataset downloading and initial preparation"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def download_kaggle_dataset(self) -> Path:
        """Download Fake Job Postings dataset from Kaggle"""
        if not KAGGLE_AVAILABLE:
            logger.warning("Kaggle API not available. Creating synthetic dataset for demonstration.")
            return self.create_synthetic_dataset()
            
        logger.info("Downloading Kaggle Fake Job Postings dataset...")
        
        try:
            # Download dataset using Kaggle API
            kaggle.api.dataset_download_files(
                'shivamb/real-or-fake-fake-jobposting-prediction',
                path=str(self.raw_dir),
                unzip=True
            )
            
            logger.info(f"Dataset downloaded successfully to {self.raw_dir}")
            return self.raw_dir / "fake_job_postings.csv"
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            logger.info("Attempting to use alternative method...")
            
            # Alternative: Create synthetic dataset for demonstration
            logger.warning("Creating synthetic dataset for demonstration purposes")
            return self.create_synthetic_dataset()
    
    def create_synthetic_dataset(self) -> Path:
        """Create a synthetic dataset for demonstration"""
        logger.info("Creating synthetic dataset...")
        
        # Sample fake job descriptions
        fake_jobs = [
            "Urgent hiring! Earn $5000 weekly working from home. No experience needed. Start today!",
            "Make money fast! Work from anywhere, earn $10000 per month with zero investment.",
            "Immediate opening! High paying job with no qualifications required. Telegram: @scam_job",
            "Easy money! Just click ads and earn $500 daily. No skills needed.",
            "Work from home and earn $8000 weekly. Send your resume to gethiredquick@gmail.com",
            "Data entry job paying $50 per hour. No experience required. Apply now!",
            "Become rich overnight! Investment opportunity with 100% returns guaranteed.",
            "Remote job paying $2000 per week for simple tasks. WhatsApp us for details.",
            "Hiring immediately! $3000 daily income from home. No interview needed.",
            "Easy work from home job earning $1500 daily. Limited spots available!"
        ]
        
        # Sample real job descriptions
        real_jobs = [
            "Senior Software Engineer at TechCorp Inc. Requirements: 5+ years Python experience, "
            "knowledge of AWS, Docker, and Kubernetes. Competitive salary $120,000-$150,000. "
            "Apply through our careers portal at techcorp.com/careers.",
            "Marketing Manager at Global Solutions Ltd. Seeking experienced professional with "
            "MBA and 3+ years in digital marketing. Salary: $80,000-$95,000. Email careers@globalsolutions.com",
            "Data Analyst position at DataDriven Co. Requirements: SQL, Python, Tableau skills. "
            "2+ years experience. $70,000-$85,000 annually. Apply via LinkedIn.",
            "Junior Web Developer at StartupXYZ. Looking for React.js and Node.js developers. "
            "1-2 years experience. $60,000-$75,000. Visit startupxyz.io/jobs",
            "Product Manager at InnovateTech. 5+ years product management experience. "
            "MBA preferred. $110,000-$130,000. careers@innovatetech.com",
            "DevOps Engineer at CloudScale. AWS, Kubernetes, CI/CD experience required. "
            "4+ years experience. $100,000-$125,000. cloudscale.com/careers",
            "UX Designer at DesignHub. Portfolio required. 3+ years experience. "
            "$75,000-$90,000. careers@designhub.io",
            "Business Analyst at FinanceFirst. CFA preferred. Financial modeling skills. "
            "$85,000-$100,000. careers@financefirst.com",
            "Machine Learning Engineer at AI Solutions. PyTorch, TensorFlow experience. "
            "3+ years experience. $115,000-$140,000. aisolutions.ai/careers",
            "Full Stack Developer at WebWorks. MERN stack experience. 2+ years. "
            "$65,000-$80,000. careers@webworks.com"
        ]
        
        # Define data
        titles = [
            'Work From Home', 'Easy Money Maker', 'Quick Cash Job', 'Ad Clicker', 
            'Remote Data Entry', 'High Pay Data Entry', 'Investment Opportunity', 
            'Simple Tasks Remote', 'Immediate Hiring', 'Easy Home Work',
            'Senior Software Engineer', 'Marketing Manager', 'Data Analyst', 
            'Junior Web Developer', 'Product Manager', 'DevOps Engineer', 
            'UX Designer', 'Business Analyst', 'ML Engineer', 'Full Stack Developer'
        ]
        
        locations = ['Remote', 'New York', 'San Francisco', 'London', 'Remote']
        departments = ['Engineering', 'Marketing', 'Data', 'Sales', 'IT']
        salary_ranges = [
            '5000-10000', '8000-12000', '3000-5000', '500-1000', '2000-4000',
            '40-50', '10000-20000', '1500-3000', '2500-3500', '1200-1800',
            '120000-150000', '80000-95000', '70000-85000', '60000-75000',
            '110000-130000', '100000-125000', '75000-90000', '85000-100000',
            '115000-140000', '65000-80000'
        ]
        company_profiles = ['Detailed profile', 'Brief profile', 'No profile']
        requirements = ['None', 'Basic skills', 'Advanced skills']
        benefits = ['None', 'Health insurance', '401k', 'Flexible hours']
        telecommuting = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
        has_company_logo = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        has_questions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        employment_types = ['Full-time', 'Part-time', 'Contract', 'Other']
        required_experiences = ['Not applicable', 'Entry level', 'Mid level', 'Senior level']
        required_educations = ['Not applicable', 'Bachelor', 'Master', 'PhD']
        industries = ['Technology', 'Finance', 'Healthcare', 'Retail', 'Other']
        functions = ['Engineering', 'Sales', 'Marketing', 'Other']
        fraudulent = [1] * 10 + [0] * 10
        
        # Create CSV file
        output_path = self.raw_dir / "fake_job_postings.csv"
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'job_id', 'title', 'location', 'department', 'salary_range',
                'company_profile', 'description', 'requirements', 'benefits',
                'telecommuting', 'has_company_logo', 'has_questions',
                'employment_type', 'required_experience', 'required_education',
                'industry', 'function', 'fraudulent'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i in range(20):
                writer.writerow({
                    'job_id': i + 1,
                    'title': titles[i],
                    'location': random.choice(locations),
                    'department': random.choice(departments),
                    'salary_range': salary_ranges[i],
                    'company_profile': random.choice(company_profiles),
                    'description': fake_jobs[i] if i < 10 else real_jobs[i - 10],
                    'requirements': random.choice(requirements),
                    'benefits': random.choice(benefits),
                    'telecommuting': telecommuting[i],
                    'has_company_logo': has_company_logo[i],
                    'has_questions': has_questions[i],
                    'employment_type': random.choice(employment_types),
                    'required_experience': random.choice(required_experiences),
                    'required_education': random.choice(required_educations),
                    'industry': random.choice(industries),
                    'function': random.choice(functions),
                    'fraudulent': fraudulent[i]
                })
        
        logger.info(f"Synthetic dataset created at {output_path}")
        logger.info(f"Dataset contains 20 samples: 10 fake, 10 real")
        
        return output_path
    
    def load_dataset(self, filepath: Path):
        """Load dataset from CSV file"""
        logger.info(f"Loading dataset from {filepath}...")
        
        try:
            if PANDAS_AVAILABLE:
                df = pd.read_csv(filepath)
                logger.info(f"Dataset loaded successfully using pandas. Shape: {df.shape}")
                return df
            else:
                # Use csv module as fallback
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
                logger.info(f"Dataset loaded successfully using csv module. Rows: {len(data)}")
                return data
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def basic_info(self, data) -> None:
        """Display basic information about the dataset"""
        logger.info("\n=== Dataset Information ===")
        
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            logger.info(f"Shape: {data.shape}")
            logger.info(f"\nColumns: {data.columns.tolist()}")
            logger.info(f"\nData types:\n{data.dtypes}")
            logger.info(f"\nMissing values:\n{data.isnull().sum()}")
            logger.info(f"\nClass distribution:\n{data['fraudulent'].value_counts()}")
            logger.info(f"\nFirst few rows:\n{data.head()}")
        else:
            # Use standard library for info
            logger.info(f"Total rows: {len(data)}")
            if data:
                logger.info(f"Columns: {list(data[0].keys())}")
                
                # Count fraudulent
                fake_count = sum(1 for row in data if row.get('fraudulent') == '1')
                real_count = sum(1 for row in data if row.get('fraudulent') == '0')
                logger.info(f"\nClass distribution: Fake={fake_count}, Real={real_count}")
                
                logger.info(f"\nFirst row: {data[0]}")
    
    def save_processed_data(self, data, filename: str) -> Path:
        """Save processed dataset"""
        output_path = self.processed_dir / filename
        
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=False)
        else:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if data:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
        
        logger.info(f"Processed data saved to {output_path}")
        return output_path

def main():
    """Main execution function"""
    preparer = DatasetPreparer()
    
    # Download dataset
    dataset_path = preparer.download_kaggle_dataset()
    
    # Load dataset
    df = preparer.load_dataset(dataset_path)
    
    # Display basic information
    preparer.basic_info(df)
    
    # Save initial processed data
    preparer.save_processed_data(df, "initial_dataset.csv")
    
    logger.info("Dataset preparation completed successfully!")

if __name__ == "__main__":
    main()
