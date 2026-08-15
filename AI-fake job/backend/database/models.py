"""
Database Models
SQLAlchemy models for PostgreSQL database
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class Prediction(Base):
    """Model for job prediction results"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    job_description = Column(Text, nullable=False)
    job_title = Column(String(255))
    company_name = Column(String(255))
    salary = Column(String(255))
    
    prediction = Column(String(10), nullable=False)  # 'fake' or 'real'
    confidence = Column(Float, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    risk_level = Column(String(10))
    
    explanation = Column(Text)
    suspicious_phrases = Column(JSON)
    highlighted_text = Column(Text)
    components = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, prediction={self.prediction}, confidence={self.confidence})>"


class CompanyVerification(Base):
    """Model for company verification results"""
    __tablename__ = "company_verifications"
    
    id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String(255), nullable=False)
    website = Column(String(255))
    email = Column(String(255))
    profile = Column(Text)
    
    legitimacy_score = Column(Float)
    is_legitimate = Column(Boolean)
    risk_level = Column(String(10))
    
    verification_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<CompanyVerification(id={self.id}, company={self.company_name}, score={self.legitimacy_score})>"


class SalaryAnalysis(Base):
    """Model for salary anomaly detection results"""
    __tablename__ = "salary_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    salary_string = Column(String(255), nullable=False)
    job_title = Column(String(255))
    
    is_anomaly = Column(Boolean)
    anomaly_score = Column(Float)
    anomaly_probability = Column(Float)
    
    features = Column(JSON)
    method = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<SalaryAnalysis(id={self.id}, is_anomaly={self.is_anomaly}, score={self.anomaly_score})>"


class User(Base):
    """Model for user accounts (if authentication is added)"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, username={self.username})>"
