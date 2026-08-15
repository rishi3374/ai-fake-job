"""
Streamlit App for AI Fake Job Detector
Simple web interface for quick deployment and demo
"""

import streamlit as st
import sys
from pathlib import Path

# Optional imports with fallbacks
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
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="AI Fake Job Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .result-fake {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .result-real {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.hybrid_model = None
    st.session_state.shap_explainer = None

# Load models
@st.cache_resource
def load_models():
    """Load models with caching"""
    try:
        from models.hybrid.hybrid_model import HybridModel
        from explainability.models.shap_explainer import SHAPExplainer
        
        hybrid_model = HybridModel()
        shap_explainer = SHAPExplainer()
        shap_explainer.initialize_explainer()
        
        return hybrid_model, shap_explainer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Sidebar
with st.sidebar:
    st.title("🛡️ AI Fake Job Detector")
    st.markdown("---")
    
    # Model loading status
    if not st.session_state.model_loaded:
        if st.button("Load Models", type="primary"):
            with st.spinner("Loading models..."):
                hybrid_model, shap_explainer = load_models()
                if hybrid_model:
                    st.session_state.hybrid_model = hybrid_model
                    st.session_state.shap_explainer = shap_explainer
                    st.session_state.model_loaded = True
                    st.success("Models loaded successfully!")
                    st.rerun()
    else:
        st.success("✅ Models loaded")
        
        if st.button("Reload Models"):
            st.session_state.model_loaded = False
            st.session_state.hybrid_model = None
            st.session_state.shap_explainer = None
            st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This system uses advanced AI to detect fraudulent job postings through:
    
    - **RoBERTa**: Deep learning text analysis
    - **SHAP**: Explainable AI insights
    - **Company Verification**: Legitimacy checks
    - **Salary Analysis**: Anomaly detection
    """)
    
    st.markdown("---")
    st.markdown("### Quick Test")
    if st.button("Test with Sample Data"):
        st.session_state.sample_data = {
            'job_title': 'Senior Software Engineer',
            'company_name': 'TechCorp',
            'salary': '$80,000 - $95,000',
            'job_description': '''Senior Software Engineer position requiring 5+ years Python experience. 
            Competitive salary and benefits. TechCorp is a leading technology company with 500 employees worldwide. 
            We specialize in software development and cloud solutions.''',
            'has_company_logo': True
        }
        st.rerun()

# Main content
st.markdown('<h1 class="main-header">AI Fake Job Detector</h1>', unsafe_allow_html=True)

# Check if models are loaded
if not st.session_state.model_loaded:
    st.warning("⚠️ Please load models from the sidebar to begin analysis")
    st.info("Click 'Load Models' in the sidebar to initialize the AI system")
else:
    # Get sample data if available
    if 'sample_data' in st.session_state:
        sample_data = st.session_state.sample_data
    else:
        sample_data = None
    
    # Input form
    with st.form("job_analysis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            job_title = st.text_input(
                "Job Title",
                value=sample_data['job_title'] if sample_data else "",
                placeholder="e.g., Senior Software Engineer"
            )
            company_name = st.text_input(
                "Company Name",
                value=sample_data['company_name'] if sample_data else "",
                placeholder="e.g., TechCorp"
            )
        
        with col2:
            salary = st.text_input(
                "Salary Information",
                value=sample_data['salary'] if sample_data else "",
                placeholder="e.g., $80,000 - $95,000"
            )
            has_company_logo = st.checkbox(
                "Company has logo",
                value=sample_data['has_company_logo'] if sample_data else False
            )
        
        job_description = st.text_area(
            "Job Description",
            value=sample_data['job_description'] if sample_data else "",
            placeholder="Paste the complete job description here...",
            height=150
        )
        
        submitted = st.form_submit_button("Analyze Job Posting", type="primary", use_container_width=True)
        
        if submitted:
            if not job_description:
                st.error("Please provide a job description")
            else:
                # Show progress
                with st.spinner("Analyzing job posting..."):
                    try:
                        # Prepare company data
                        company_data = {
                            'name': company_name,
                            'has_company_logo': 1 if has_company_logo else 0,
                            'company_profile': '',
                            'email': '',
                            'has_questions': 0
                        }
                        
                        # Make prediction
                        result = st.session_state.hybrid_model.predict(
                            text=job_description,
                            salary_str=salary,
                            company_data=company_data,
                            job_title=job_title
                        )
                        
                        # Store result in session state
                        st.session_state.last_result = result
                        st.session_state.last_input = {
                            'job_title': job_title,
                            'company_name': company_name,
                            'salary': salary,
                            'job_description': job_description
                        }
                        
                        # Clear sample data after use
                        if 'sample_data' in st.session_state:
                            del st.session_state.sample_data
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

# Display results
if 'last_result' in st.session_state and 'last_input' in st.session_state:
    result = st.session_state.last_result
    input_data = st.session_state.last_input
    
    st.markdown("---")
    
    # Main result
    if result['is_fake']:
        st.markdown('<div class="result-fake">', unsafe_allow_html=True)
        st.markdown("### 🚨 FAKE Job Posting Detected")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-real">', unsafe_allow_html=True)
        st.markdown("### ✅ REAL Job Posting")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Confidence",
            f"{result['confidence']:.1%}",
            delta="High" if result['confidence'] > 0.8 else "Medium" if result['confidence'] > 0.5 else "Low"
        )
    
    with col2:
        st.metric(
            "Fraud Probability",
            f"{result['fraud_probability']:.1%}",
            delta="High" if result['fraud_probability'] > 0.7 else "Medium" if result['fraud_probability'] > 0.3 else "Low"
        )
    
    with col3:
        st.metric(
            "Risk Level",
            result['risk_level'],
            delta="⚠️" if result['risk_level'] == "High" else "⚡" if result['risk_level'] == "Medium" else "✅"
        )
    
    # Explanation
    if st.session_state.shap_explainer:
        try:
            with st.spinner("Generating explanation..."):
                shap_result = st.session_state.shap_explainer.explain_with_shap(input_data['job_description'])
                explanation = st.session_state.hybrid_model.explain_prediction(result)
                
                st.markdown("---")
                st.markdown("### 📋 Explanation")
                st.write(explanation)
                
                # Suspicious phrases
                if shap_result.get('top_suspicious_phrases'):
                    st.markdown("### 🔍 Suspicious Phrases Detected")
                    for phrase in shap_result['top_suspicious_phrases']:
                        st.markdown(f"- **{phrase}**")
        except Exception as e:
            st.warning(f"Could not generate detailed explanation: {e}")
    
    # Component scores
    st.markdown("---")
    st.markdown("### 📊 Component Analysis")
    
    components = result['components']
    
    for component, score in components.items():
        if isinstance(score, dict):
            # Handle nested components (like roberta)
            st.markdown(f"**{component.replace('_', ' ').title()}**")
            for sub_key, sub_score in score.items():
                if isinstance(sub_score, (int, float)):
                    st.progress(sub_score if sub_key in ['probability', 'confidence'] else sub_score * 100 if sub_score <= 1 else sub_score / 100)
                    st.caption(f"{sub_key.replace('_', ' ').title()}: {sub_score:.2%}" if isinstance(sub_score, float) else f"{sub_key.replace('_', ' ').title()}: {sub_score}")
        elif isinstance(score, (int, float)):
            st.markdown(f"**{component.replace('_', ' ').title()}**")
            st.progress(score * 100 if score <= 1 else score / 100)
            st.caption(f"Score: {score:.2%}" if score <= 1 else f"Score: {score:.2f}")
    
    # Action buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Analyze Another Job"):
            st.session_state.last_result = None
            st.session_state.last_input = None
            st.rerun()
    
    with col2:
        if st.button("📋 Copy Results"):
            results_text = f"""
AI Fake Job Detector Analysis Results
=====================================

Job Title: {input_data['job_title']}
Company: {input_data['company_name']}
Salary: {input_data['salary']}

Prediction: {'FAKE' if result['is_fake'] else 'REAL'}
Confidence: {result['confidence']:.1%}
Fraud Probability: {result['fraud_probability']:.1%}
Risk Level: {result['risk_level']}

Component Scores:
"""
            for component, score in components.items():
                if isinstance(score, dict):
                    results_text += f"\n{component}:"
                    for sub_key, sub_score in score.items():
                        if isinstance(sub_score, (int, float)):
                            results_text += f"\n  {sub_key}: {sub_score:.2%}" if isinstance(sub_score, float) else f"\n  {sub_key}: {sub_score}"
                elif isinstance(score, (int, float)):
                    results_text += f"\n{component}: {score:.2%}" if score <= 1 else f"\n{component}: {score:.2f}"
            
            st.code(results_text, language=None)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; font-size: 0.875rem;'>
<p>Built with ❤️ using RoBERTa, SHAP, and Ensemble Methods</p>
<p>© 2024 AI Fake Job Detector | IEEE-level Project</p>
</div>
""", unsafe_allow_html=True)
