# User Manual

This manual provides comprehensive instructions for using the AI Fake Job Detector system.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Web Interface](#web-interface)
3. [API Usage](#api-usage)
4. [Understanding Results](#understanding-results)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

## Getting Started

### Accessing the Application

**Web Interface:**
- Development: `http://localhost:3000`
- Production: `https://your-domain.com`

**API:**
- Development: `http://localhost:8000`
- Production: `https://your-domain.com/api`

### First-Time Setup

1. **Open the web interface** in your browser
2. **Navigate to the Analyze page** using the navigation menu
3. **Enter job information** in the provided form
4. **Click "Analyze Job Posting"** to get results

## Web Interface

### Home Page

The home page provides:
- **Overview** of the system capabilities
- **Feature highlights** (AI analysis, real-time processing, explainable AI, multi-modal analysis)
- **How it works** guide
- **System statistics**

### Analyze Page

The analyze page allows you to analyze job postings:

#### Input Fields

**Job Title:**
- Enter the position title (e.g., "Senior Software Engineer")
- Helps with context and salary analysis

**Company Name:**
- Enter the company name (e.g., "TechCorp")
- Used for company legitimacy verification

**Salary Information:**
- Enter salary details (e.g., "$80,000 - $95,000")
- Supports various formats:
  - Annual: "$80,000 - $95,000"
  - Monthly: "$6,500 - $7,500/month"
  - Weekly: "$1,500/week"
  - Hourly: "$40/hour"

**Job Description:**
- Paste the complete job description
- The more detailed, the better the analysis
- Include requirements, benefits, and company information

**Company Logo:**
- Check if the job posting includes a company logo
- Helps with legitimacy assessment

#### Image Upload

Upload job posting screenshots for OCR analysis:
- Supported formats: PNG, JPG, JPEG, PDF
- Maximum file size: 10MB
- OCR extracts text for analysis

### History Page

View your analysis history:
- **Recent analyses** with timestamps
- **Prediction results** (fake/real)
- **Confidence scores**
- **Statistics** (total analyses, flagged as fake, verified as real)

### About Page

Learn about:
- **System technology** (RoBERTa, SHAP, OCR, company verification)
- **Features** (multi-modal analysis, real-time detection, explainable AI)
- **Research background**
- **Privacy and security**

## API Usage

### Quick Start

**Text Analysis:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "Senior Software Engineer position...",
    "company_name": "TechCorp",
    "salary": "$80,000 - $95,000"
  }'
```

**Image Analysis:**
```bash
curl -X POST http://localhost:8000/image-predict \
  -F "file=@job_posting.png"
```

### Python Example

```python
import requests

def analyze_job(job_description, company_name="", salary=""):
    """Analyze a job posting"""
    response = requests.post('http://localhost:8000/predict', json={
        'job_description': job_description,
        'company_name': company_name,
        'salary': salary
    })
    
    return response.json()

# Example usage
result = analyze_job(
    job_description="Senior Software Engineer position...",
    company_name="TechCorp",
    salary="$80,000 - $95,000"
)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

### JavaScript Example

```javascript
async function analyzeJob(jobDescription, companyName = "", salary = "") {
    const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            job_description: jobDescription,
            company_name: companyName,
            salary: salary
        })
    });
    
    return await response.json();
}

// Example usage
const result = await analyzeJob(
    "Senior Software Engineer position...",
    "TechCorp",
    "$80,000 - $95,000"
);

console.log('Prediction:', result.prediction);
console.log('Confidence:', (result.confidence * 100).toFixed(1) + '%');
```

## Understanding Results

### Prediction Output

**Prediction:** `fake` or `real`
- **fake**: Job posting appears fraudulent
- **real**: Job posting appears legitimate

**Confidence:** 0.0 to 1.0
- Higher values indicate more confidence in the prediction
- > 0.8: High confidence
- 0.5 - 0.8: Medium confidence
- < 0.5: Low confidence

**Risk Level:** `High`, `Medium`, or `Low`
- **High**: Strong indicators of fraud
- **Medium**: Some suspicious elements
- **Low**: Appears legitimate

**Fraud Probability:** 0.0 to 1.0
- Probability that the job is fake
- Complement of confidence for real predictions

### Component Scores

The system uses multiple components:

**RoBERTa:** Text analysis using deep learning
- Score: 0.0 (fake) to 1.0 (real)
- Higher scores indicate legitimate text

**Salary Anomaly:** Salary realism check
- Score: 0.0 (anomalous) to 1.0 (normal)
- Low scores indicate unrealistic salaries

**Company Legitimacy:** Company verification
- Score: 0.0 (suspicious) to 1.0 (legitimate)
- Higher scores indicate legitimate companies

**OCR Confidence:** Image text extraction quality
- Score: 0.0 (poor) to 1.0 (excellent)
- Only applicable for image uploads

**Heuristic Score:** Rule-based analysis
- Score: 0.0 (legitimate) to 1.0 (suspicious)
- Based on scam keywords and patterns

### Suspicious Phrases

The system highlights suspicious phrases:
- **Urgency language**: "URGENT", "immediate", "ASAP"
- **Unrealistic promises**: "earn $5000 weekly", "get rich overnight"
- **Suspicious contact**: "telegram", "whatsapp", free email providers
- **Missing requirements**: "no experience needed", "no qualification"

### Explanation

The system provides a human-readable explanation:
- Summarizes the prediction
- Lists key factors
- Provides context for the decision

## Best Practices

### For Job Seekers

**Before Applying:**
1. **Analyze the job posting** using this tool
2. **Check the company** independently (website, LinkedIn, reviews)
3. **Verify contact information** (company email, not personal)
4. **Research salary ranges** for the position and location
5. **Be cautious of** urgent hiring and unrealistic promises

**Red Flags:**
- Unrealistic salary for the position
- Urgent hiring with no interview process
- Requests for payment or personal information
- Poor grammar and spelling in job description
- Company has no online presence
- Contact via messaging apps only
- No specific job requirements

### For Recruiters

**To Ensure Legitimacy:**
1. **Provide detailed job descriptions** with specific requirements
2. **Include company information** and website
3. **Use professional email** (company domain, not free providers)
4. **Offer realistic compensation** based on market rates
5. **Include company logo** and branding
6. **Provide clear application process**

### For Bulk Analysis

**API Integration:**
- Use the API for batch processing
- Implement rate limiting for large datasets
- Store results for analysis and tracking
- Monitor prediction confidence scores
- Review false positives/negatives

## Troubleshooting

### Common Issues

**Issue: Prediction takes too long**
- **Solution**: Check internet connection, server status
- **Cause**: Large job descriptions or server load

**Issue: Low confidence score**
- **Solution**: Provide more detailed job information
- **Cause**: Insufficient information for analysis

**Issue: False positive (legitimate job flagged as fake)**
- **Solution**: Report the job for model improvement
- **Cause**: Model may need retraining with similar examples

**Issue: False negative (fake job not detected)**
- **Solution**: Provide feedback on the job posting
- **Cause**: New fraud patterns not in training data

**Issue: Image upload fails**
- **Solution**: Check file format and size
- **Cause**: Unsupported format or file too large

### Getting Help

**Documentation:**
- [Installation Guide](INSTALLATION.md)
- [API Documentation](API.md)
- [Developer Guide](DEVELOPER.md)

**Support:**
- GitHub Issues: Report bugs and feature requests
- Email: support@example.com
- Community Forum: Discuss with other users

### Feedback

**Provide Feedback:**
- Report false positives/negatives
- Suggest new features
- Share improvement ideas
- Contribute to the project

**Feedback Channels:**
- GitHub Issues
- Community Forum
- Email feedback

## Security Tips

### Protecting Yourself

**Never Share:**
- Social Security Number
- Bank account information
- Credit card details
- Passwords
- Personal financial information

**Verify Before:**
- Providing personal information
- Paying for anything
- Downloading files
- Clicking links in messages

**Best Practices:**
- Use the tool to verify job postings
- Research companies independently
- Trust your instincts
- Report suspicious postings
- Keep personal information private

## Advanced Features

### Batch Processing

For processing multiple job postings:

```python
import requests
import pandas as pd

# Load job postings
jobs = pd.read_csv('job_postings.csv')

# Process each job
results = []
for _, job in jobs.iterrows():
    result = requests.post('http://localhost:8000/predict', json={
        'job_description': job['description'],
        'company_name': job['company'],
        'salary': job['salary']
    }).json()
    results.append(result)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('analysis_results.csv', index=False)
```

### Custom Integration

Integrate with your application:

```python
class FakeJobDetector:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
    
    def analyze(self, job_data):
        """Analyze a job posting"""
        response = requests.post(
            f"{self.api_url}/predict",
            json=job_data
        )
        return response.json()
    
    def analyze_batch(self, jobs):
        """Analyze multiple job postings"""
        return [self.analyze(job) for job in jobs]

# Usage
detector = FakeJobDetector()
result = detector.analyze({
    'job_description': 'Senior Software Engineer...',
    'company_name': 'TechCorp'
})
```

## Updates and Maintenance

### System Updates

The system is regularly updated with:
- **New fraud patterns** detected
- **Model improvements** and retraining
- **Feature enhancements**
- **Bug fixes** and security patches

### Data Privacy

- Job descriptions are not stored permanently
- Analysis results are logged for improvement
- Personal information is never collected
- Data is processed securely

## Conclusion

The AI Fake Job Detector is a powerful tool for identifying fraudulent job postings. By following this manual and using the system effectively, you can protect yourself from job scams and make informed decisions about job opportunities.

For additional support or questions, refer to the documentation or contact the support team.
