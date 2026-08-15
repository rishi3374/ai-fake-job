import { useState } from 'react'
import { Upload, FileText, AlertCircle, CheckCircle, AlertTriangle } from 'lucide-react'

function Analyze() {
  const [jobDescription, setJobDescription] = useState('')
  const [companyName, setCompanyName] = useState('')
  const [salary, setSalary] = useState('')
  const [jobTitle, setJobTitle] = useState('')
  const [hasCompanyLogo, setHasCompanyLogo] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState('')

  const analyzeJob = async () => {
    setLoading(true)
    setError('')
    setResult(null)

    try {
      // This will connect to the FastAPI backend
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          job_description: jobDescription,
          company_name: companyName,
          salary: salary,
          job_title: jobTitle,
          has_company_logo: hasCompanyLogo,
        }),
      })

      if (!response.ok) {
        throw new Error('Analysis failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError('Failed to analyze job posting. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel) {
      case 'High':
        return <AlertCircle className="w-6 h-6 text-danger-600" />
      case 'Medium':
        return <AlertTriangle className="w-6 h-6 text-yellow-600" />
      case 'Low':
        return <CheckCircle className="w-6 h-6 text-success-600" />
      default:
        return null
    }
  }

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'High':
        return 'bg-danger-50 border-danger-200'
      case 'Medium':
        return 'bg-yellow-50 border-yellow-200'
      case 'Low':
        return 'bg-success-50 border-success-200'
      default:
        return 'bg-gray-50 border-gray-200'
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <h1 className="text-3xl font-bold">Analyze Job Posting</h1>
      
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Job Information</h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Job Title
            </label>
            <input
              type="text"
              className="input-field"
              value={jobTitle}
              onChange={(e) => setJobTitle(e.target.value)}
              placeholder="e.g., Senior Software Engineer"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Company Name
            </label>
            <input
              type="text"
              className="input-field"
              value={companyName}
              onChange={(e) => setCompanyName(e.target.value)}
              placeholder="e.g., TechCorp"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Salary Information
            </label>
            <input
              type="text"
              className="input-field"
              value={salary}
              onChange={(e) => setSalary(e.target.value)}
              placeholder="e.g., $80,000 - $95,000"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Job Description
            </label>
            <textarea
              className="input-field min-h-32"
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              placeholder="Paste the complete job description here..."
            />
          </div>
          
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="companyLogo"
              checked={hasCompanyLogo}
              onChange={(e) => setHasCompanyLogo(e.target.checked)}
              className="w-4 h-4 text-primary-600"
            />
            <label htmlFor="companyLogo" className="text-sm text-gray-700">
              Company has logo
            </label>
          </div>
          
          <button
            onClick={analyzeJob}
            disabled={loading || !jobDescription}
            className="btn-primary w-full"
          >
            {loading ? 'Analyzing...' : 'Analyze Job Posting'}
          </button>
        </div>
      </div>
      
      {error && (
        <div className="card bg-danger-50 border border-danger-200">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-5 h-5 text-danger-600" />
            <p className="text-danger-800">{error}</p>
          </div>
        </div>
      )}
      
      {result && (
        <div className={`card border-2 ${getRiskColor(result.risk_level)}`}>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              {getRiskIcon(result.risk_level)}
              <h2 className="text-2xl font-bold">
                {result.prediction === 'fake' ? 'FAKE' : 'REAL'} Job Posting
              </h2>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-600">Confidence</div>
              <div className="text-2xl font-bold">
                {(result.confidence * 100).toFixed(1)}%
              </div>
            </div>
          </div>
          
          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white p-4 rounded-lg">
              <div className="text-sm text-gray-600">Risk Level</div>
              <div className="text-xl font-semibold">{result.risk_level}</div>
            </div>
            <div className="bg-white p-4 rounded-lg">
              <div className="text-sm text-gray-600">Fraud Probability</div>
              <div className="text-xl font-semibold">
                {(result.fraud_probability * 100).toFixed(1)}%
              </div>
            </div>
          </div>
          
          {result.explanation && (
            <div className="bg-white p-4 rounded-lg mb-4">
              <h3 className="font-semibold mb-2">Explanation</h3>
              <p className="text-gray-700">{result.explanation}</p>
            </div>
          )}
          
          {result.suspicious_phrases && result.suspicious_phrases.length > 0 && (
            <div className="bg-white p-4 rounded-lg mb-4">
              <h3 className="font-semibold mb-2">Suspicious Phrases Detected</h3>
              <div className="flex flex-wrap gap-2">
                {result.suspicious_phrases.map((phrase: string, index: number) => (
                  <span
                    key={index}
                    className="px-3 py-1 bg-danger-100 text-danger-800 rounded-full text-sm"
                  >
                    {phrase}
                  </span>
                ))}
              </div>
            </div>
          )}
          
          {result.components && (
            <div className="bg-white p-4 rounded-lg">
              <h3 className="font-semibold mb-2">Component Scores</h3>
              <div className="space-y-2">
                {Object.entries(result.components).map(([key, value]: [string, any]) => (
                  <div key={key} className="flex justify-between items-center">
                    <span className="text-gray-700 capitalize">
                      {key.replace('_', ' ')}
                    </span>
                    <div className="flex items-center space-x-2">
                      <div className="w-32 bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-primary-600 h-2 rounded-full"
                          style={{ width: `${(value * 100 || 0)}%` }}
                        />
                      </div>
                      <span className="text-sm text-gray-600">
                        {typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : 'N/A'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
      
      <div className="card border-dashed border-2 border-gray-300">
        <div className="text-center">
          <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="font-semibold mb-2">Upload Job Posting Image</h3>
          <p className="text-gray-600 text-sm mb-4">
            Upload a screenshot of a job posting for OCR analysis
          </p>
          <button className="btn-secondary">
            Select Image
          </button>
        </div>
      </div>
    </div>
  )
}

export default Analyze
