import { Link } from 'react-router-dom'
import { Shield, Zap, Eye, BarChart3 } from 'lucide-react'

function Home() {
  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <div className="text-center py-12">
        <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
          AI Fake Job Detector
        </h1>
        <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
          Protect yourself from fraudulent job postings using advanced AI and explainable machine learning
        </p>
        <Link to="/analyze" className="btn-primary inline-block text-lg">
          Analyze a Job Posting
        </Link>
      </div>

      {/* Features Section */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="card text-center">
          <Shield className="w-12 h-12 text-primary-600 mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">Advanced AI</h3>
          <p className="text-gray-600 text-sm">
            Uses RoBERTa and hybrid models for accurate detection
          </p>
        </div>
        
        <div className="card text-center">
          <Zap className="w-12 h-12 text-primary-600 mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">Real-time Analysis</h3>
          <p className="text-gray-600 text-sm">
            Get instant results with confidence scores
          </p>
        </div>
        
        <div className="card text-center">
          <Eye className="w-12 h-12 text-primary-600 mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">Explainable AI</h3>
          <p className="text-gray-600 text-sm">
            Understand why a job is flagged as suspicious
          </p>
        </div>
        
        <div className="card text-center">
          <BarChart3 className="w-12 h-12 text-primary-600 mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">Multi-modal</h3>
          <p className="text-gray-600 text-sm">
            Analyzes text, images, and company data
          </p>
        </div>
      </div>

      {/* How It Works */}
      <div className="card">
        <h2 className="text-2xl font-bold mb-6">How It Works</h2>
        <div className="space-y-4">
          <div className="flex items-start space-x-4">
            <div className="w-8 h-8 bg-primary-600 text-white rounded-full flex items-center justify-center font-bold">1</div>
            <div>
              <h3 className="font-semibold">Paste Job Description</h3>
              <p className="text-gray-600">Enter the job posting text or upload an image</p>
            </div>
          </div>
          
          <div className="flex items-start space-x-4">
            <div className="w-8 h-8 bg-primary-600 text-white rounded-full flex items-center justify-center font-bold">2</div>
            <div>
              <h3 className="font-semibold">AI Analysis</h3>
              <p className="text-gray-600">Our models analyze text, salary, and company legitimacy</p>
            </div>
          </div>
          
          <div className="flex items-start space-x-4">
            <div className="w-8 h-8 bg-primary-600 text-white rounded-full flex items-center justify-center font-bold">3</div>
            <div>
              <h3 className="font-semibold">Get Results</h3>
              <p className="text-gray-600">Receive detailed analysis with explanations</p>
            </div>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid md:grid-cols-3 gap-6">
        <div className="card text-center">
          <div className="text-3xl font-bold text-primary-600 mb-2">95%</div>
          <div className="text-gray-600">Detection Accuracy</div>
        </div>
        
        <div className="card text-center">
          <div className="text-3xl font-bold text-primary-600 mb-2">50K+</div>
          <div className="text-gray-600">Jobs Analyzed</div>
        </div>
        
        <div className="card text-center">
          <div className="text-3xl font-bold text-primary-600 mb-2">&lt;2s</div>
          <div className="text-gray-600">Average Response Time</div>
        </div>
      </div>
    </div>
  )
}

export default Home
