import { Shield, Brain, Code, Users, Globe } from 'lucide-react'

function About() {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <h1 className="text-3xl font-bold">About AI Fake Job Detector</h1>
      
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Our Mission</h2>
        <p className="text-gray-700 leading-relaxed">
          AI Fake Job Detector is a cutting-edge system designed to protect job seekers from fraudulent 
          job postings. Using advanced machine learning and explainable AI, we analyze job descriptions, 
          company legitimacy, salary claims, and even image screenshots to identify suspicious job postings 
          with high accuracy.
        </p>
      </div>
      
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Technology Stack</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="flex items-start space-x-3">
            <Brain className="w-6 h-6 text-primary-600 mt-1" />
            <div>
              <h3 className="font-semibold">RoBERTa & Hybrid Models</h3>
              <p className="text-sm text-gray-600">
                State-of-the-art NLP models for text analysis
              </p>
            </div>
          </div>
          
          <div className="flex items-start space-x-3">
            <Shield className="w-6 h-6 text-primary-600 mt-1" />
            <div>
              <h3 className="font-semibold">SHAP Explainability</h3>
              <p className="text-sm text-gray-600">
                Understand why predictions are made
              </p>
            </div>
          </div>
          
          <div className="flex items-start space-x-3">
            <Code className="w-6 h-6 text-primary-600 mt-1" />
            <div>
              <h3 className="font-semibold">OCR Integration</h3>
              <p className="text-sm text-gray-600">
                Analyze job posting screenshots with EasyOCR
              </p>
            </div>
          </div>
          
          <div className="flex items-start space-x-3">
            <Users className="w-6 h-6 text-primary-600 mt-1" />
            <div>
              <h3 className="font-semibold">Company Verification</h3>
              <p className="text-sm text-gray-600">
                Multi-source company legitimacy checks
              </p>
            </div>
          </div>
        </div>
      </div>
      
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Features</h2>
        <ul className="space-y-3">
          <li className="flex items-start space-x-2">
            <Globe className="w-5 h-5 text-primary-600 mt-0.5" />
            <span className="text-gray-700">
              <strong>Multi-modal Analysis:</strong> Text, images, and structured data
            </span>
          </li>
          <li className="flex items-start space-x-2">
            <Globe className="w-5 h-5 text-primary-600 mt-0.5" />
            <span className="text-gray-700">
              <strong>Real-time Detection:</strong> Instant analysis with confidence scores
            </span>
          </li>
          <li className="flex items-start space-x-2">
            <Globe className="w-5 h-5 text-primary-600 mt-0.5" />
            <span className="text-gray-700">
              <strong>Explainable AI:</strong> Highlight suspicious phrases and reasoning
            </span>
          </li>
          <li className="flex items-start space-x-2">
            <Globe className="w-5 h-5 text-primary-600 mt-0.5" />
            <span className="text-gray-700">
              <strong>Salary Anomaly Detection:</strong> Identify unrealistic salary claims
            </span>
          </li>
          <li className="flex items-start space-x-2">
            <Globe className="w-5 h-5 text-primary-600 mt-0.5" />
            <span className="text-gray-700">
              <strong>Company Legitimacy:</strong> Verify company authenticity
            </span>
          </li>
        </ul>
      </div>
      
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Research & Development</h2>
        <p className="text-gray-700 leading-relaxed mb-4">
          This system is built on research from the Kaggle Fake Job Postings Dataset and incorporates 
          state-of-the-art techniques from academic literature on fraud detection and explainable AI.
        </p>
        <p className="text-gray-700 leading-relaxed">
          Our hybrid approach combines traditional machine learning (Logistic Regression, Random Forest, XGBoost) 
          with modern deep learning (RoBERTa) and rule-based systems to achieve high accuracy while maintaining 
          interpretability.
        </p>
      </div>
      
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Privacy & Security</h2>
        <p className="text-gray-700 leading-relaxed">
          We take your privacy seriously. Job descriptions are analyzed in real-time and are not stored 
          permanently unless you choose to save them to your history. All data is processed securely and 
          we do not share your information with third parties.
        </p>
      </div>
      
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Contact & Support</h2>
        <p className="text-gray-700 leading-relaxed">
          For questions, feedback, or support, please contact our team. We continuously improve our 
          models based on user feedback and emerging fraud patterns.
        </p>
      </div>
    </div>
  )
}

export default About
