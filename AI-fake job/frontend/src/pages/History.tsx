import { Clock, Shield, AlertCircle } from 'lucide-react'

function History() {
  // This would fetch from the API in production
  const mockHistory = [
    {
      id: 1,
      timestamp: '2024-01-15T10:30:00',
      job_title: 'Senior Software Engineer',
      company: 'TechCorp',
      prediction: 'real',
      confidence: 0.92,
    },
    {
      id: 2,
      timestamp: '2024-01-15T09:15:00',
      job_title: 'Easy Money Maker',
      company: 'QuickCash Inc',
      prediction: 'fake',
      confidence: 0.87,
    },
    {
      id: 3,
      timestamp: '2024-01-14T16:45:00',
      job_title: 'Data Analyst',
      company: 'DataDriven Co',
      prediction: 'real',
      confidence: 0.95,
    },
  ]

  const getStatusIcon = (prediction: string) => {
    return prediction === 'fake' 
      ? <AlertCircle className="w-5 h-5 text-danger-600" />
      : <Shield className="w-5 h-5 text-success-600" />
  }

  const getStatusColor = (prediction: string) => {
    return prediction === 'fake'
      ? 'bg-danger-50 border-danger-200'
      : 'bg-success-50 border-success-200'
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <h1 className="text-3xl font-bold">Analysis History</h1>
      
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Recent Analyses</h2>
          <button className="btn-secondary text-sm">Clear History</button>
        </div>
        
        <div className="space-y-4">
          {mockHistory.map((item) => (
            <div
              key={item.id}
              className={`p-4 rounded-lg border-2 ${getStatusColor(item.prediction)}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-2">
                    {getStatusIcon(item.prediction)}
                    <h3 className="font-semibold">{item.job_title}</h3>
                  </div>
                  <p className="text-sm text-gray-600 mb-2">{item.company}</p>
                  <div className="flex items-center space-x-4 text-sm">
                    <span className="flex items-center space-x-1">
                      <Clock className="w-4 h-4" />
                      {new Date(item.timestamp).toLocaleString()}
                    </span>
                    <span className="font-semibold">
                      Confidence: {(item.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                <div className={`px-3 py-1 rounded-full text-sm font-semibold ${
                  item.prediction === 'fake' 
                    ? 'bg-danger-100 text-danger-800' 
                    : 'bg-success-100 text-success-800'
                }`}>
                  {item.prediction.toUpperCase()}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Statistics</h2>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-3xl font-bold text-primary-600">3</div>
            <div className="text-gray-600">Total Analyses</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-danger-600">1</div>
            <div className="text-gray-600">Flagged as Fake</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-success-600">2</div>
            <div className="text-gray-600">Verified as Real</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default History
