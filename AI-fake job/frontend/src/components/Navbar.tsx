import { Link } from 'react-router-dom'
import { Shield, Menu, X } from 'lucide-react'
import { useState } from 'react'

function Navbar() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <nav className="bg-white shadow-md">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center py-4">
          <Link to="/" className="flex items-center space-x-2">
            <Shield className="w-8 h-8 text-primary-600" />
            <span className="text-xl font-bold text-gray-900">AI Fake Job Detector</span>
          </Link>
          
          <div className="hidden md:flex space-x-8">
            <Link to="/" className="text-gray-700 hover:text-primary-600 transition-colors">Home</Link>
            <Link to="/analyze" className="text-gray-700 hover:text-primary-600 transition-colors">Analyze</Link>
            <Link to="/history" className="text-gray-700 hover:text-primary-600 transition-colors">History</Link>
            <Link to="/about" className="text-gray-700 hover:text-primary-600 transition-colors">About</Link>
          </div>
          
          <button className="md:hidden" onClick={() => setIsOpen(!isOpen)}>
            {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
        
        {isOpen && (
          <div className="md:hidden py-4 space-y-2">
            <Link to="/" className="block text-gray-700 hover:text-primary-600 transition-colors">Home</Link>
            <Link to="/analyze" className="block text-gray-700 hover:text-primary-600 transition-colors">Analyze</Link>
            <Link to="/history" className="block text-gray-700 hover:text-primary-600 transition-colors">History</Link>
            <Link to="/about" className="block text-gray-700 hover:text-primary-600 transition-colors">About</Link>
          </div>
        )}
      </div>
    </nav>
  )
}

export default Navbar
