"""
OCR Pipeline Module
Handles image text extraction using EasyOCR for job posting screenshots
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
from PIL import Image
import numpy as np

# Optional EasyOCR import
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OCRProcessor:
    """OCR processor for extracting text from job posting images"""
    
    def __init__(self, model_dir: str = "ocr/models", languages: List[str] = None):
        """
        Initialize OCR processor
        
        Args:
            model_dir: Directory to store OCR models
            languages: List of languages to detect (default: ['en'])
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.languages = languages or ['en']
        self.reader = None
        
        # Initialize OCR reader
        self._initialize_reader()
        
        logger.info(f"OCR Processor initialized with languages: {self.languages}")
    
    def _initialize_reader(self):
        """Initialize EasyOCR reader"""
        if not EASYOCR_AVAILABLE:
            logger.warning("EasyOCR not available, OCR functionality will be limited")
            return
        
        try:
            self.reader = easyocr.Reader(self.languages, model_storage_directory=str(self.model_dir))
            logger.info("EasyOCR reader initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize EasyOCR reader: {e}")
    
    def extract_text(self, image_path: str, detail: int = 1) -> Dict:
        """
        Extract text from image using OCR
        
        Args:
            image_path: Path to image file
            detail: Detail level (0=boxes only, 1=boxes+text+conf, 2=everything)
            
        Returns:
            Dictionary with OCR results
        """
        if not EASYOCR_AVAILABLE or self.reader is None:
            return self._fallback_extraction(image_path)
        
        try:
            # Perform OCR
            results = self.reader.readtext(image_path, detail=detail)
            
            # Process results
            extracted_data = self._process_ocr_results(results)
            
            logger.info(f"OCR extraction completed from {image_path}")
            return extracted_data
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return self._fallback_extraction(image_path)
    
    def _process_ocr_results(self, results: List) -> Dict:
        """
        Process OCR results into structured format
        
        Args:
            results: Raw OCR results from EasyOCR
            
        Returns:
            Processed OCR data
        """
        texts = []
        confidences = []
        boxes = []
        
        for result in results:
            if len(result) >= 3:
                bbox, text, confidence = result[:3]
                texts.append(text)
                confidences.append(confidence)
                boxes.append(bbox)
        
        # Calculate average confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Combine all text
        full_text = ' '.join(texts)
        
        return {
            'text': full_text,
            'texts': texts,
            'confidences': confidences,
            'boxes': boxes,
            'average_confidence': float(avg_confidence),
            'text_count': len(texts),
            'method': 'easyocr'
        }
    
    def _fallback_extraction(self, image_path: str) -> Dict:
        """
        Fallback extraction method (placeholder)
        
        Args:
            image_path: Path to image file
            
        Returns:
            Placeholder OCR results
        """
        logger.warning("Using fallback OCR extraction")
        
        try:
            # Try to get basic image info
            with Image.open(image_path) as img:
                width, height = img.size
                
            return {
                'text': "",
                'texts': [],
                'confidences': [],
                'boxes': [],
                'average_confidence': 0.0,
                'text_count': 0,
                'method': 'fallback',
                'image_info': {
                    'width': width,
                    'height': height,
                    'format': img.format
                }
            }
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return {
                'text': "",
                'texts': [],
                'confidences': [],
                'boxes': [],
                'average_confidence': 0.0,
                'text_count': 0,
                'method': 'failed'
            }
    
    def extract_job_posting_info(self, image_path: str) -> Dict:
        """
        Extract structured job posting information from image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Structured job posting information
        """
        # Get OCR results
        ocr_results = self.extract_text(image_path)
        text = ocr_results['text']
        
        if not text:
            return {
                'success': False,
                'error': 'No text extracted from image',
                'ocr_results': ocr_results
            }
        
        # Extract job posting components
        job_info = {
            'success': True,
            'full_text': text,
            'title': self._extract_job_title(text),
            'company': self._extract_company(text),
            'salary': self._extract_salary(text),
            'requirements': self._extract_requirements(text),
            'description': text,
            'contact': self._extract_contact(text),
            'ocr_confidence': ocr_results['average_confidence'],
            'ocr_results': ocr_results
        }
        
        return job_info
    
    def _extract_job_title(self, text: str) -> str:
        """Extract job title from text"""
        # Common job title patterns
        title_patterns = [
            r'(?:position|role|job title|title):\s*([^\n.]+)',
            r'(?:hiring|looking for|seeking)\s+(?:a\s+)?([^\n.]+?)(?:\s+(?:with|for|at))',
            r'^([A-Z][A-Za-z\s]+(?:\s+(?:Engineer|Manager|Developer|Analyst|Specialist|Director)))'
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_company(self, text: str) -> str:
        """Extract company name from text"""
        # Company patterns
        company_patterns = [
            r'(?:at|@)\s+([A-Z][A-Za-z\s]+?)(?:\s+(?:Inc|Ltd|LLC|Corp))',
            r'(?:company|organization):\s*([^\n.]+)',
            r'([A-Z][A-Za-z]+\s+(?:Inc|Ltd|LLC|Corp|Technologies|Solutions))'
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_salary(self, text: str) -> str:
        """Extract salary information from text"""
        # Salary patterns
        salary_patterns = [
            r'\$?\d{1,3}(?:,\d{3})*(?:\s*(?:to|-|–)\s*\$?\d{1,3}(?:,\d{3})*)?\s*(?:per|/|a)\s*(?:year|yr|annum|month|mo|week|wk|day|hr|hour)',
            r'\$?\d{1,3}(?:,\d{3})*\s*-\s*\$?\d{1,3}(?:,\d{3})*\s*(?:annually|yearly|monthly|weekly|daily|hourly)',
            r'(?:salary|pay|compensation|rate):\s*([^\n.]+)'
        ]
        
        for pattern in salary_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        
        return ""
    
    def _extract_requirements(self, text: str) -> str:
        """Extract job requirements from text"""
        # Requirement patterns
        requirement_patterns = [
            r'(?:requirements|qualifications|skills|must have):\s*([^\n]+(?:\n[^.]+)*?)(?=\n\n|\n(?:benefits|responsibilities))',
            r'(?:required|needed):\s*([^\n.]+)'
        ]
        
        for pattern in requirement_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_contact(self, text: str) -> str:
        """Extract contact information from text"""
        contact_info = {}
        
        # Email
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if email_match:
            contact_info['email'] = email_match.group(0)
        
        # Phone
        phone_match = re.search(r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
        if phone_match:
            contact_info['phone'] = phone_match.group(0)
        
        # Website
        website_match = re.search(r'https?://[^\s]+|www\.[^\s]+', text)
        if website_match:
            contact_info['website'] = website_match.group(0)
        
        return contact_info
    
    def process_whatsapp_screenshot(self, image_path: str) -> Dict:
        """
        Process WhatsApp/Telegram screenshot specifically
        
        Args:
            image_path: Path to screenshot image
            
        Returns:
            Processed job posting info
        """
        # Extract text
        job_info = self.extract_job_posting_info(image_path)
        
        # Add platform-specific processing
        job_info['platform'] = self._detect_platform(job_info['full_text'])
        job_info['is_messaging_app'] = job_info['platform'] in ['whatsapp', 'telegram']
        
        # Messaging app posts are higher risk
        if job_info['is_messaging_app']:
            job_info['risk_factor'] = 'high'
            job_info['risk_reason'] = 'Job posted on messaging platform'
        
        return job_info
    
    def _detect_platform(self, text: str) -> str:
        """Detect if text is from WhatsApp/Telegram"""
        text_lower = text.lower()
        
        if 'whatsapp' in text_lower or 'wa.me' in text_lower:
            return 'whatsapp'
        elif 'telegram' in text_lower or 't.me' in text_lower:
            return 'telegram'
        else:
            return 'unknown'
    
    def batch_process(self, image_paths: List[str]) -> List[Dict]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of processed results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.extract_job_posting_info(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'image_path': image_path
                })
        
        return results


def main():
    """Main execution function for testing"""
    # Create OCR processor
    ocr_processor = OCRProcessor()
    
    # Test with a placeholder (since we don't have actual images)
    print("=== OCR Processor Test ===")
    print(f"EasyOCR available: {EASYOCR_AVAILABLE}")
    
    if EASYOCR_AVAILABLE:
        print("OCR processor is ready to process images.")
        print("To test with actual images, provide image paths to the extract_text method.")
    else:
        print("EasyOCR not installed. Install with: pip install easyocr")
        print("OCR functionality will use fallback methods.")


if __name__ == "__main__":
    main()
