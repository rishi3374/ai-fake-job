"""
Company Legitimacy Verification Module
Verifies company legitimacy using various data sources and heuristics
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
from datetime import datetime
import json

# Optional requests import
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CompanyVerifier:
    """Company legitimacy verification system"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize company verifier
        
        Args:
            cache_dir: Directory to cache verification results
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_file = self.cache_dir / "company_cache.json"
        self.cache = self._load_cache()
        
        # Suspicious domains
        self.suspicious_domains = [
            'tempmail.com', 'guerrillamail.com', 'mailinator.com',
            '10minutemail.com', 'yopmail.com', 'trashmail.com'
        ]
        
        # Free email providers
        self.free_email_providers = [
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
            'aol.com', 'mail.com', 'protonmail.com', 'zoho.com'
        ]
        
        # Legitimate company indicators
        self.legitimate_indicators = [
            'linkedin.com', 'indeed.com', 'glassdoor.com',
            'crunchbase.com', 'bloomberg.com', 'forbes.com'
        ]
        
        logger.info("Company Verifier initialized")
    
    def _load_cache(self) -> Dict:
        """Load verification cache"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save verification cache"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def verify_company(self, company_data: Dict) -> Dict:
        """
        Verify company legitimacy
        
        Args:
            company_data: Dictionary containing company information
            
        Returns:
            Verification result with legitimacy score
        """
        company_name = company_data.get('name', '')
        company_website = company_data.get('website', '')
        company_email = company_data.get('email', '')
        company_profile = company_data.get('profile', '')
        
        # Check cache
        cache_key = f"{company_name}_{company_website}"
        if cache_key in self.cache:
            logger.info(f"Returning cached result for {company_name}")
            return self.cache[cache_key]
        
        # Perform verification checks
        verification_result = {
            'company_name': company_name,
            'verification_timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        # Individual checks
        verification_result['checks']['domain_check'] = self._check_domain(company_website, company_email)
        verification_result['checks']['email_check'] = self._check_email(company_email)
        verification_result['checks']['profile_check'] = self._check_profile(company_profile)
        verification_result['checks']['name_check'] = self._check_company_name(company_name)
        verification_result['checks']['online_presence'] = self._check_online_presence(company_name, company_website)
        
        # Calculate overall legitimacy score
        legitimacy_score = self._calculate_legitimacy_score(verification_result['checks'])
        verification_result['legitimacy_score'] = legitimacy_score
        verification_result['is_legitimate'] = legitimacy_score > 0.5
        verification_result['risk_level'] = self._determine_risk_level(legitimacy_score)
        
        # Cache result
        self.cache[cache_key] = verification_result
        self._save_cache()
        
        logger.info(f"Company verification completed for {company_name}: Score {legitimacy_score:.2f}")
        
        return verification_result
    
    def _check_domain(self, website: str, email: str) -> Dict:
        """Check domain legitimacy"""
        result = {
            'has_custom_domain': False,
            'domain_age': 'unknown',
            'suspicious_domain': False,
            'score': 0.5
        }
        
        if not website:
            return result
        
        # Check if custom domain (not free email)
        if not any(provider in website.lower() for provider in self.free_email_providers):
            result['has_custom_domain'] = True
            result['score'] += 0.3
        
        # Check for suspicious domains
        if any(suspicious in website.lower() for suspicious in self.suspicious_domains):
            result['suspicious_domain'] = True
            result['score'] -= 0.5
        
        # Check domain age (placeholder - would need external API)
        # In production, use whois API to check domain age
        result['domain_age'] = 'not_checked'
        
        return result
    
    def _check_email(self, email: str) -> Dict:
        """Check email legitimacy"""
        result = {
            'has_company_email': False,
            'uses_free_provider': False,
            'suspicious_email': False,
            'score': 0.5
        }
        
        if not email:
            return result
        
        # Check if using free email provider
        if any(provider in email.lower() for provider in self.free_email_providers):
            result['uses_free_provider'] = True
            result['score'] -= 0.2
        else:
            result['has_company_email'] = True
            result['score'] += 0.3
        
        # Check for suspicious email patterns
        if any(suspicious in email.lower() for suspicious in self.suspicious_domains):
            result['suspicious_email'] = True
            result['score'] -= 0.4
        
        return result
    
    def _check_profile(self, profile: str) -> Dict:
        """Check company profile completeness"""
        result = {
            'has_profile': False,
            'profile_length': 0,
            'has_detailed_info': False,
            'score': 0.0
        }
        
        if not profile:
            return result
        
        result['has_profile'] = True
        result['profile_length'] = len(profile)
        
        # Check for detailed information
        if len(profile) > 100:
            result['has_detailed_info'] = True
            result['score'] += 0.4
        elif len(profile) > 50:
            result['score'] += 0.2
        
        # Check for legitimate indicators in profile
        profile_lower = profile.lower()
        legitimate_count = sum(1 for indicator in self.legitimate_indicators if indicator in profile_lower)
        
        if legitimate_count > 0:
            result['score'] += min(legitimate_count * 0.1, 0.3)
        
        return result
    
    def _check_company_name(self, company_name: str) -> Dict:
        """Check company name patterns"""
        result = {
            'has_name': False,
            'name_length': 0,
            'suspicious_name': False,
            'score': 0.0
        }
        
        if not company_name:
            return result
        
        result['has_name'] = True
        result['name_length'] = len(company_name)
        
        # Check for suspicious name patterns
        suspicious_patterns = [
            r'^\d+',  # Starts with numbers
            r'.*\d{4,}.*',  # Contains 4+ consecutive numbers
            r'^[A-Z]{3,}$',  # All caps acronym
            r'.*(easy|quick|fast|instant).*',  # Suspicious words
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, company_name, re.IGNORECASE):
                result['suspicious_name'] = True
                result['score'] -= 0.2
                break
        
        if not result['suspicious_name']:
            result['score'] += 0.3
        
        return result
    
    def _check_online_presence(self, company_name: str, website: str) -> Dict:
        """Check online presence (placeholder for external API calls)"""
        result = {
            'has_website': bool(website),
            'website_accessible': False,
            'social_media_presence': False,
            'news_mentions': False,
            'score': 0.0
        }
        
        if website:
            result['score'] += 0.2
        
        # In production, would make actual HTTP requests
        # For now, use heuristics
        if REQUESTS_AVAILABLE:
            try:
                # Try to access website
                if website:
                    response = requests.head(website, timeout=5)
                    if response.status_code == 200:
                        result['website_accessible'] = True
                        result['score'] += 0.3
            except Exception as e:
                logger.debug(f"Could not check website accessibility: {e}")
        
        return result
    
    def _calculate_legitimacy_score(self, checks: Dict) -> float:
        """Calculate overall legitimacy score from individual checks"""
        total_score = 0.0
        total_weight = 0.0
        
        weights = {
            'domain_check': 0.25,
            'email_check': 0.20,
            'profile_check': 0.20,
            'name_check': 0.15,
            'online_presence': 0.20
        }
        
        for check_name, check_result in checks.items():
            weight = weights.get(check_name, 0.1)
            score = check_result.get('score', 0.5)
            total_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        return 0.5
    
    def _determine_risk_level(self, legitimacy_score: float) -> str:
        """Determine risk level based on legitimacy score"""
        if legitimacy_score > 0.7:
            return 'Low'
        elif legitimacy_score > 0.4:
            return 'Medium'
        else:
            return 'High'
    
    def batch_verify(self, companies: List[Dict]) -> List[Dict]:
        """
        Verify multiple companies in batch
        
        Args:
            companies: List of company data dictionaries
            
        Returns:
            List of verification results
        """
        results = []
        
        for company_data in companies:
            try:
                result = self.verify_company(company_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to verify company: {e}")
                results.append({
                    'error': str(e),
                    'company_name': company_data.get('name', 'Unknown')
                })
        
        return results
    
    def get_verification_report(self, verification_result: Dict) -> str:
        """
        Generate human-readable verification report
        
        Args:
            verification_result: Verification result dictionary
            
        Returns:
            Formatted report string
        """
        report_parts = []
        
        company_name = verification_result.get('company_name', 'Unknown Company')
        legitimacy_score = verification_result.get('legitimacy_score', 0.5)
        risk_level = verification_result.get('risk_level', 'Medium')
        
        report_parts.append(f"Company: {company_name}")
        report_parts.append(f"Legitimacy Score: {legitimacy_score:.1%}")
        report_parts.append(f"Risk Level: {risk_level}")
        
        # Add check details
        checks = verification_result.get('checks', {})
        for check_name, check_result in checks.items():
            score = check_result.get('score', 0.5)
            status = 'PASS' if score > 0.5 else 'FAIL'
            report_parts.append(f"  {check_name}: {status} (score: {score:.2f})")
        
        return '\n'.join(report_parts)


def main():
    """Main execution function for testing"""
    # Create company verifier
    verifier = CompanyVerifier()
    
    # Test verification
    test_company = {
        'name': 'TechCorp Solutions',
        'website': 'https://techcorp.com',
        'email': 'careers@techcorp.com',
        'profile': 'TechCorp is a leading technology company with 500 employees worldwide. We specialize in software development and cloud solutions.'
    }
    
    print("=== Company Verification Test ===")
    result = verifier.verify_company(test_company)
    
    print(verifier.get_verification_report(result))
    
    # Test with suspicious company
    suspicious_company = {
        'name': 'EasyMoney123',
        'website': 'https://tempmail.com',
        'email': 'gethiredquick@gmail.com',
        'profile': ''
    }
    
    print("\n=== Suspicious Company Test ===")
    result_suspicious = verifier.verify_company(suspicious_company)
    print(verifier.get_verification_report(result_suspicious))


if __name__ == "__main__":
    main()
