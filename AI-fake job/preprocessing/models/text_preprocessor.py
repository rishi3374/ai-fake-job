"""
Text Preprocessing Module
Comprehensive text preprocessing pipeline for job descriptions
"""

import re
import string
from typing import List, Optional, Union
import logging

# Optional imports for advanced NLP
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Comprehensive text preprocessing pipeline"""
    
    def __init__(self, use_spacy: bool = False, spacy_model: str = 'en_core_web_sm'):
        """
        Initialize the text preprocessor
        
        Args:
            use_spacy: Whether to use spaCy for preprocessing (slower but more accurate)
            spacy_model: Name of spaCy model to use
        """
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        
        # Initialize NLTK components
        self.nltk_available = NLTK_AVAILABLE
        if self.nltk_available:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                
                self.stop_words = set(stopwords.words('english'))
                self.lemmatizer = WordNetLemmatizer()
                logger.info("NLTK components initialized successfully")
            except Exception as e:
                logger.warning(f"NLTK initialization failed: {e}")
                self.nltk_available = False
        
        # Initialize spaCy if requested
        if self.use_spacy:
            try:
                self.nlp = spacy.load(spacy_model)
                logger.info(f"spaCy model '{spacy_model}' loaded successfully")
            except Exception as e:
                logger.warning(f"spaCy initialization failed: {e}, falling back to NLTK")
                self.use_spacy = False
        
        # Define scam-related keywords for special handling
        self.scam_keywords = {
            'urgent', 'immediate', 'asap', 'start today', 'no experience',
            'easy money', 'quick cash', 'work from home', 'earn', 'weekly',
            'telegram', 'whatsapp', 'gmail.com', 'yahoo.com', 'hotmail.com',
            'investment', 'guaranteed', 'risk-free', 'limited spots',
            'no interview', 'hire now', 'apply now', 'click here'
        }
        
        # Define urgency indicators
        self.urgency_words = {
            'urgent', 'immediately', 'asap', 'today', 'now', 'hurry',
            'limited time', 'ending soon', 'last chance', 'don\'t miss'
        }
    
    def lowercase(self, text: str) -> str:
        """Convert text to lowercase"""
        return text.lower()
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def remove_html(self, text: str) -> str:
        """Remove HTML tags from text"""
        if BS4_AVAILABLE:
            return BeautifulSoup(text, 'html.parser').get_text()
        else:
            # Fallback regex-based HTML removal
            html_pattern = r'<[^>]+>'
            return re.sub(html_pattern, '', text)
    
    def remove_email_addresses(self, text: str) -> str:
        """Remove email addresses from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '[EMAIL]', text)
    
    def remove_phone_numbers(self, text: str) -> str:
        """Remove phone numbers from text"""
        phone_pattern = r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
        return re.sub(phone_pattern, '[PHONE]', text)
    
    def remove_special_characters(self, text: str, keep_punctuation: bool = True) -> str:
        """Remove special characters from text"""
        if keep_punctuation:
            # Keep basic punctuation
            pattern = r'[^a-zA-Z0-9\s\.\,\!\?\-]'
        else:
            # Remove all non-alphanumeric characters
            pattern = r'[^a-zA-Z0-9\s]'
        
        return re.sub(pattern, '', text)
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace"""
        return ' '.join(text.split())
    
    def remove_numbers(self, text: str) -> str:
        """Remove numbers from text"""
        return re.sub(r'\d+', '', text)
    
    def expand_contractions(self, text: str) -> str:
        """Expand common contractions"""
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'t": " not",
            "'ve": " have",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def remove_stopwords(self, text: str, custom_stopwords: Optional[set] = None) -> str:
        """Remove stopwords from text"""
        if not self.nltk_available:
            return text
        
        stopword_set = self.stop_words
        if custom_stopwords:
            stopword_set = stopword_set.union(custom_stopwords)
        
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in stopword_set]
        
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text: str) -> str:
        """Lemmatize text using NLTK"""
        if not self.nltk_available:
            return text
        
        words = word_tokenize(text)
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(lemmatized_words)
    
    def lemmatize_text_spacy(self, text: str) -> str:
        """Lemmatize text using spaCy"""
        if not self.use_spacy:
            return self.lemmatize_text(text)
        
        doc = self.nlp(text)
        lemmatized_words = [token.lemma_ for token in doc]
        
        return ' '.join(lemmatized_words)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if self.nltk_available:
            return word_tokenize(text)
        else:
            # Simple tokenization fallback
            return text.split()
    
    def normalize_email(self, text: str) -> str:
        """Normalize email addresses to a standard format"""
        # Replace free email providers with [FREE_EMAIL]
        free_email_pattern = r'\b[A-Za-z0-9._%+-]+@(gmail|yahoo|hotmail|outlook|aol)\.[A-Z|a-z]{2,}\b'
        text = re.sub(free_email_pattern, '[FREE_EMAIL]', text, flags=re.IGNORECASE)
        
        # Replace other emails with [EMAIL]
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        text = re.sub(email_pattern, '[EMAIL]', text)
        
        return text
    
    def detect_urgency(self, text: str) -> int:
        """Detect urgency indicators in text"""
        text_lower = text.lower()
        urgency_count = sum(1 for word in self.urgency_words if word in text_lower)
        return urgency_count
    
    def detect_scam_keywords(self, text: str) -> int:
        """Count scam-related keywords in text"""
        text_lower = text.lower()
        scam_count = sum(1 for keyword in self.scam_keywords if keyword in text_lower)
        return scam_count
    
    def preprocess(self, 
                   text: str,
                   lowercase: bool = True,
                   remove_urls_flag: bool = True,
                   remove_html_flag: bool = True,
                   normalize_emails_flag: bool = True,
                   remove_phones_flag: bool = True,
                   expand_contractions_flag: bool = True,
                   remove_special_chars: bool = True,
                   remove_stopwords_flag: bool = True,
                   lemmatize_flag: bool = True,
                   keep_punctuation: bool = False) -> dict:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text to preprocess
            lowercase: Convert to lowercase
            remove_urls_flag: Remove URLs
            remove_html_flag: Remove HTML tags
            normalize_emails_flag: Normalize email addresses
            remove_phones_flag: Remove phone numbers
            expand_contractions_flag: Expand contractions
            remove_special_chars: Remove special characters
            remove_stopwords_flag: Remove stopwords
            lemmatize_flag: Lemmatize text
            keep_punctuation: Keep punctuation when removing special chars
            
        Returns:
            Dictionary with processed text and metadata
        """
        if not isinstance(text, str):
            text = str(text)
        
        original_text = text
        
        # Store metadata
        metadata = {
            'original_length': len(text),
            'urgency_score': self.detect_urgency(text),
            'scam_keyword_count': self.detect_scam_keywords(text)
        }
        
        # Apply preprocessing steps
        if remove_html_flag:
            text = self.remove_html(text)
        
        if lowercase:
            text = self.lowercase(text)
        
        if expand_contractions_flag:
            text = self.expand_contractions(text)
        
        if remove_urls_flag:
            text = self.remove_urls(text)
        
        if normalize_emails_flag:
            text = self.normalize_email(text)
        
        if remove_phones_flag:
            text = self.remove_phone_numbers(text)
        
        if remove_special_chars:
            text = self.remove_special_characters(text, keep_punctuation=keep_punctuation)
        
        if remove_stopwords_flag:
            text = self.remove_stopwords(text)
        
        if lemmatize_flag:
            if self.use_spacy:
                text = self.lemmatize_text_spacy(text)
            else:
                text = self.lemmatize_text(text)
        
        text = self.remove_extra_whitespace(text)
        
        # Store final metadata
        metadata['processed_length'] = len(text)
        metadata['processed_text'] = text
        metadata['tokens'] = self.tokenize(text)
        metadata['token_count'] = len(metadata['tokens'])
        
        return metadata
    
    def preprocess_batch(self, 
                         texts: List[str],
                         **kwargs) -> List[dict]:
        """
        Preprocess a batch of texts
        
        Args:
            texts: List of texts to preprocess
            **kwargs: Arguments to pass to preprocess method
            
        Returns:
            List of preprocessing results
        """
        results = []
        for text in texts:
            result = self.preprocess(text, **kwargs)
            results.append(result)
        
        return results


# Convenience function for quick preprocessing
def quick_preprocess(text: str) -> str:
    """
    Quick preprocessing with default settings
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    preprocessor = TextPreprocessor()
    result = preprocessor.preprocess(text)
    return result['processed_text']


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    test_text = """
    URGENT HIRING! Earn $5000 weekly working from home. No experience needed. 
    Visit http://scam-job.com or email us at gethiredquick@gmail.com. 
    Call us at 555-123-4567. <b>Apply NOW!</b> Limited spots available.
    """
    
    print("Original text:")
    print(test_text)
    print("\n" + "="*50 + "\n")
    
    result = preprocessor.preprocess(test_text)
    
    print("Processed text:")
    print(result['processed_text'])
    print("\n" + "="*50 + "\n")
    
    print("Metadata:")
    for key, value in result.items():
        if key != 'processed_text' and key != 'tokens':
            print(f"{key}: {value}")
