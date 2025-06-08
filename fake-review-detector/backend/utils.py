import re
import requests
import time
import random
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
import json
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextCleaner:
    """
    Advanced text cleaning utilities for review processing
    """
    
    @staticmethod
    def basic_clean(text):
        """
        Basic text cleaning function
        """
        if not text:
            return ""
        
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    @staticmethod
    def advanced_clean(text):
        """
        Advanced text cleaning with better preprocessing
        """
        if not text or pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs and web addresses
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Handle contractions
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
            "'m": " am", "it's": "it is", "that's": "that is",
            "i'm": "i am", "you're": "you are", "we're": "we are",
            "they're": "they are", "i've": "i have", "you've": "you have",
            "we've": "we have", "they've": "they have", "i'll": "i will",
            "you'll": "you will", "we'll": "we will", "they'll": "they will",
            "i'd": "i would", "you'd": "you would", "we'd": "we would",
            "they'd": "they would", "isn't": "is not", "aren't": "are not",
            "wasn't": "was not", "weren't": "were not", "haven't": "have not",
            "hasn't": "has not", "hadn't": "had not", "won't": "will not",
            "wouldn't": "would not", "don't": "do not", "doesn't": "does not",
            "didn't": "did not", "can't": "cannot", "couldn't": "could not",
            "shouldn't": "should not", "mightn't": "might not",
            "mustn't": "must not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Fix excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def extract_features(text):
        """
        Extract features that might indicate fake reviews
        """
        if not text:
            return {}
        
        features = {}
        
        # Basic statistics
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len([s for s in text.split('.') if s.strip()])
        features['avg_word_length'] = sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0
        
        # Punctuation analysis
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_count'] = sum(1 for c in text if c.isupper())
        features['caps_ratio'] = features['caps_count'] / len(text) if text else 0
        
        # Sentiment-related features
        positive_words = [
            'amazing', 'excellent', 'fantastic', 'perfect', 'incredible',
            'wonderful', 'outstanding', 'superb', 'brilliant', 'magnificent',
            'love', 'best', 'great', 'awesome', 'highly recommend'
        ]
        
        negative_words = [
            'terrible', 'awful', 'horrible', 'worst', 'useless',
            'garbage', 'trash', 'pathetic', 'disappointing', 'waste',
            'hate', 'regret', 'never buy', 'dont buy', 'avoid'
        ]
        
        features['positive_word_count'] = sum(1 for word in positive_words if word in text.lower())
        features['negative_word_count'] = sum(1 for word in negative_words if word in text.lower())
        
        # Repetition analysis
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 2:  # Ignore very short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        features['max_word_repetition'] = max(word_freq.values()) if word_freq else 0
        features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
        
        # Fake review indicators
        fake_indicators = [
            'highly recommend', 'must buy', 'buy it now', 'perfect product',
            'amazing quality', 'super fast delivery', 'exceeded expectations',
            'waste of money', 'total garbage', 'complete scam', 'fake product'
        ]
        
        features['fake_indicator_count'] = sum(1 for indicator in fake_indicators if indicator in text.lower())
        
        return features

class ProductLinkParser:
    """
    Parse and extract information from product URLs
    """
    
    @staticmethod
    def parse_amazon_url(url):
        """
        Extract product information from Amazon URL
        """
        try:
            parsed_url = urlparse(url)
            
            # Extract ASIN (Amazon Standard Identification Number)
            path_parts = parsed_url.path.split('/')
            asin = None
            
            # Look for ASIN in different URL formats
            if '/dp/' in url:
                asin_index = path_parts.index('dp') + 1
                if asin_index < len(path_parts):
                    asin = path_parts[asin_index]
            elif '/gp/product/' in url:
                asin_index = path_parts.index('product') + 1
                if asin_index < len(path_parts):
                    asin = path_parts[asin_index]
            
            # Extract other parameters
            query_params = parse_qs(parsed_url.query)
            
            return {
                'platform': 'amazon',
                'asin': asin,
                'domain': parsed_url.netloc,
                'full_url': url,
                'is_valid': asin is not None
            }
        except Exception as e:
            logger.error(f"Error parsing Amazon URL: {e}")
            return {'platform': 'amazon', 'is_valid': False, 'error': str(e)}
    
    @staticmethod
    def parse_flipkart_url(url):
        """
        Extract product information from Flipkart URL
        """
        try:
            parsed_url = urlparse(url)
            
            # Extract product ID from Flipkart URL
            product_id = None
            path_parts = parsed_url.path.split('/')
            
            # Flipkart URLs usually have product ID at the end
            if path_parts:
                product_id = path_parts[-1] if path_parts[-1] else path_parts[-2]
            
            # Extract pid parameter if available
            query_params = parse_qs(parsed_url.query)
            pid = query_params.get('pid', [None])[0]
            
            return {
                'platform': 'flipkart',
                'product_id': product_id,
                'pid': pid,
                'domain': parsed_url.netloc,
                'full_url': url,
                'is_valid': product_id is not None
            }
        except Exception as e:
            logger.error(f"Error parsing Flipkart URL: {e}")
            return {'platform': 'flipkart', 'is_valid': False, 'error': str(e)}
    
    @staticmethod
    def identify_platform(url):
        """
        Identify the e-commerce platform from URL
        """
        url_lower = url.lower()
        
        if 'amazon.' in url_lower:
            return 'amazon'
        elif 'flipkart.' in url_lower:
            return 'flipkart'
        else:
            return 'unknown'

class ReviewScraper:
    """
    Web scraper for extracting reviews from product pages
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_page_content(self, url, max_retries=3):
        """
        Get page content with retry logic
        """
        for attempt in range(max_retries):
            try:
                # Add random delay to avoid being blocked
                time.sleep(random.uniform(1, 3))
                
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                return response.text
            
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(random.uniform(2, 5))
        
        return None
    
    def extract_amazon_reviews(self, html_content):
        """
        Extract reviews from Amazon product page HTML
        Note: This is a basic implementation. Amazon's structure changes frequently.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            reviews = []
            
            # Common selectors for Amazon reviews (may need updates)
            review_selectors = [
                '[data-hook="review-body"] span',
                '.review-text',
                '.cr-original-review-text',
                '[data-hook="review-body"]'
            ]
            
            for selector in review_selectors:
                review_elements = soup.select(selector)
                if review_elements:
                    for element in review_elements:
                        review_text = element.get_text(strip=True)
                        if review_text and len(review_text) > 10:
                            reviews.append(review_text)
                    break
            
            return list(set(reviews))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting Amazon reviews: {e}")
            return []
    
    def extract_flipkart_reviews(self, html_content):
        """
        Extract reviews from Flipkart product page HTML
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            reviews = []
            
            # Common selectors for Flipkart reviews
            review_selectors = [
                '.t-ZTKy',
                '.qwjRop',
                '._2-N8zT',
                '.ZmyHeo'
            ]
            
            for selector in review_selectors:
                review_elements = soup.select(selector)
                if review_elements:
                    for element in review_elements:
                        review_text = element.get_text(strip=True)
                        if review_text and len(review_text) > 10:
                            reviews.append(review_text)
                    break
            
            return list(set(reviews))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting Flipkart reviews: {e}")
            return []
    
    def scrape_reviews_from_url(self, url, max_reviews=20):
        """
        Scrape reviews from a product URL
        """
        try:
            # Identify platform
            platform = ProductLinkParser.identify_platform(url)
            
            if platform == 'unknown':
                return {
                    'success': False,
                    'error': 'Unsupported platform. Only Amazon and Flipkart are supported.',
                    'reviews': []
                }
            
            # Get page content
            logger.info(f"Scraping reviews from {platform} URL...")
            html_content = self.get_page_content(url)
            
            if not html_content:
                return {
                    'success': False,
                    'error': 'Failed to fetch page content',
                    'reviews': []
                }
            
            # Extract reviews based on platform
            if platform == 'amazon':
                reviews = self.extract_amazon_reviews(html_content)
            elif platform == 'flipkart':
                reviews = self.extract_flipkart_reviews(html_content)
            else:
                reviews = []
            
            # Limit number of reviews
            reviews = reviews[:max_reviews]
            
            # Clean reviews
            cleaned_reviews = []
            for review in reviews:
                cleaned = TextCleaner.advanced_clean(review)
                if len(cleaned) > 10:  # Filter out very short reviews
                    cleaned_reviews.append(cleaned)
            
            return {
                'success': True,
                'platform': platform,
                'total_reviews': len(cleaned_reviews),
                'reviews': cleaned_reviews,
                'url': url,
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error scraping reviews: {e}")
            return {
                'success': False,
                'error': str(e),
                'reviews': []
            }

class DataProcessor:
    """
    Data processing utilities for the fake review detector
    """
    
    @staticmethod
    def process_review_batch(reviews, include_features=False):
        """
        Process a batch of reviews
        """
        processed_reviews = []
        
        for i, review in enumerate(reviews):
            processed = {
                'index': i,
                'original_text': review,
                'cleaned_text': TextCleaner.advanced_clean(review),
                'length': len(review),
                'word_count': len(review.split()) if review else 0
            }
            
            if include_features:
                processed['features'] = TextCleaner.extract_features(review)
            
            processed_reviews.append(processed)
        
        return processed_reviews
    
    @staticmethod
    def save_results_to_json(results, filename):
        """
        Save results to JSON file
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    @staticmethod
    def load_results_from_json(filename):
        """
        Load results from JSON file
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return None

# Legacy functions for backward compatibility
def clean_text(text):
    """Legacy function - uses advanced cleaning"""
    return TextCleaner.advanced_clean(text)

def extract_reviews_from_html(html, platform="amazon"):
    """Legacy function - uses new scraper"""
    scraper = ReviewScraper()
    if platform.lower() == "amazon":
        return scraper.extract_amazon_reviews(html)
    elif platform.lower() == "flipkart":
        return scraper.extract_flipkart_reviews(html)
    else:
        return []

# Main utility functions
def scrape_product_reviews(url, max_reviews=20):
    """
    Main function to scrape reviews from a product URL
    """
    scraper = ReviewScraper()
    return scraper.scrape_reviews_from_url(url, max_reviews)

def parse_product_url(url):
    """
    Parse product URL and extract information
    """
    platform = ProductLinkParser.identify_platform(url)
    
    if platform == 'amazon':
        return ProductLinkParser.parse_amazon_url(url)
    elif platform == 'flipkart':
        return ProductLinkParser.parse_flipkart_url(url)
    else:
        return {'platform': 'unknown', 'is_valid': False}

def analyze_text_features(text):
    """
    Analyze text and extract features
    """
    return TextCleaner.extract_features(text)

# Example usage and testing
if __name__ == "__main__":
    print("🛠️ Fake Review Detector - Utilities Module")
    print("="*50)
    
    # Test text cleaning
    sample_text = "This is an AMAZING product!!! I can't believe how PERFECT it is! Check http://example.com for more info. It's the BEST purchase I've ever made!!!"
    
    print("Original text:")
    print(f"'{sample_text}'")
    print("\nCleaned text:")
    print(f"'{TextCleaner.advanced_clean(sample_text)}'")
    
    # Test feature extraction
    features = TextCleaner.extract_features(sample_text)
    print("\nExtracted features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    # Test URL parsing
    print("\n" + "="*30)
    print("URL PARSING TESTS")
    print("="*30)
    
    test_urls = [
        "https://www.amazon.in/dp/B08N5WRWNW",
        "https://www.flipkart.com/apple-iphone-13/p/itm6ac6485b58b50",
        "https://www.example.com/product"
    ]
    
    for url in test_urls:
        result = parse_product_url(url)
        print(f"\nURL: {url}")
        print(f"Platform: {result.get('platform', 'unknown')}")
        print(f"Valid: {result.get('is_valid', False)}")
        if result.get('asin'):
            print(f"ASIN: {result['asin']}")
        if result.get('product_id'):
            print(f"Product ID: {result['product_id']}")
    
    print("\n🎉 Utilities module test completed!")