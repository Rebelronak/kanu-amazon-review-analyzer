import re
import logging
from datetime import datetime

class FakeReviewDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Fake review patterns
        self.fake_patterns = {
            'excessive_praise': {
                'words': ['amazing', 'incredible', 'best ever', 'perfect', 'outstanding', 'fantastic', 'mind blowing'],
                'weight': 2.0
            },
            'repetitive': {
                'patterns': [r'(\w+)\s+\1', 'very very', 'really really', 'amazing amazing', 'perfect perfect'],
                'weight': 3.0
            },
            'generic_phrases': {
                'words': ['must buy', 'highly recommend', 'best product', 'five stars', 'love love'],
                'weight': 1.5
            },
            'urgency': {
                'words': ['must have', 'dont miss', 'hurry', 'limited time'],
                'weight': 2.5
            }
        }
        
        # Genuine review indicators
        self.genuine_patterns = {
            'specific_details': {
                'words': ['purchased', 'delivery', 'packaging', 'installation', 'build quality', 'customer service'],
                'weight': 2.0
            },
            'balanced_opinion': {
                'words': ['however', 'but', 'although', 'mixed feelings', 'pros and cons'],
                'weight': 2.5
            },
            'constructive': {
                'words': ['could be better', 'improvement', 'suggestion', 'issue', 'problem'],
                'weight': 2.0
            },
            'temporal_context': {
                'words': ['been using', 'for weeks', 'for months', 'recently bought', 'past month'],
                'weight': 1.8
            }
        }
    
    def analyze_review(self, review_text):
        """Analyze review and return prediction and confidence"""
        if not review_text or len(review_text.strip()) < 20:
            return 1, 0.9  # Too short, likely fake
        
        text_lower = review_text.lower()
        fake_score = 0
        genuine_score = 0
        
        # Check fake patterns
        for category, data in self.fake_patterns.items():
            if category == 'repetitive':
                # Handle regex patterns
                for pattern in data['patterns']:
                    if isinstance(pattern, str) and pattern in text_lower:
                        fake_score += data['weight']
                    elif re.search(pattern, text_lower):
                        fake_score += data['weight']
            else:
                # Handle word patterns
                for word in data['words']:
                    if word in text_lower:
                        fake_score += data['weight']
        
        # Check genuine patterns
        for category, data in self.genuine_patterns.items():
            for word in data['words']:
                if word in text_lower:
                    genuine_score += data['weight']
        
        # Additional analysis
        
        # Length analysis
        if len(review_text) < 50:
            fake_score += 2.0
        elif len(review_text) > 200:
            genuine_score += 1.0
        
        # Exclamation analysis
        exclamations = review_text.count('!')
        if exclamations > 3:
            fake_score += exclamations * 0.5
        
        # Word diversity
        words = text_lower.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.6:
                fake_score += 1.5
            elif unique_ratio > 0.8:
                genuine_score += 1.0
        
        # Calculate prediction
        total_score = max(fake_score + genuine_score, 1)
        fake_probability = fake_score / total_score
        
        prediction = 1 if fake_probability > 0.45 else 0
        confidence = min(max(abs(fake_probability - 0.5) + 0.5, 0.6), 0.95)
        
        return prediction, confidence
    
    def get_analysis_details(self, review_text):
        """Get detailed analysis breakdown"""
        analysis = {
            'length': len(review_text),
            'word_count': len(review_text.split()),
            'exclamation_count': review_text.count('!'),
            'fake_indicators': [],
            'genuine_indicators': []
        }
        
        text_lower = review_text.lower()
        
        # Check for fake indicators
        for category, data in self.fake_patterns.items():
            found_words = []
            if category == 'repetitive':
                for pattern in data['patterns']:
                    if isinstance(pattern, str) and pattern in text_lower:
                        found_words.append(pattern)
                    elif re.search(pattern, text_lower):
                        found_words.append('repetitive pattern')
            else:
                for word in data['words']:
                    if word in text_lower:
                        found_words.append(word)
            
            if found_words:
                analysis['fake_indicators'].append({
                    'category': category,
                    'found': found_words,
                    'weight': data['weight']
                })
        
        # Check for genuine indicators
        for category, data in self.genuine_patterns.items():
            found_words = []
            for word in data['words']:
                if word in text_lower:
                    found_words.append(word)
            
            if found_words:
                analysis['genuine_indicators'].append({
                    'category': category,
                    'found': found_words,
                    'weight': data['weight']
                })
        
        return analysis