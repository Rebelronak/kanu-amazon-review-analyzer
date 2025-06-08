import random
import re
from datetime import datetime

class ReviewAnalyzer:
    def __init__(self):
        self.fake_indicators = [
            'amazing', 'perfect', 'best ever', 'life changing',
            'highly recommend', 'five stars', 'must buy'
        ]
    
    def analyze_reviews(self, reviews):
        """Analyze reviews and return predictions"""
        analyzed_reviews = []
        
        for review in reviews:
            # Simple fake detection logic
            text = review.get('text', '').lower()
            rating = review.get('rating', 3)
            
            # Count fake indicators
            fake_score = 0
            for indicator in self.fake_indicators:
                if indicator in text:
                    fake_score += 1
            
            # Simple prediction logic
            if fake_score >= 2 or rating == 5:
                prediction = 'Fake'
                confidence = min(85 + fake_score * 5, 95)
            else:
                prediction = 'Genuine'
                confidence = min(75 + random.randint(5, 15), 90)
            
            analyzed_reviews.append({
                'text': review.get('text', ''),
                'prediction': prediction,
                'confidence': confidence,
                'rating': rating
            })
        
        return analyzed_reviews
    
    def generate_recommendation(self, analysis_results):
        """Generate recommendation based on analysis"""
        if not analysis_results:
            return {
                'decision': 'INSUFFICIENT DATA',
                'reason': 'No reviews available for analysis.'
            }
        
        total_reviews = len(analysis_results)
        fake_reviews = len([r for r in analysis_results if r['prediction'] == 'Fake'])
        fake_percentage = (fake_reviews / total_reviews) * 100
        
        if fake_percentage < 20:
            return {
                'decision': 'LOW RISK - Safe to Purchase',
                'reason': f'Only {fake_percentage:.1f}% of reviews appear suspicious. This product seems trustworthy.'
            }
        elif fake_percentage < 50:
            return {
                'decision': 'MEDIUM RISK - Purchase with Caution',
                'reason': f'{fake_percentage:.1f}% of reviews appear suspicious. Consider reading reviews carefully.'
            }
        else:
            return {
                'decision': 'HIGH RISK - Avoid Purchase',
                'reason': f'{fake_percentage:.1f}% of reviews appear suspicious. This product may have fake reviews.'
            }