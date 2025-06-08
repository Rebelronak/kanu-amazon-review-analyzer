import joblib
import json
import re
import os
import sys
from datetime import datetime

class FakeReviewPredictor:
    """
    Advanced Fake Review Prediction Class
    """
    
    def __init__(self, config_path="model/config.json"):
        """
        Initialize the predictor with model and configuration
        """
        self.model = None
        self.vectorizer = None
        self.config = None
        self.load_config(config_path)
        self.load_models()
    
    def load_config(self, config_path):
        """
        Load configuration from JSON file
        """
        try:
            with open(config_path) as f:
                self.config = json.load(f)
            print("✅ Configuration loaded successfully")
        except FileNotFoundError:
            print(f"❌ Config file not found at: {config_path}")
            sys.exit(1)
        except json.JSONDecodeError:
            print("❌ Invalid JSON in config file")
            sys.exit(1)
    
    def load_models(self):
        """
        Load trained model and vectorizer
        """
        try:
            print("📦 Loading trained model...")
            self.model = joblib.load(self.config["model_path"])
            self.vectorizer = joblib.load(self.config["vectorizer_path"])
            print("✅ Models loaded successfully!")
        except FileNotFoundError as e:
            print(f"❌ Model files not found: {e}")
            print("Please train the model first using: python train_model.py")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            sys.exit(1)
    
    def clean_text(self, text):
        """
        Advanced text cleaning for better prediction accuracy
        """
        if not text:
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs and web addresses
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove excessive punctuation but keep some
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Remove special characters except basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_features(self, text):
        """
        Extract additional features from text that might indicate fake reviews
        """
        features = {}
        
        # Basic text statistics
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len([s for s in text.split('.') if s.strip()])
        
        # Exclamation and question marks (fake reviews often overuse these)
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Common fake review indicators
        fake_indicators = [
            'amazing', 'perfect', 'excellent', 'fantastic', 'incredible',
            'worst', 'terrible', 'awful', 'horrible', 'useless',
            'highly recommend', 'must buy', 'waste of money', 'dont buy'
        ]
        
        features['fake_indicator_count'] = sum(1 for indicator in fake_indicators if indicator in text.lower())
        
        # Repetitive words (fake reviews often repeat words)
        words = text.lower().split()
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        max_word_freq = max(word_freq.values()) if word_freq else 0
        features['max_word_repetition'] = max_word_freq
        features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
        
        return features
    
    def predict_review(self, review_text, include_features=False):
        """
        Predict if a review is fake or genuine
        """
        if not review_text or len(str(review_text).strip()) == 0:
            return {
                "error": "Empty review text provided",
                "prediction": None,
                "confidence": 0
            }
        
        # Clean the text
        cleaned_text = self.clean_text(review_text)
        
        if len(cleaned_text) < 5:
            return {
                "error": "Review text too short after cleaning",
                "prediction": None,
                "confidence": 0
            }
        
        try:
            # Vectorize the text
            X_vec = self.vectorizer.transform([cleaned_text])
            
            # Make prediction
            prediction = self.model.predict(X_vec)[0]
            probabilities = self.model.predict_proba(X_vec)[0]
            confidence = float(max(probabilities))
            
            # Determine result
            result = "Fake" if prediction == 1 else "Genuine"
            
            # Calculate confidence level
            if confidence > 0.8:
                confidence_level = "Very High"
            elif confidence > 0.7:
                confidence_level = "High"
            elif confidence > 0.6:
                confidence_level = "Medium"
            elif confidence > 0.5:
                confidence_level = "Low"
            else:
                confidence_level = "Very Low"
            
            # Generate recommendation
            if result == "Fake" and confidence > 0.7:
                recommendation = "❌ Do NOT Buy - High fake review probability"
                risk_level = "High Risk"
            elif result == "Fake" and confidence > 0.5:
                recommendation = "⚠️ Be Cautious - Possible fake reviews"
                risk_level = "Medium Risk"
            elif result == "Genuine" and confidence > 0.7:
                recommendation = "✅ Safe to Buy - Reviews appear genuine"
                risk_level = "Low Risk"
            else:
                recommendation = "🤔 Uncertain - Mixed signals in reviews"
                risk_level = "Unknown Risk"
            
            response = {
                "prediction": result,
                "confidence": round(confidence * 100, 2),
                "confidence_level": confidence_level,
                "recommendation": recommendation,
                "risk_level": risk_level,
                "original_length": len(review_text),
                "cleaned_length": len(cleaned_text),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add text features if requested
            if include_features:
                response["text_features"] = self.extract_features(cleaned_text)
            
            return response
            
        except Exception as e:
            return {
                "error": f"Prediction failed: {str(e)}",
                "prediction": None,
                "confidence": 0
            }
    
    def predict_batch(self, reviews, include_features=False):
        """
        Predict multiple reviews at once
        """
        if not reviews or not isinstance(reviews, list):
            return {"error": "Invalid reviews list provided"}
        
        if len(reviews) > 100:
            return {"error": "Maximum 100 reviews allowed per batch"}
        
        results = []
        fake_count = 0
        genuine_count = 0
        total_confidence = 0
        
        for i, review in enumerate(reviews):
            result = self.predict_review(review, include_features)
            result["index"] = i
            
            if result.get("prediction"):
                if result["prediction"] == "Fake":
                    fake_count += 1
                else:
                    genuine_count += 1
                total_confidence += result.get("confidence", 0)
            
            results.append(result)
        
        # Calculate batch statistics
        valid_predictions = fake_count + genuine_count
        avg_confidence = total_confidence / valid_predictions if valid_predictions > 0 else 0
        fake_percentage = (fake_count / valid_predictions * 100) if valid_predictions > 0 else 0
        
        # Overall recommendation
        if fake_percentage > 60:
            overall_recommendation = "❌ HIGH RISK - Many fake reviews detected"
        elif fake_percentage > 30:
            overall_recommendation = "⚠️ MEDIUM RISK - Some fake reviews detected"
        elif fake_percentage > 10:
            overall_recommendation = "🟡 LOW RISK - Few fake reviews detected"
        else:
            overall_recommendation = "✅ SAFE - Most reviews appear genuine"
        
        return {
            "batch_summary": {
                "total_reviews": len(reviews),
                "valid_predictions": valid_predictions,
                "fake_reviews": fake_count,
                "genuine_reviews": genuine_count,
                "fake_percentage": round(fake_percentage, 2),
                "average_confidence": round(avg_confidence, 2),
                "overall_recommendation": overall_recommendation
            },
            "individual_results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def analyze_product_reviews(self, product_reviews):
        """
        Analyze all reviews for a product and provide buying recommendation
        """
        if not product_reviews:
            return {"error": "No reviews provided"}
        
        batch_result = self.predict_batch(product_reviews, include_features=True)
        
        if "error" in batch_result:
            return batch_result
        
        summary = batch_result["batch_summary"]
        fake_percentage = summary["fake_percentage"]
        
        # Detailed analysis
        analysis = {
            "product_analysis": {
                "total_reviews_analyzed": summary["total_reviews"],
                "fake_review_count": summary["fake_reviews"],
                "genuine_review_count": summary["genuine_reviews"],
                "fake_review_percentage": fake_percentage,
                "confidence_score": summary["average_confidence"]
            },
            "buying_recommendation": {
                "should_buy": fake_percentage < 30,
                "confidence_level": "High" if summary["average_confidence"] > 70 else "Medium" if summary["average_confidence"] > 50 else "Low",
                "risk_assessment": summary["overall_recommendation"],
                "reasons": []
            },
            "detailed_results": batch_result["individual_results"]
        }
        
        # Add specific reasons for recommendation
        if fake_percentage > 50:
            analysis["buying_recommendation"]["reasons"].append("High percentage of fake reviews detected")
        if fake_percentage > 30:
            analysis["buying_recommendation"]["reasons"].append("Significant fake review presence")
        if summary["average_confidence"] < 60:
            analysis["buying_recommendation"]["reasons"].append("Low prediction confidence")
        if fake_percentage < 10:
            analysis["buying_recommendation"]["reasons"].append("Most reviews appear authentic")
        
        return analysis

# Global predictor instance
predictor = None

def initialize_predictor():
    """
    Initialize global predictor instance
    """
    global predictor
    if predictor is None:
        predictor = FakeReviewPredictor()
    return predictor

def predict_review(review_text):
    """
    Simple function for backward compatibility
    """
    pred = initialize_predictor()
    return pred.predict_review(review_text)

def predict_batch(reviews):
    """
    Simple function for batch prediction
    """
    pred = initialize_predictor()
    return pred.predict_batch(reviews)

# Example usage and testing
if __name__ == "__main__":
    print("🕵️‍♂️ Fake Review Detector - Prediction Module")
    print("=" * 50)
    
    try:
        # Initialize predictor
        pred = FakeReviewPredictor()
        
        while True:
            print("\nChoose an option:")
            print("1. Analyze single review")
            print("2. Analyze multiple reviews")
            print("3. Test with sample reviews")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                review = input("\nEnter a review to analyze: ").strip()
                if review:
                    result = pred.predict_review(review, include_features=True)
                    print("\n" + "="*30)
                    print("ANALYSIS RESULT:")
                    print("="*30)
                    if "error" in result:
                        print(f"❌ Error: {result['error']}")
                    else:
                        print(f"📊 Prediction: {result['prediction']}")
                        print(f"🎯 Confidence: {result['confidence']}%")
                        print(f"📈 Confidence Level: {result['confidence_level']}")
                        print(f"💡 Recommendation: {result['recommendation']}")
                        print(f"⚠️ Risk Level: {result['risk_level']}")
                        if 'text_features' in result:
                            features = result['text_features']
                            print(f"📝 Word Count: {features['word_count']}")
                            print(f"📏 Character Length: {features['length']}")
                            print(f"🔤 Unique Word Ratio: {features['unique_word_ratio']:.2f}")
                else:
                    print("❌ No review entered!")
            
            elif choice == "2":
                print("\nEnter reviews (one per line, press Enter twice to finish):")
                reviews = []
                while True:
                    review = input().strip()
                    if not review:
                        break
                    reviews.append(review)
                
                if reviews:
                    result = pred.predict_batch(reviews)
                    if "error" in result:
                        print(f"❌ Error: {result['error']}")
                    else:
                        summary = result['batch_summary']
                        print("\n" + "="*40)
                        print("BATCH ANALYSIS RESULT:")
                        print("="*40)
                        print(f"📊 Total Reviews: {summary['total_reviews']}")
                        print(f"🔴 Fake Reviews: {summary['fake_reviews']}")
                        print(f"🟢 Genuine Reviews: {summary['genuine_reviews']}")
                        print(f"📈 Fake Percentage: {summary['fake_percentage']}%")
                        print(f"🎯 Average Confidence: {summary['average_confidence']}%")
                        print(f"💡 Overall: {summary['overall_recommendation']}")
                else:
                    print("❌ No reviews entered!")
            
            elif choice == "3":
                # Test with sample reviews
                sample_reviews = [
                    "This product is absolutely amazing! Best purchase ever!",
                    "Terrible quality. Waste of money. Don't buy this.",
                    "Good value for money. Works as expected.",
                    "Perfect! Perfect! Perfect! Buy it now!",
                    "Had some issues initially but customer service helped resolve them."
                ]
                
                print("\n🧪 Testing with sample reviews...")
                result = pred.predict_batch(sample_reviews)
                
                if "error" not in result:
                    summary = result['batch_summary']
                    print("\n" + "="*40)
                    print("SAMPLE TEST RESULTS:")
                    print("="*40)
                    print(f"📊 Total Reviews: {summary['total_reviews']}")
                    print(f"🔴 Fake Reviews: {summary['fake_reviews']}")
                    print(f"🟢 Genuine Reviews: {summary['genuine_reviews']}")
                    print(f"📈 Fake Percentage: {summary['fake_percentage']}%")
                    print(f"💡 Overall: {summary['overall_recommendation']}")
                    
                    print("\nIndividual Results:")
                    for i, res in enumerate(result['individual_results']):
                        if 'prediction' in res and res['prediction']:
                            print(f"{i+1}. {res['prediction']} ({res['confidence']}%) - {sample_reviews[i][:50]}...")
            
            elif choice == "4":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice! Please enter 1-4.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")