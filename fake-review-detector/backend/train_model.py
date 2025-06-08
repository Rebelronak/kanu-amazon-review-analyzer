import pandas as pd
import numpy as np
import joblib
import json
import os
import re
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
# Removed SVM import completely - too slow for large datasets
import warnings
warnings.filterwarnings('ignore')

class FakeReviewModelTrainer:
    """
    Advanced Model Training Class for Fake Review Detection
    Optimized for 8GB RAM with fast training
    """
    
    def __init__(self, config_path="model/config.json"):
        """
        Initialize the trainer with configuration
        """
        self.config = self.load_config(config_path)
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.best_vectorizer = None
        self.best_score = 0
        self.training_history = {}
        
    def load_config(self, config_path):
        """
        Load configuration from JSON file
        """
        try:
            with open(config_path) as f:
                config = json.load(f)
            print("✅ Configuration loaded successfully")
            return config
        except FileNotFoundError:
            print(f"❌ Config file not found at: {config_path}")
            sys.exit(1)
        except json.JSONDecodeError:
            print("❌ Invalid JSON in config file")
            sys.exit(1)
    
    def advanced_clean_text(self, text):
        """
        Advanced text cleaning for better model performance
        """
        if pd.isna(text) or not text:
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs and web addresses
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Fix common contractions
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
            "'m": " am", "it's": "it is", "that's": "that is"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove excessive punctuation but keep some structure
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Remove special characters but keep basic punctuation and apostrophes
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def load_and_preprocess_data(self):
        """
        Load and preprocess the dataset with memory optimization
        """
        print("🔍 Loading dataset...")
        try:
            # First, check file size and available memory
            file_path = self.config["data_path"]
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"📁 Dataset size: {file_size:.1f} MB")
            
            # Ultra-fast loading for 8GB RAM (reduced from 100K to 20K)
            if file_size > 100:  # If larger than 100MB
                print("⚠️  Large dataset detected! Using ultra-fast sample for 8GB RAM...")
                print("💡 This ensures fast training without memory issues")
                
                # Use 20K sample for ultra-fast training
                SAMPLE_SIZE = 20000  # Reduced from 100000
                print(f"📊 Loading {SAMPLE_SIZE:,} reviews (ultra-fast training)...")
                
                self.data = pd.read_csv(file_path, nrows=SAMPLE_SIZE)
                print(f"✅ Sample dataset loaded: {len(self.data):,} reviews")
                print(f"🧠 Memory usage: ~1-2GB (very safe for 8GB RAM)")
                print(f"⚡ Expected training time: 2-3 minutes")
                
            else:
                # Small dataset - load fully
                self.data = pd.read_csv(file_path)
                print(f"✅ Full dataset loaded: {len(self.data):,} reviews")
                
        except FileNotFoundError:
            print(f"❌ Dataset not found at: {self.config['data_path']}")
            print("Please run: python dataset/download_dataset.py")
            sys.exit(1)
        except MemoryError:
            print("❌ Not enough memory to load dataset!")
            print("💡 Try reducing SAMPLE_SIZE further in the code")
            sys.exit(1)
        
        # Check required columns
        required_columns = ['text', 'label']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            print(f"❌ Missing columns in dataset: {missing_columns}")
            print(f"Available columns: {list(self.data.columns)}")
            sys.exit(1)
        
        print("🧹 Preprocessing data...")
        
        # Remove duplicates
        initial_size = len(self.data)
        self.data = self.data.drop_duplicates(subset=['text'])
        print(f"📊 Removed {initial_size - len(self.data):,} duplicate reviews")
        
        # Remove empty reviews
        self.data = self.data.dropna(subset=['text', 'label'])
        self.data = self.data[self.data['text'].str.strip() != '']
        
        # Clean text
        print("🔧 Cleaning text data...")
        self.data['text_cleaned'] = self.data['text'].apply(self.advanced_clean_text)
        
        # Remove very short reviews (likely spam or uninformative)
        min_length = 10
        self.data = self.data[self.data['text_cleaned'].str.len() >= min_length]
        print(f"📊 Removed reviews shorter than {min_length} characters")
        
        # Display dataset statistics
        self.display_dataset_stats()
        
        return self.data
    
    def display_dataset_stats(self):
        """
        Display comprehensive dataset statistics
        """
        print("\n" + "="*50)
        print("📊 DATASET STATISTICS")
        print("="*50)
        
        print(f"Total Reviews: {len(self.data):,}")
        print(f"Label Distribution:")
        label_counts = self.data['label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(self.data)) * 100
            label_name = "Fake" if label == 1 else "Genuine"
            print(f"  {label_name}: {count:,} ({percentage:.1f}%)")
        
        # Text length statistics
        text_lengths = self.data['text_cleaned'].str.len()
        print(f"\nText Length Statistics:")
        print(f"  Average: {text_lengths.mean():.1f} characters")
        print(f"  Median: {text_lengths.median():.1f} characters")
        print(f"  Min: {text_lengths.min()} characters")
        print(f"  Max: {text_lengths.max()} characters")
        
        # Word count statistics
        word_counts = self.data['text_cleaned'].str.split().str.len()
        print(f"\nWord Count Statistics:")
        print(f"  Average: {word_counts.mean():.1f} words")
        print(f"  Median: {word_counts.median():.1f} words")
        print(f"  Min: {word_counts.min()} words")
        print(f"  Max: {word_counts.max()} words")
        
        print("="*50)
    
    def prepare_data_splits(self):
        """
        Prepare train/test splits with stratification
        """
        print("🔧 Preparing data splits...")
        
        X = self.data['text_cleaned']
        y = self.data['label']
        
        # Stratified split to maintain label distribution
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.config["test_size"], 
            random_state=self.config["random_state"],
            stratify=y
        )
        
        print(f"✅ Training set: {len(self.X_train):,} reviews")
        print(f"✅ Testing set: {len(self.X_test):,} reviews")
        
        # Display split statistics
        train_dist = self.y_train.value_counts(normalize=True)
        test_dist = self.y_test.value_counts(normalize=True)
        
        print(f"\nTraining set distribution:")
        for label, pct in train_dist.items():
            label_name = "Fake" if label == 1 else "Genuine"
            print(f"  {label_name}: {pct:.1%}")
        
        print(f"\nTesting set distribution:")
        for label, pct in test_dist.items():
            label_name = "Fake" if label == 1 else "Genuine"
            print(f"  {label_name}: {pct:.1%}")
    
    def create_vectorizers(self):
        """
        Create different vectorizer configurations (ultra-fast for 8GB RAM)
        """
        # Ultra-reduced features for fast training
        max_features = min(3000, self.config.get("max_features", 10000))  # Further reduced
        
        vectorizers = {
            'tfidf_basic': TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 1),
                min_df=2
            ),
            'tfidf_optimized': TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        }
        
        print(f"🔧 Vectorizers configured with max_features={max_features} (ultra-fast)")
        return vectorizers
    
    def create_models(self):
        """
        Create fast model configurations (NO SVM!)
        """
        models = {
            'logistic_regression': LogisticRegression(
                max_iter=500,  # Reduced iterations
                random_state=self.config["random_state"],
                class_weight='balanced',
                solver='liblinear'  # Fast solver
            ),
            'logistic_regression_tuned': LogisticRegression(
                max_iter=1000,
                random_state=self.config["random_state"],
                class_weight='balanced',
                C=1.0,
                solver='liblinear'
            ),
            'random_forest_fast': RandomForestClassifier(
                n_estimators=30,  # Reduced from 50
                random_state=self.config["random_state"],
                class_weight='balanced',
                max_depth=6,      # Reduced depth
                n_jobs=-1,        # Use all CPU cores
                max_features='sqrt'  # Faster feature selection
            ),
            'naive_bayes': MultinomialNB(
                alpha=1.0  # Very fast for text classification
            ),
            'gradient_boosting_fast': GradientBoostingClassifier(
                n_estimators=30,  # Reduced from 50
                learning_rate=0.1,
                max_depth=3,
                random_state=self.config["random_state"],
                subsample=0.8  # Faster training
            )
        }
        
        print("🚀 Fast models configured (NO SVM - ultra-fast training!)")
        return models
    
    def train_and_evaluate_models(self):
        """
        Train and evaluate multiple model configurations
        """
        print("\n🤖 Training and evaluating models...")
        print("="*60)
        
        vectorizers = self.create_vectorizers()
        models = self.create_models()
        
        results = []
        
        for vec_name, vectorizer in vectorizers.items():
            print(f"\n🔧 Testing vectorizer: {vec_name}")
            
            # Fit vectorizer on training data
            print("   📊 Vectorizing data...")
            X_train_vec = vectorizer.fit_transform(self.X_train)
            X_test_vec = vectorizer.transform(self.X_test)
            print(f"   ✅ Vector shape: {X_train_vec.shape}")
            
            for model_name, model in models.items():
                print(f"  🎯 Training model: {model_name}")
                
                try:
                    # Train model
                    model.fit(X_train_vec, self.y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test_vec)
                    y_pred_proba = model.predict_proba(X_test_vec)[:, 1]
                    
                    # Calculate metrics
                    accuracy = accuracy_score(self.y_test, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        self.y_test, y_pred, average='weighted'
                    )
                    auc_score = roc_auc_score(self.y_test, y_pred_proba)
                    
                    # Fast cross-validation (3-fold instead of 5)
                    cv_scores = cross_val_score(
                        model, X_train_vec, self.y_train, cv=3, scoring='accuracy'
                    )
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    result = {
                        'vectorizer': vec_name,
                        'model': model_name,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'auc_score': auc_score,
                        'cv_mean': cv_mean,
                        'cv_std': cv_std,
                        'model_obj': model,
                        'vectorizer_obj': vectorizer
                    }
                    
                    results.append(result)
                    
                    print(f"    ✅ Accuracy: {accuracy:.4f}")
                    print(f"    ✅ F1-Score: {f1:.4f}")
                    print(f"    ✅ AUC: {auc_score:.4f}")
                    print(f"    ✅ CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
                    
                except Exception as e:
                    print(f"    ❌ Failed to train {model_name}: {str(e)}")
                    continue
        
        # Find best model
        if results:
            # Sort by F1 score (good balance of precision and recall)
            best_result = max(results, key=lambda x: x['f1_score'])
            
            print(f"\n🏆 BEST MODEL FOUND:")
            print(f"   Vectorizer: {best_result['vectorizer']}")
            print(f"   Model: {best_result['model']}")
            print(f"   Accuracy: {best_result['accuracy']:.4f}")
            print(f"   Precision: {best_result['precision']:.4f}")
            print(f"   Recall: {best_result['recall']:.4f}")
            print(f"   F1-Score: {best_result['f1_score']:.4f}")
            print(f"   AUC Score: {best_result['auc_score']:.4f}")
            print(f"   CV Score: {best_result['cv_mean']:.4f} (±{best_result['cv_std']:.4f})")
            
            self.best_model = best_result['model_obj']
            self.best_vectorizer = best_result['vectorizer_obj']
            self.best_score = best_result['f1_score']
            self.training_history = best_result
            
            return results
        else:
            print("❌ No models were successfully trained!")
            return None
    
    def detailed_evaluation(self):
        """
        Perform detailed evaluation of the best model
        """
        if not self.best_model or not self.best_vectorizer:
            print("❌ No trained model available for evaluation")
            return
        
        print("\n📊 DETAILED MODEL EVALUATION")
        print("="*50)
        
        # Transform test data
        X_test_vec = self.best_vectorizer.transform(self.X_test)
        y_pred = self.best_model.predict(X_test_vec)
        y_pred_proba = self.best_model.predict_proba(X_test_vec)[:, 1]
        
        # Classification report
        print("\n📋 Classification Report:")
        target_names = ['Genuine', 'Fake']
        print(classification_report(self.y_test, y_pred, target_names=target_names))
        
        # Confusion Matrix
        print("\n🔍 Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"                Predicted")
        print(f"                Genuine  Fake")
        print(f"Actual Genuine    {cm[0,0]:4d}   {cm[0,1]:4d}")
        print(f"       Fake       {cm[1,0]:4d}   {cm[1,1]:4d}")
        
        # Additional metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        
        print(f"\n📈 Additional Metrics:")
        print(f"   True Positives (Fake detected correctly): {tp}")
        print(f"   True Negatives (Genuine detected correctly): {tn}")
        print(f"   False Positives (Genuine misclassified as Fake): {fp}")
        print(f"   False Negatives (Fake misclassified as Genuine): {fn}")
        print(f"   Sensitivity (Recall): {sensitivity:.4f}")
        print(f"   Specificity: {specificity:.4f}")
    
    def save_model(self):
        """
        Save the trained model and vectorizer
        """
        if not self.best_model or not self.best_vectorizer:
            print("❌ No trained model to save")
            return False
        
        print("\n💾 Saving model and vectorizer...")
        
        try:
            # Create model directory
            os.makedirs(os.path.dirname(self.config["model_path"]), exist_ok=True)
            
            # Save model and vectorizer
            joblib.dump(self.best_model, self.config["model_path"])
            joblib.dump(self.best_vectorizer, self.config["vectorizer_path"])
            
            # Save training metadata
            metadata = {
                'training_date': datetime.now().isoformat(),
                'dataset_size': len(self.data),
                'train_size': len(self.X_train),
                'test_size': len(self.X_test),
                'best_model_type': self.training_history['model'],
                'best_vectorizer_type': self.training_history['vectorizer'],
                'model_performance': {
                    'accuracy': float(self.training_history['accuracy']),
                    'precision': float(self.training_history['precision']),
                    'recall': float(self.training_history['recall']),
                    'f1_score': float(self.training_history['f1_score']),
                    'auc_score': float(self.training_history['auc_score']),
                    'cv_mean': float(self.training_history['cv_mean']),
                    'cv_std': float(self.training_history['cv_std'])
                },
                'optimization': 'ultra_fast_8gb_ram',
                'sample_size': '20k_reviews',
                'creator': 'Ronak Kanani',
                'config_used': self.config
            }
            
            metadata_path = os.path.join(os.path.dirname(self.config["model_path"]), "training_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Model saved to: {self.config['model_path']}")
            print(f"✅ Vectorizer saved to: {self.config['vectorizer_path']}")
            print(f"✅ Metadata saved to: {metadata_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            return False
    
    def train_complete_pipeline(self):
        """
        Run the complete training pipeline
        """
        print("🚀 Starting Ultra-Fast Training Pipeline")
        print("="*60)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Prepare data splits
        self.prepare_data_splits()
        
        # Train and evaluate models
        results = self.train_and_evaluate_models()
        
        if results:
            # Detailed evaluation
            self.detailed_evaluation()
            
            # Save best model
            if self.save_model():
                print("\n🎉 ULTRA-FAST TRAINING COMPLETED SUCCESSFULLY!")
                print("="*60)
                print(f"🏆 Best Model: {self.training_history['model']}")
                print(f"🏆 Best Vectorizer: {self.training_history['vectorizer']}")
                print(f"🏆 F1-Score: {self.best_score:.4f}")
                print(f"⚡ Sample Size: {len(self.data):,} reviews")
                print(f"🧠 Memory Optimized: 8GB RAM compatible")
                print(f"👨‍💻 Created by: Ronak Kanani")
                return True
            else:
                print("\n❌ Training completed but model saving failed!")
                return False
        else:
            print("\n❌ Training failed!")
            return False

# Legacy function for backward compatibility
def train():
    """
    Legacy training function - uses the new advanced trainer
    """
    trainer = FakeReviewModelTrainer()
    return trainer.train_complete_pipeline()

# Main execution
if __name__ == "__main__":
    print("🕵️‍♂️ Fake Review Detector - Ultra-Fast Training")
    print("="*60)
    print("⚡ Optimized for 8GB RAM - NO SVM - Fast Models Only")
    print("="*60)
    
    try:
        trainer = FakeReviewModelTrainer()
        success = trainer.train_complete_pipeline()
        
        if success:
            print("\n" + "="*60)
            print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("Next steps:")
            print("1. Run 'python app.py' to start the API server")
            print("2. Open the frontend to test predictions")
            print("3. Use 'python predict.py' for command-line testing")
            print("\n🔗 Created by: Ronak Kanani")
            print("📊 Ultra-fast 20K sample training completed!")
        else:
            print("\n" + "="*60)
            print("❌ TRAINING FAILED!")
            print("="*60)
            print("Please check the error messages above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during training: {str(e)}")
        sys.exit(1)