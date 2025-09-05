#!/usr/bin/env python3
"""
AI Phishing & Malware Email Detector - Kaggle Dataset Training Script
====================================================================
Modified version to work with real phishing email datasets from Kaggle
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import re
import os
from datetime import datetime

class MultiDatasetPhishingTrainer:
    def __init__(self):
        self.email_model = None
        self.vectorizer = None
        self.scaler = None

    def load_enron_dataset(self, csv_path="Enron.csv"):
        """Load Enron dataset (legitimate emails)"""
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ Enron dataset loaded: {len(df)} rows")
            print(f"Columns: {list(df.columns)}")  # 👈 show available columns

            # Detect the right text column
            possible_text_cols = ["message", "body", "content", "text", "email"]
            text_col = next((c for c in possible_text_cols if c in df.columns), None)

            if not text_col:
                raise ValueError("No valid text column found in Enron dataset")

            df = df.rename(columns={text_col: "text"})
            df["is_phishing"] = 0
            return df[["text", "is_phishing"]]
        except Exception as e:
            print(f"❌ Error loading Enron dataset: {e}")
            return pd.DataFrame()

    def load_nazario_dataset(self, csv_path="Nazario.csv"):
        """Load Nazario dataset (phishing emails)"""
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ Nazario dataset loaded: {len(df)} rows")
            print(f"Columns: {list(df.columns)}")  # 👈 show available columns

            # Detect the right text column
            possible_text_cols = ["body", "email_text", "message", "content", "text"]
            text_col = next((c for c in possible_text_cols if c in df.columns), None)

            if not text_col:
                raise ValueError("No valid text column found in Nazario dataset")

            df = df.rename(columns={text_col: "text"})
            df["is_phishing"] = 1
            return df[["text", "is_phishing"]]
        except Exception as e:
            print(f"❌ Error loading Nazario dataset: {e}")
            return pd.DataFrame()


    def combine_datasets(self, enron_path="Enron.csv", nazario_path="Nazario.csv"):
        """Combine Enron (ham) and Nazario (phish) datasets"""
        enron_df = self.load_enron_dataset(enron_path)
        nazario_df = self.load_nazario_dataset(nazario_path)
        df = pd.concat([enron_df, nazario_df], ignore_index=True)
        print(f"📊 Combined dataset: {len(df)} rows ({df['is_phishing'].value_counts().to_dict()})")
        return df
    
    def test_specific_email(self, text):
     """Test the trained model on a specific email text"""
     if not hasattr(self, "model") or not hasattr(self, "vectorizer"):
        print("⚠️ Model or vectorizer not found. Train the model first.")
        return

     X_test = self.vectorizer.transform([text])
     prediction = self.model.predict(X_test)[0]
     label = "Phishing" if prediction == 1 else "Legitimate"
     print("\n🔎 Test Specific Email:")
     print(f"Text: {text[:200]}...")  # print only first 200 chars
     print(f"Prediction: {label}")


    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s!?.,]', ' ', text)
        return text.strip()

    def prepare_features(self, df):
       print("Extracting TF-IDF features...")
       vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
       X = vectorizer.fit_transform(df["text"])
       y = df["is_phishing"].values

    # ✅ Save vectorizer to self
       self.vectorizer = vectorizer  

       return X, y


    def train_email_model(self, X, y):
       print("Training phishing detection model...")
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

       model = LogisticRegression(max_iter=1000)
       model.fit(X_train, y_train)
 
    # ✅ Save model to self so test_specific_email can access it
       self.model = model  

       print("Evaluating model...")
       y_pred = model.predict(X_test)
       print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
       print("Confusion Matrix:")
       print(confusion_matrix(y_test, y_pred))

    def save_models(self):
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.email_model, "models/email_phishing_model.pkl")
        joblib.dump(self.vectorizer, "models/email_vectorizer.pkl")
        joblib.dump(self.scaler, "models/email_scaler.pkl")
        print("✅ Models saved in /models folder")

class KagglePhishingDetectorTrainer:
    def __init__(self):
        self.email_model = None
        self.file_model = None
        self.vectorizer = None
        self.scaler = None
        self.label_encoder = None
        
    def load_kaggle_dataset(self, csv_path, text_column, label_column):
        """
        Load phishing email dataset from Kaggle CSV
        
        Args:
            csv_path: Path to the CSV file
            text_column: Name of column containing email text
            label_column: Name of column containing labels
        """
        print(f"Loading dataset from: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            print(f"Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns")
            print(f"Columns available: {list(df.columns)}")
            
            # Check if specified columns exist
            if text_column not in df.columns:
                print(f"Error: Column '{text_column}' not found in dataset")
                print(f"Available columns: {list(df.columns)}")
                return None
                
            if label_column not in df.columns:
                print(f"Error: Column '{label_column}' not found in dataset")
                print(f"Available columns: {list(df.columns)}")
                return None
            
            # Clean the data
            df = df.dropna(subset=[text_column, label_column])
            print(f"After removing null values: {len(df)} rows")
            
            # Show label distribution
            print(f"\nLabel distribution:")
            print(df[label_column].value_counts())
            
            # Encode labels if they're strings
            if df[label_column].dtype == 'object':
                self.label_encoder = LabelEncoder()
                df['is_phishing'] = self.label_encoder.fit_transform(df[label_column])
                print(f"Label mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
            else:
                df['is_phishing'] = df[label_column]
            
            # Rename text column for consistency
            df['text'] = df[text_column]
            
            # Show some sample data
            print(f"\nSample data:")
            for i in range(min(3, len(df))):
                print(f"Text: {df.iloc[i]['text'][:100]}...")
                print(f"Label: {df.iloc[i]['is_phishing']}")
                print("-" * 50)
            
            return df[['text', 'is_phishing']]
            
        except FileNotFoundError:
            print(f"Error: File '{csv_path}' not found")
            return None
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return None
    
    def preprocess_text(self, text):
        """Clean and preprocess email text"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s!?.,]', ' ', text)
        
        return text.strip()
    
    def extract_improved_email_features(self, email_text):
        """Extract improved features with better phishing indicators"""
        features = {}
        
        # Preprocess text
        email_text = self.preprocess_text(email_text)
        text_lower = email_text.lower()
        
        if not email_text:
            # Return zero features for empty text
            return {
                'length': 0, 'word_count': 0, 'exclamation_count': 0, 'question_count': 0,
                'uppercase_ratio': 0, 'urgency_score': 0, 'security_terms': 0, 'money_terms': 0,
                'action_requests': 0, 'url_count': 0, 'suspicious_phrases': 0, 'business_terms': 0,
                'spelling_errors': 0
            }
        
        # Basic text features
        features['length'] = len(email_text)
        features['word_count'] = len(email_text.split())
        features['exclamation_count'] = email_text.count('!')
        features['question_count'] = email_text.count('?')
        features['uppercase_ratio'] = sum(1 for c in email_text if c.isupper()) / len(email_text)
        
        # More specific suspicious patterns
        urgent_words = ['urgent', 'immediate', 'asap', 'emergency', 'expire', 'expires', 'deadline']
        features['urgency_score'] = sum(1 for word in urgent_words if word in text_lower)
        
        # Account/security related terms
        security_terms = ['verify', 'confirm', 'suspended', 'locked', 'security alert', 'unauthorized']
        features['security_terms'] = sum(1 for term in security_terms if term in text_lower)
        
        # Financial incentives
        money_terms = ['winner', 'prize', 'won', 'lottery', 'refund', 'discount', 'free', 'claim']
        features['money_terms'] = sum(1 for term in money_terms if term in text_lower)
        
        # Action requests
        action_words = ['click', 'download', 'install', 'update', 'enter', 'submit', 'provide']
        features['action_requests'] = sum(1 for word in action_words if word in text_lower)
        
        # URL and link patterns
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        features['url_count'] = len(re.findall(url_pattern, email_text))
        
        # Suspicious phrases
        suspicious_phrases = ['click here', 'click now', 'act now', 'limited time', 'expires soon', 
                             'verify now', 'confirm now', 'update now', 'download now']
        features['suspicious_phrases'] = sum(1 for phrase in suspicious_phrases if phrase in text_lower)
        
        # Legitimate business indicators
        business_terms = ['meeting', 'conference', 'agenda', 'deadline', 'project', 'team', 
                         'schedule', 'appointment', 'invoice', 'report']
        features['business_terms'] = sum(1 for term in business_terms if term in text_lower)
        
        # Grammar and spelling quality
        features['spelling_errors'] = self.count_potential_spelling_errors(email_text)
        
        return features
    
    def count_potential_spelling_errors(self, text):
        """Simple heuristic for potential spelling errors"""
        words = re.findall(r'\b\w+\b', text.lower())
        error_count = 0
        
        for word in words:
            if len(word) > 3:
                # Repeated characters (like 'winnner', 'hurrrry')
                if re.search(r'(.)\1{2,}', word):
                    error_count += 1
                # Mixed case in middle of words (like 'WinNer')
                if re.search(r'[a-z][A-Z][a-z]', word):
                    error_count += 1
                    
        return error_count
    
    def prepare_features(self, df):
        """Prepare features from the dataset"""
        print("Extracting features from emails...")
        
        # Extract features for all emails
        features_list = []
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processing email {idx}/{len(df)}")
            
            features = self.extract_improved_email_features(row['text'])
            features_list.append(features)
        
        # Create features DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Add text and labels
        features_df['text'] = df['text'].values
        features_df['is_phishing'] = df['is_phishing'].values
        
        print("Feature extraction completed!")
        print(f"Feature statistics:")
        print(features_df.describe())
        
        return features_df
    
    def train_email_model(self, df):
        """Train an improved ensemble email phishing detection model"""
        print("Training email phishing detection model...")
        
        # Prepare features
        feature_columns = ['length', 'word_count', 'exclamation_count', 'question_count', 
                          'uppercase_ratio', 'urgency_score', 'security_terms', 'money_terms',
                          'action_requests', 'url_count', 'suspicious_phrases', 'business_terms',
                          'spelling_errors']
        
        X_features = df[feature_columns]
        X_text = df['text']
        y = df['is_phishing']
        
        print(f"Training on {len(df)} samples")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Increased for real dataset
            stop_words='english',
            ngram_range=(1, 2),
            min_df=5,  # Ignore rare terms
            max_df=0.95  # Ignore very common terms
        )
        
        # Preprocess text data
        X_text_clean = X_text.fillna('').apply(self.preprocess_text)
        
        # Combine text features with numerical features
        print("Creating TF-IDF features...")
        X_text_tfidf = self.vectorizer.fit_transform(X_text_clean).toarray()
        
        # Scale numerical features
        self.scaler = StandardScaler()
        X_features_scaled = self.scaler.fit_transform(X_features)
        
        X_combined = np.hstack([X_text_tfidf, X_features_scaled])
        
        print(f"Combined feature matrix shape: {X_combined.shape}")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create ensemble model
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        
        # Use voting classifier
        self.email_model = VotingClassifier(
            estimators=[('rf', rf), ('lr', lr)],
            voting='soft'
        )
        
        # Train model
        print("Training ensemble model...")
        self.email_model.fit(X_train, y_train)
        
        # Evaluate with cross-validation
        print("Evaluating model...")
        cv_scores = cross_val_score(self.email_model, X_train, y_train, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Test set evaluation
        y_pred = self.email_model.predict(X_test)
        y_pred_proba = self.email_model.predict_proba(X_test)
        
        print(f"Email Model Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print("\nEmail Model Classification Report:")
        
        # Create target names based on label encoder
        if self.label_encoder:
            target_names = self.label_encoder.classes_
        else:
            target_names = ['Legitimate', 'Phishing']
        
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return self.email_model
    
    def test_specific_email(self, email_text):
        """Test a specific email and show feature breakdown"""
        features = self.extract_improved_email_features(email_text)
        
        print(f"\n🔍 Analyzing email: '{email_text[:60]}...'")
        print("Feature breakdown:")
        for feature, value in features.items():
            print(f"  {feature}: {value}")
        
        # Prepare features for prediction
        feature_columns = ['length', 'word_count', 'exclamation_count', 'question_count', 
                          'uppercase_ratio', 'urgency_score', 'security_terms', 'money_terms',
                          'action_requests', 'url_count', 'suspicious_phrases', 'business_terms',
                          'spelling_errors']
        
        X_features = np.array([[features[col] for col in feature_columns]])
        X_text_tfidf = self.vectorizer.transform([self.preprocess_text(email_text)]).toarray()
        X_features_scaled = self.scaler.transform(X_features)
        X_combined = np.hstack([X_text_tfidf, X_features_scaled])
        
        # Make prediction
        prediction = self.email_model.predict(X_combined)[0]
        probability = self.email_model.predict_proba(X_combined)[0]
        
        # Map prediction back to original labels
        if self.label_encoder:
            pred_label = self.label_encoder.inverse_transform([prediction])[0]
            print(f"\n📊 Prediction: {pred_label}")
        else:
            print(f"\n📊 Prediction: {'⚠️ PHISHING' if prediction else '✅ LEGITIMATE'}")
        
        print(f"Confidence: {max(probability):.3f}")
        print(f"Probabilities: {probability}")
        
        return prediction, probability
    
    def save_models(self):
        """Save trained models and preprocessing components"""
        os.makedirs('models', exist_ok=True)
        
        # Save all components
        joblib.dump(self.email_model, 'models/email_phishing_model.pkl')
        joblib.dump(self.vectorizer, 'models/email_vectorizer.pkl')
        joblib.dump(self.scaler, 'models/email_scaler.pkl')
        
        if self.label_encoder:
            joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        
        print("\nModels saved successfully:")
        print("- models/email_phishing_model.pkl")
        print("- models/email_vectorizer.pkl") 
        print("- models/email_scaler.pkl")
        if self.label_encoder:
            print("- models/label_encoder.pkl")



def main():
    print("🔒 AI Phishing Email Detector - Enron + Nazario Training")
    print("=" * 60)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    trainer = MultiDatasetPhishingTrainer()

    # Load datasets
    df = trainer.combine_datasets("Enron.csv", "Nazario.csv")
    if df is None or df.empty:
        print("❌ No datasets loaded. Exiting.")
        return

    # Feature engineering → get X and y
    X, y = trainer.prepare_features(df)

    # Train model
    trainer.train_email_model(X, y)

    # Quick test on first sample
    if len(df) > 0:
        trainer.test_specific_email(df.iloc[0]['text'])

    # Save models
    trainer.save_models()

    print(f"\n✅ Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext steps:")
    print("1. Verify your trained model files are saved (vectorizer.pkl, model.pkl)")
    print("2. Start your FastAPI backend with the trained models")
    print("3. Try real-world test emails 🚀")

if __name__ == "__main__":
    main()
