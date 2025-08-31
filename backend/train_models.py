# backend/train_models.py
"""
Training script that can be imported by the backend
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import re
import os

def extract_email_features(email_text):
    """Extract features from email text"""
    features = {}
    
    # Text-based features
    features['length'] = len(email_text)
    features['word_count'] = len(email_text.split())
    features['exclamation_count'] = email_text.count('!')
    features['question_count'] = email_text.count('?')
    features['uppercase_ratio'] = sum(1 for c in email_text if c.isupper()) / len(email_text) if email_text else 0
    
    # Suspicious keywords
    suspicious_words = ['urgent', 'immediate', 'verify', 'suspended', 'click', 'winner', 'prize', 
                      'congratulations', 'limited time', 'act now', 'confirm', 'security alert']
    features['suspicious_word_count'] = sum(1 for word in suspicious_words if word.lower() in email_text.lower())
    
    # URL patterns
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    features['url_count'] = len(re.findall(url_pattern, email_text))
    
    # Financial terms
    financial_terms = ['money', 'payment', 'credit card', 'bank', 'account', 'refund', 'charge']
    features['financial_terms_count'] = sum(1 for term in financial_terms if term.lower() in email_text.lower())
    
    return features

def extract_file_features(file_info):
    """Extract features from file information"""
    features = {}
    
    filename = file_info['filename']
    size = file_info['size']
    extension = file_info['extension'].lower()
    
    # Basic features
    features['file_size'] = size
    features['filename_length'] = len(filename)
    features['extension_length'] = len(extension)
    
    # Extension-based features
    executable_extensions = ['.exe', '.scr', '.bat', '.com', '.pif', '.vbs', '.js']
    features['is_executable'] = 1 if extension in executable_extensions else 0
    
    office_extensions = ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']
    features['is_office_doc'] = 1 if extension in office_extensions else 0
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    features['is_image'] = 1 if extension in image_extensions else 0
    
    # Suspicious patterns
    features['has_double_extension'] = 1 if filename.count('.') > 1 else 0
    features['has_spaces'] = 1 if ' ' in filename else 0
    features['very_small_file'] = 1 if size < 1024 else 0
    features['very_large_file'] = 1 if size > 50 * 1024 * 1024 else 0
    
    return features

def create_sample_email_data():
    """Create email training data"""
    # Legitimate emails
    legitimate_emails = [
        "Meeting scheduled for tomorrow at 10 AM in conference room B",
        "Your order has been shipped and will arrive in 2-3 business days",
        "Weekly team sync - please review the attached agenda",
        "Reminder: Project deadline is next Friday",
        "Thank you for your payment. Invoice #12345 is now paid",
        "Welcome to our service! Here's how to get started",
        "Your subscription will renew automatically next month",
        "Please review and approve the quarterly budget proposal",
        "System maintenance scheduled for this weekend",
        "New feature update available - check out what's new",
        "Flight booking confirmed: NYC to LA, 15th September",
        "Password changed successfully for your company portal",
        "Doctor appointment confirmed for Monday at 3 PM",
        "Your movie tickets are attached. Enjoy your show!",
        "Company newsletter: August edition now available"
    ]
    
    # Phishing emails
    phishing_emails = [
        "URGENT: Your account will be suspended! Click here immediately to verify",
        "Congratulations! You've won $10,000! Claim your prize now",
        "Security alert: Suspicious login detected. Verify your identity now",
        "Your PayPal account has been limited. Click to restore access",
        "IRS Notice: You have a pending refund. Download form immediately",
        "Your credit card has been charged $500. Dispute this transaction now",
        "Amazon: Your order could not be delivered. Update payment method",
        "Your bank account will be closed. Verify your information to prevent this",
        "Lottery winner! You've been selected for a million dollar prize",
        "Microsoft security team: Your computer is infected. Download fix now",
        "We couldn't deliver your package. Please enter details to reschedule",
        "Apple ID locked due to unusual activity. Confirm to restore access",
        "Update your Gmail password immediately to avoid account deactivation",
        "Final notice: Outstanding bill must be paid today",
        "Your insurance policy has expired. Renew to continue coverage"
    ]
    
    # Create DataFrame
    emails = legitimate_emails + phishing_emails
    labels = [0] * len(legitimate_emails) + [1] * len(phishing_emails)
    
    email_data = []
    for email, label in zip(emails, labels):
        features = extract_email_features(email)
        features['text'] = email
        features['is_phishing'] = label
        email_data.append(features)
        
    return pd.DataFrame(email_data)

def create_sample_file_data():
    """Create file training data"""
    # Safe files
    safe_files = [
        {'filename': 'document.pdf', 'size': 524288, 'extension': '.pdf'},
        {'filename': 'image.jpg', 'size': 1048576, 'extension': '.jpg'},
        {'filename': 'report.docx', 'size': 2097152, 'extension': '.docx'},
        {'filename': 'data.xlsx', 'size': 1572864, 'extension': '.xlsx'},
        {'filename': 'presentation.pptx', 'size': 5242880, 'extension': '.pptx'},
        {'filename': 'notes.txt', 'size': 10240, 'extension': '.txt'},
        {'filename': 'photo.png', 'size': 204800, 'extension': '.png'},
        {'filename': 'music.mp3', 'size': 3145728, 'extension': '.mp3'},
    ]
    
    # Suspicious files
    suspicious_files = [
        {'filename': 'resume.pdf.exe', 'size': 4096, 'extension': '.exe'},
        {'filename': 'financials.xlsm', 'size': 1048576, 'extension': '.xlsm'},
        {'filename': 'macro_doc.docm', 'size': 512000, 'extension': '.docm'},
        {'filename': 'compressed.js', 'size': 2048, 'extension': '.js'},
        {'filename': 'hidden.scr', 'size': 1024, 'extension': '.scr'},
        {'filename': 'script.vbs', 'size': 2048, 'extension': '.vbs'},
        {'filename': 'archive.zip.exe', 'size': 512, 'extension': '.exe'},
        {'filename': 'powershell.ps1', 'size': 20480, 'extension': '.ps1'},
    ]
    
    file_data = []
    
    # Process safe files
    for file_info in safe_files:
        features = extract_file_features(file_info)
        features['is_malware'] = 0
        file_data.append(features)
        
    # Process suspicious files
    for file_info in suspicious_files:
        features = extract_file_features(file_info)
        features['is_malware'] = 1
        file_data.append(features)
        
    return pd.DataFrame(file_data)

def train_models():
    """Train and save both models"""
    try:
        print("🔄 Starting model training...")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Train email model
        email_df = create_sample_email_data()
        
        feature_columns = ['length', 'word_count', 'exclamation_count', 'question_count', 
                          'uppercase_ratio', 'suspicious_word_count', 'url_count', 'financial_terms_count']
        
        X_features = email_df[feature_columns]
        X_text = email_df['text']
        y = email_df['is_phishing']
        
        # Create and fit vectorizer
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        X_text_tfidf = vectorizer.fit_transform(X_text).toarray()
        X_combined = np.hstack([X_text_tfidf, X_features.values])
        
        # Train email model
        email_model = RandomForestClassifier(n_estimators=100, random_state=42)
        email_model.fit(X_combined, y)
        
        # Train file model
        file_df = create_sample_file_data()
        
        file_feature_columns = ['file_size', 'filename_length', 'extension_length', 'is_executable',
                               'is_office_doc', 'is_image', 'has_double_extension', 'has_spaces',
                               'very_small_file', 'very_large_file']
        
        X_file = file_df[file_feature_columns]
        y_file = file_df['is_malware']
        
        file_model = LogisticRegression(random_state=42)
        file_model.fit(X_file, y_file)
        
        # Save models
        joblib.dump(email_model, 'models/email_phishing_model.pkl')
        joblib.dump(vectorizer, 'models/email_vectorizer.pkl')
        joblib.dump(file_model, 'models/file_malware_model.pkl')
        
        print("✅ Models trained and saved successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error training models: {e}")
        return False