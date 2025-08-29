#!/usr/bin/env python3
"""
AI Phishing & Malware Email Detector - Model Training Script
===========================================================
This script trains machine learning models to detect:
1. Phishing emails (based on subject, body, sender patterns)
2. Malicious file attachments (based on file characteristics)

The trained models are saved for use in the web application.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import re
import os
from datetime import datetime

class PhishingDetectorTrainer:
    def __init__(self):
        self.email_model = None
        self.file_model = None
        self.vectorizer = None
        
    def create_sample_email_data(self):

        # Legitimate emails (label=0)
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
          "Company newsletter: August edition now available",
          "Team lunch planned for Friday, please RSVP",
          "Library due date reminder: Books must be returned by 20th",
          "Conference call details for project kick-off attached",
          "Performance review feedback session scheduled for next week",
          "Holiday greetings from HR! Enjoy the festive season"
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
          "We couldn’t deliver your package. Please enter details to reschedule",
          "Apple ID locked due to unusual activity. Confirm to restore access",
          "Update your Gmail password immediately to avoid account deactivation",
          "Final notice: Outstanding bill must be paid today",
          "Your insurance policy has expired. Renew to continue coverage",
          "Urgent payroll issue: Verify account details to receive salary",
          "We detected unusual activity on your Netflix account. Reset password now",
          "Pending tax return refund. Submit form to receive payment",
          "Dropbox storage full. Upgrade now to avoid data loss",
          "Payment failed. Enter card details to continue subscription"
          "Limited-time offer! Get 90% discount on all products today only",
          "Act now! Only 2 seats left for this exclusive webinar",
          "Congratulations, you have been pre-approved for a loan",
          "Click here to unlock your free gift voucher",
          "Unsubscribe now to avoid being charged next month",
          "Increase your credit score instantly - apply here",
          "You have been selected for a free trial. Confirm now",
          "Your friend mentioned you in a photo. View it online",
          "Exclusive deal: luxury watches at unbelievable prices",
          "Hurry! Your cart will expire in 2 hours",
          "Earn $500 a day working from home, no experience needed",
          "Claim your free iPhone by clicking this link",
          "Lose 20 pounds in 2 weeks with this miracle pill",
          "Limited seats available for our crypto investment seminar",
          "Your email has won a lucky draw entry",
          "Watch unlimited movies online for free - register today",
          "Boost your productivity with this one simple trick",
          "Join our beta program and earn instant rewards",
          "Sign up today for guaranteed cashback offers",
          "Get exclusive travel deals, only valid for 24 hours"
          "Hi John, as discussed with the CFO, please transfer $50,000 to the vendor today. Use the attached account details.",
          "Internal HR Notice: Please complete the attached W-2 form to avoid payroll delays.",
          "Confidential: Legal case documents attached. Do not share with anyone outside the company.",
          "CEO Request: Urgent approval needed for international wire transfer.",
          "IT Helpdesk: We detected a password mismatch. Login with your credentials on the internal portal.",
          "New Vendor Contract Agreement attached-requires your digital signature.",
          "Board Meeting Slides: Encrypted file attached. Enter your email credentials to view.",
          "COVID-19 Health Policy Update: Please download and acknowledge the new guidelines.",
          "Your Office 365 license will expire today. Re-authenticate to keep access.",
          "Urgent security update from IT: Install the patch from the attachment immediately.",
          "CFO Notice: Payment approval form attached. Complete immediately.",
          "Project confidential details attached - review before tomorrow’s client call.",
          "HR annual review system update: Login with your employee ID to confirm participation.",
          "Partner NDA document attached for urgent signature.",
          "Vendor change request form attached - update records by EOD.",
          "Government audit inquiry - provide company data through the attached portal.",
          "Encrypted payroll database backup attached for review.",
          "Executive travel reimbursement form attached for urgent approval.",
          "Shareholder meeting invitation: Access slides using your company email credentials.",
          "IT support: VPN certificate renewal required. Download and install immediately."
    ]  
        
        # Create DataFrame
        emails = legitimate_emails + phishing_emails
        labels = [0] * len(legitimate_emails) + [1] * len(phishing_emails)
        
        # Add more sophisticated features
        email_data = []
        for email, label in zip(emails, labels):
            features = self.extract_email_features(email)
            features['text'] = email
            features['is_phishing'] = label
            email_data.append(features)
            
        return pd.DataFrame(email_data)
    
    def extract_email_features(self, email_text):
        """Extract features that indicate phishing attempts"""
        features = {}
        
        # Text-based features
        features['length'] = len(email_text)
        features['word_count'] = len(email_text.split())
        features['exclamation_count'] = email_text.count('!')
        features['question_count'] = email_text.count('?')
        features['uppercase_ratio'] = sum(1 for c in email_text if c.isupper()) / len(email_text) if email_text else 0
        
        # Suspicious keywords (common in phishing)
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
    
    def create_sample_file_data(self):
        # Safe files (label=0)
        safe_files = [
            {'filename': 'document.pdf', 'size': 524288, 'extension': '.pdf'},
            {'filename': 'image.jpg', 'size': 1048576, 'extension': '.jpg'},
            {'filename': 'report.docx', 'size': 2097152, 'extension': '.docx'},
            {'filename': 'data.xlsx', 'size': 1572864, 'extension': '.xlsx'},
            {'filename': 'presentation.pptx', 'size': 5242880, 'extension': '.pptx'},
            {'filename': 'notes.txt', 'size': 10240, 'extension': '.txt'},
            {'filename': 'archive.zip', 'size': 10485760, 'extension': '.zip'},
            {'filename': 'photo.png', 'size': 204800, 'extension': '.png'},
            {'filename': 'music.mp3', 'size': 3145728, 'extension': '.mp3'},
            {'filename': 'video.mp4', 'size': 52428800, 'extension': '.mp4'},
            {'filename': 'ebook.epub', 'size': 2048000, 'extension': '.epub'},
            {'filename': 'presentation.key', 'size': 4096000, 'extension': '.key'},
            {'filename': 'spreadsheet.ods', 'size': 1048576, 'extension': '.ods'},
            {'filename': 'drawing.svg', 'size': 51200, 'extension': '.svg'},
            {'filename': 'calendar.ics', 'size': 10240, 'extension': '.ics'},
        ]

        
        # Suspicious files (label=1) - based on characteristics, not actual malware
        suspicious_files = [
            {'filename': 'resume.pdf.exe', 'size': 4096, 'extension': '.exe'},
            {'filename': 'financials.xlsm', 'size': 1048576, 'extension': '.xlsm'},
            {'filename': 'macro_doc.docm', 'size': 512000, 'extension': '.docm'},
            {'filename': 'compressed.js', 'size': 2048, 'extension': '.js'},
            {'filename': 'hidden.scr', 'size': 1024, 'extension': '.scr'},
            {'filename': 'autorun.inf', 'size': 128, 'extension': '.inf'},
            {'filename': 'script.vbs', 'size': 2048, 'extension': '.vbs'},
            {'filename': 'archive.zip.exe', 'size': 512, 'extension': '.exe'},
            {'filename': 'setup.pkg', 'size': 2048000, 'extension': '.pkg'},
            {'filename': 'game_mod.jar', 'size': 102400, 'extension': '.jar'},
            {'filename': 'shortcut.lnk', 'size': 1024, 'extension': '.lnk'},
            {'filename': 'app_installer.dmg', 'size': 52428800, 'extension': '.dmg'},
            {'filename': 'powershell.ps1', 'size': 20480, 'extension': '.ps1'},
            {'filename': 'shell.sh', 'size': 10240, 'extension': '.sh'},
            {'filename': 'logfile.bat', 'size': 512, 'extension': '.bat'},
            {'filename': 'payload.exe', 'size': 1024, 'extension': '.exe'},
            {'filename': 'ransomware.ps1', 'size': 4096, 'extension': '.ps1'},
            {'filename': 'worm.bat', 'size': 256, 'extension': '.bat'},
            {'filename': 'rootkit.sys', 'size': 8192, 'extension': '.sys'},
            {'filename': 'backdoor.dll', 'size': 4096, 'extension': '.dll'},
            {'filename': 'keylogger.pif', 'size': 1024, 'extension': '.pif'},
            {'filename': 'trojan.jar', 'size': 2048, 'extension': '.jar'},
            {'filename': 'crypto_miner.exe', 'size': 20480, 'extension': '.exe'},
            {'filename': 'stealer.apk', 'size': 102400, 'extension': '.apk'},
            {'filename': 'exploit_tool.elf', 'size': 20480, 'extension': '.elf'},
            {'filename': 'virus.com', 'size': 1024, 'extension': '.com'},
            {'filename': 'driver_injector.sys', 'size': 16384, 'extension': '.sys'},
            {'filename': 'remote_access_tool.exe', 'size': 5120, 'extension': '.exe'},
            {'filename': 'spyware.ocx', 'size': 3072, 'extension': '.ocx'},
            {'filename': 'adware.cpl', 'size': 2048, 'extension': '.cpl'},
        ]
        
        file_data = []
        
        # Process safe files
        for file_info in safe_files:
            features = self.extract_file_features(file_info)
            features['is_malware'] = 0
            file_data.append(features)
            
        # Process suspicious files
        for file_info in suspicious_files:
            features = self.extract_file_features(file_info)
            features['is_malware'] = 1
            file_data.append(features)
            
        return pd.DataFrame(file_data)
    
    def extract_file_features(self, file_info):
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
        
        # Suspicious filename patterns
        features['has_double_extension'] = 1 if filename.count('.') > 1 else 0
        features['has_spaces'] = 1 if ' ' in filename else 0
        
        # Size-based features
        features['very_small_file'] = 1 if size < 1024 else 0  # < 1KB
        features['very_large_file'] = 1 if size > 50 * 1024 * 1024 else 0  # > 50MB
        
        return features
    
    def train_email_model(self, df):
        """Train the email phishing detection model"""
        print("Training email phishing detection model...")
        
        # Prepare features
        feature_columns = ['length', 'word_count', 'exclamation_count', 'question_count', 
                          'uppercase_ratio', 'suspicious_word_count', 'url_count', 'financial_terms_count']
        
        X_features = df[feature_columns]
        X_text = df['text']
        y = df['is_phishing']
        
        # Create pipeline with TF-IDF for text and additional features
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        
        # Combine text features with numerical features
        X_text_tfidf = self.vectorizer.fit_transform(X_text).toarray()
        X_combined = np.hstack([X_text_tfidf, X_features.values])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.email_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.email_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.email_model.predict(X_test)
        print(f"Email Model Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print("\nEmail Model Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
        
        return self.email_model
    
    def train_file_model(self, df):
        """Train the file malware detection model"""
        print("\nTraining file malware detection model...")
        
        # Prepare features
        feature_columns = ['file_size', 'filename_length', 'extension_length', 'is_executable',
                          'is_office_doc', 'is_image', 'has_double_extension', 'has_spaces',
                          'very_small_file', 'very_large_file']
        
        X = df[feature_columns]
        y = df['is_malware']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.file_model = LogisticRegression(random_state=42)
        self.file_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.file_model.predict(X_test)
        print(f"File Model Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print("\nFile Model Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Safe', 'Malware']))
        
        return self.file_model
    
    def save_models(self):
        """Save trained models and vectorizer"""
        os.makedirs('models', exist_ok=True)
        
        # Save email model and vectorizer
        joblib.dump(self.email_model, 'models/email_phishing_model.pkl')
        joblib.dump(self.vectorizer, 'models/email_vectorizer.pkl')
        
        # Save file model
        joblib.dump(self.file_model, 'models/file_malware_model.pkl')
        
        print("\nModels saved successfully:")
        print("- models/email_phishing_model.pkl")
        print("- models/email_vectorizer.pkl")
        print("- models/file_malware_model.pkl")

def main():
    """Main training function"""
    print("🔒 AI Phishing & Malware Detector - Model Training")
    print("=" * 50)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    trainer = PhishingDetectorTrainer()
    
    # Create and train email model
    print("\n📧 Creating email dataset...")
    email_df = trainer.create_sample_email_data()
    print(f"Email dataset created with {len(email_df)} samples")
    
    trainer.train_email_model(email_df)
    
    # Create and train file model
    print("\n📁 Creating file dataset...")
    file_df = trainer.create_sample_file_data()
    print(f"File dataset created with {len(file_df)} samples")
    
    trainer.train_file_model(file_df)
    
    # Save models
    trainer.save_models()
    
    print(f"\n✅ Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext steps:")
    print("1. Run this script: python ml/train_model.py")
    print("2. Start the FastAPI backend: python backend/app.py")
    print("3. Open the web interface and start scanning!")

if __name__ == "__main__":
    main()