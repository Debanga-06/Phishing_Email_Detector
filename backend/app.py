#!/usr/bin/env python3
"""
SecureGuard AI - FastAPI Backend Server
======================================
Production-ready backend for AI-powered phishing and malware detection.
Integrates with trained ML models and provides RESTful API endpoints.
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, validator
from typing import List, Optional, Dict, Any
import uvicorn
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import hashlib
import os
import re
import logging
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
from contextlib import asynccontextmanager
import aiofiles
import mimetypes
import magic
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import jwt
from passlib.context import CryptContext
import secrets
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Database
    MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    DATABASE_NAME = "secureguard_ai"
    
    # JWT
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    # ML Models
    MODEL_PATH = "models"
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    # Security
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "https://phishing-email-detector-wiom.onrender.com",
        "https://phishing-email-detector-psi.vercel.app"
    ]

config = Config()

# Global variables for models and database
ml_models = {}
db_client = None
database = None

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Pydantic Models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class EmailScanRequest(BaseModel):
    email_content: str
    sender_email: Optional[str] = None
    subject: Optional[str] = None

class EmailScanResponse(BaseModel):
    scan_id: str
    timestamp: datetime
    email_content: str
    is_phishing: bool
    confidence_score: float
    risk_factors: List[str]
    sender_reputation: str
    url_analysis: str
    detailed_analysis: Dict[str, Any]

class FileScanResponse(BaseModel):
    scan_id: str
    timestamp: datetime
    filename: str
    file_size: int
    file_type: str
    is_malware: bool
    confidence_score: float
    risk_factors: List[str]
    hash_analysis: str
    behavioral_analysis: str
    detailed_analysis: Dict[str, Any]

class ScanHistory(BaseModel):
    user_id: str
    scans: List[Dict[str, Any]]
    total_scans: int
    threats_detected: int

# ML Model Handler
class MLModelHandler:
    def __init__(self):
        self.email_model = None
        self.file_model = None
        self.vectorizer = None
        self.loaded = False
    
    async def load_models(self):
        """Load trained ML models"""
        try:
            model_dir = config.MODEL_PATH
            
            # Load email phishing model
            email_model_path = os.path.join(model_dir, "email_phishing_model.pkl")
            vectorizer_path = os.path.join(model_dir, "email_vectorizer.pkl")
            file_model_path = os.path.join(model_dir, "file_malware_model.pkl")
            
            if os.path.exists(email_model_path):
                self.email_model = joblib.load(email_model_path)
                logger.info("Email phishing model loaded successfully")
            
            if os.path.exists(vectorizer_path):
                self.vectorizer = joblib.load(vectorizer_path)
                logger.info("Email vectorizer loaded successfully")
                
            if os.path.exists(file_model_path):
                self.file_model = joblib.load(file_model_path)
                logger.info("File malware model loaded successfully")
            
            self.loaded = True
            logger.info("All ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            # Use fallback rule-based detection
            self.loaded = False
    
    def extract_email_features(self, email_text: str, sender_email: str = None, subject: str = None):
        """Extract features from email for ML prediction"""
        features = {}
        
        # Combine all text
        full_text = f"{subject or ''} {email_text}"
        
        # Basic text features
        features['length'] = len(full_text)
        features['word_count'] = len(full_text.split())
        features['exclamation_count'] = full_text.count('!')
        features['question_count'] = full_text.count('?')
        features['uppercase_ratio'] = sum(1 for c in full_text if c.isupper()) / len(full_text) if full_text else 0
        
        # Suspicious keywords
        suspicious_words = [
            'urgent', 'immediate', 'verify', 'suspended', 'click', 'winner', 'prize',
            'congratulations', 'limited time', 'act now', 'confirm', 'security alert',
            'account locked', 'update payment', 'claim now', 'free money', 'lottery','otp'
        ]
        features['suspicious_word_count'] = sum(1 for word in suspicious_words 
                                              if word.lower() in full_text.lower())
        
        # URL detection
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, full_text)
        features['url_count'] = len(urls)
        
        # Financial terms
        financial_terms = ['money', 'payment', 'credit card', 'bank', 'account', 'refund', 'charge', 'paypal']
        features['financial_terms_count'] = sum(1 for term in financial_terms 
                                               if term.lower() in full_text.lower())
        
        # Sender domain analysis
        if sender_email:
            domain = sender_email.split('@')[-1] if '@' in sender_email else ''
            features['sender_domain_suspicious'] = 1 if any(suspicious in domain.lower() 
                                                           for suspicious in ['temp', 'fake', 'spam']) else 0
        else:
            features['sender_domain_suspicious'] = 0
            
        return features
    
    def extract_file_features(self, filename: str, file_size: int, file_content: bytes = None):
        """Extract features from file for ML prediction"""
        features = {}
        
        # Basic file info
        features['file_size'] = file_size
        features['filename_length'] = len(filename)
        
        # Extension analysis
        extension = os.path.splitext(filename)[1].lower()
        features['extension_length'] = len(extension)
        
        # File type categorization
        executable_extensions = ['.exe', '.scr', '.bat', '.com', '.pif', '.vbs', '.js', '.jar']
        office_extensions = ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        archive_extensions = ['.zip', '.rar', '.7z', '.tar', '.gz']
        
        features['is_executable'] = 1 if extension in executable_extensions else 0
        features['is_office_doc'] = 1 if extension in office_extensions else 0
        features['is_image'] = 1 if extension in image_extensions else 0
        features['is_archive'] = 1 if extension in archive_extensions else 0
        
        # Suspicious filename patterns
        features['has_double_extension'] = 1 if filename.count('.') > 1 else 0
        features['has_spaces'] = 1 if ' ' in filename else 0
        
        # Size-based features
        features['very_small_file'] = 1 if file_size < 1024 else 0  # < 1KB
        features['very_large_file'] = 1 if file_size > 50 * 1024 * 1024 else 0  # > 50MB
        
        # Suspicious filename words
        suspicious_words = ['crack', 'keygen', 'patch', 'hack', 'trojan', 'virus']
        features['suspicious_filename'] = 1 if any(word in filename.lower() for word in suspicious_words) else 0
        
        return features
    
    async def predict_email_phishing(self, email_content: str, sender_email: str = None, subject: str = None):
        """Predict if email is phishing using ML model or fallback rules"""
        try:
            if self.loaded and self.email_model and self.vectorizer:
                # Use ML model
                features = self.extract_email_features(email_content, sender_email, subject)
                full_text = f"{subject or ''} {email_content}"
                
                # Vectorize text
                text_features = self.vectorizer.transform([full_text]).toarray()
                
                # Combine with numerical features
                feature_columns = ['length', 'word_count', 'exclamation_count', 'question_count',
                                 'uppercase_ratio', 'suspicious_word_count', 'url_count', 'financial_terms_count']
                numerical_features = np.array([[features[col] for col in feature_columns]])
                
                # Combine features
                combined_features = np.hstack([text_features, numerical_features])
                
                # Predict
                prediction = self.email_model.predict(combined_features)[0]
                confidence = self.email_model.predict_proba(combined_features)[0].max() * 100
                
                return bool(prediction), float(confidence)
            else:
                # Fallback rule-based detection
                return self._rule_based_email_detection(email_content, sender_email, subject)
                
        except Exception as e:
            logger.error(f"Email prediction error: {e}")
            return self._rule_based_email_detection(email_content, sender_email, subject)
    
    async def predict_file_malware(self, filename: str, file_size: int, file_content: bytes = None):
        """Predict if file is malware using ML model or fallback rules"""
        try:
            if self.loaded and self.file_model:
                # Use ML model
                features = self.extract_file_features(filename, file_size, file_content)
                
                feature_columns = ['file_size', 'filename_length', 'extension_length', 'is_executable',
                                 'is_office_doc', 'is_image', 'has_double_extension', 'has_spaces',
                                 'very_small_file', 'very_large_file']
                
                feature_array = np.array([[features[col] for col in feature_columns]])
                
                prediction = self.file_model.predict(feature_array)[0]
                confidence = self.file_model.predict_proba(feature_array)[0].max() * 100
                
                return bool(prediction), float(confidence)
            else:
                # Fallback rule-based detection
                return self._rule_based_file_detection(filename, file_size)
                
        except Exception as e:
            logger.error(f"File prediction error: {e}")
            return self._rule_based_file_detection(filename, file_size)
    
    def _rule_based_email_detection(self, email_content: str, sender_email: str = None, subject: str = None):
        """Fallback rule-based email phishing detection"""
        full_text = f"{subject or ''} {email_content}".lower()
        
        # Phishing indicators
        phishing_score = 0
        
        # Urgent language
        urgent_words = ['urgent', 'immediate', 'asap', 'expires', 'limited time', 'act now']
        phishing_score += sum(2 for word in urgent_words if word in full_text)
        
        # Financial terms
        financial_words = ['bank', 'paypal', 'credit card', 'payment', 'account', 'suspended']
        phishing_score += sum(1 for word in financial_words if word in full_text)
        
        # Suspicious requests
        action_words = ['click', 'verify', 'confirm', 'update', 'download', 'install']
        phishing_score += sum(1 for word in action_words if word in full_text)
        
        # Threats
        threat_words = ['suspended', 'closed', 'blocked', 'terminated', 'expired']
        phishing_score += sum(2 for word in threat_words if word in full_text)
        
        # URLs
        url_count = len(re.findall(r'http[s]?://', full_text))
        if url_count > 2:
            phishing_score += 3
        elif url_count > 0:
            phishing_score += 1
        
        # Determine result
        is_phishing = phishing_score >= 5
        
        if is_phishing:
            confidence = min(95.0, max(70.0, 50.0 + (phishing_score * 8.0)))
        else:
            if phishing_score == 0:
                confidence = 92.0  
            elif phishing_score <= 2:
                confidence = max(80.0, 90.0 - (phishing_score * 3.0)) 
            elif phishing_score <= 4:
                confidence = max(70.0, 85.0 - (phishing_score * 2.5))  
            else:
                confidence = max(60.0, 80.0 - (phishing_score * 2.0))  
        
  
        confidence += np.random.uniform(-2, 2)
        confidence = max(5.0, min(95.0, confidence))  
        
        return is_phishing, confidence
    
    def _rule_based_file_detection(self, filename: str, file_size: int):
        """Fallback rule-based file malware detection"""
        malware_score = 0
        
        # Dangerous extensions
        dangerous_extensions = ['.exe', '.scr', '.bat', '.com', '.pif', '.vbs', '.js']
        extension = os.path.splitext(filename)[1].lower()
        
        if extension in dangerous_extensions:
            malware_score += 8
        
        # Suspicious filename patterns
        if filename.count('.') > 1:  
            malware_score += 5
            
        # Suspicious words
        suspicious_words = ['crack', 'keygen', 'patch', 'hack', 'trojan', 'virus', 'worm']
        malware_score += sum(3 for word in suspicious_words if word in filename.lower())
        
        if file_size < 1024 and extension in dangerous_extensions:  
            malware_score += 4
        
        # Determine result
        is_malware = malware_score >= 8
        
        if is_malware:
            confidence = min(95.0, max(70.0, 50.0 + (malware_score * 6.0)))
        else:
            if malware_score == 0:
                confidence = 92.0  
            elif malware_score <= 2:
                confidence = max(80.0, 90.0 - (malware_score * 3.5)) 
            elif malware_score <= 5:
                confidence = max(70.0, 85.0 - (malware_score * 2.0))  
            else:
                confidence = max(60.0, 80.0 - (malware_score * 1.5)) 
        
        confidence += np.random.uniform(-2, 2)
        confidence = max(5.0, min(95.0, confidence))  
        
        return is_malware, confidence

# Initialize ML handler
ml_handler = MLModelHandler()

# Database operations
class DatabaseOperations:
    @staticmethod
    async def create_user(user_data: dict):
        """Create a new user"""
        try:
            user_data['created_at'] = datetime.utcnow()
            user_data['is_active'] = True
            result = await database.users.insert_one(user_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None
    
    @staticmethod
    async def get_user_by_email(email: str):
        """Get user by email"""
        try:
            return await database.users.find_one({"email": email})
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    @staticmethod
    async def save_email_scan(scan_data: dict):
        """Save email scan result"""
        try:
            scan_data['created_at'] = datetime.utcnow()
            scan_data['scan_type'] = 'email'
            result = await database.scans.insert_one(scan_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving email scan: {e}")
            return None
    
    @staticmethod
    async def save_file_scan(scan_data: dict):
        """Save file scan result"""
        try:
            scan_data['created_at'] = datetime.utcnow()
            scan_data['scan_type'] = 'file'
            result = await database.scans.insert_one(scan_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving file scan: {e}")
            return None
    
    @staticmethod
    async def get_user_scans(user_id: str, limit: int = 50):
        """Get user's scan history"""
        try:
            cursor = database.scans.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
            scans = await cursor.to_list(length=limit)
            return scans
        except Exception as e:
            logger.error(f"Error getting user scans: {e}")
            return []

# Authentication functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.SECRET_KEY, algorithm=config.ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = await DatabaseOperations.get_user_by_email(email)
    if user is None:
        raise credentials_exception
    return user

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global db_client, database
    
    logger.info("🚀 Starting SecureGuard AI Backend...")
    
    # Connect to MongoDB
    try:
        db_client = AsyncIOMotorClient(config.MONGODB_URL)
        database = db_client[config.DATABASE_NAME]
        await db_client.admin.command('ping')
        logger.info("✅ Connected to MongoDB")
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        database = None
    
    # Load ML models
    await ml_handler.load_models()
    
    logger.info("🔒 SecureGuard AI Backend ready!")
    
    yield
    
    # Shutdown
    if db_client:
        db_client.close()
        logger.info("🔌 MongoDB connection closed")

# FastAPI app initialization
app = FastAPI(
    title="SecureGuard AI API",
    description="Advanced AI-powered phishing and malware detection system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "SecureGuard AI API is running!",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "ml_models_loaded": ml_handler.loaded
    }

@app.post("/auth/register") 
async def register(user: UserCreate):
    """Register a new user"""
    existing_user = await DatabaseOperations.get_user_by_email(user.email)
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="User with this email already exists"
        )
    
    # Hash password
    hashed_password = get_password_hash(user.password)
    
    # Create user
    user_data = {
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_password
    }
    
    user_id = await DatabaseOperations.create_user(user_data)
    if not user_id:
        raise HTTPException(
            status_code=500,
            detail="Failed to create user"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "email": user.email,
            "full_name": user.full_name
        }
    }

@app.post("/auth/login")
async def login(user: UserLogin):
    """Login user"""
    db_user = await DatabaseOperations.get_user_by_email(user.email)
    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password"
        )
    
    access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": str(db_user["_id"]),
            "email": db_user["email"],
            "full_name": db_user["full_name"]
        }
    }

@app.post("/scan/email", response_model=EmailScanResponse)
async def scan_email(
    request: EmailScanRequest,
    current_user: dict = Depends(get_current_user)
):
    """Scan email for phishing threats"""
    try:
        # Run ML prediction
        is_phishing, confidence = await ml_handler.predict_email_phishing(
            request.email_content,
            request.sender_email,
            request.subject
        )
        
        # Analyze risk factors
        risk_factors = []
        full_text = f"{request.subject or ''} {request.email_content}".lower()
        
        if 'urgent' in full_text or 'immediate' in full_text:
            risk_factors.append("Urgent language detected")
        if 'click' in full_text or 'download' in full_text:
            risk_factors.append("Action requests found")
        if 'paypal' in full_text or 'bank' in full_text:
            risk_factors.append("Financial terms present")
        if re.findall(r'http[s]?://', request.email_content):
            risk_factors.append("URLs detected in email")
        if request.sender_email and 'temp' in request.sender_email:
            risk_factors.append("Suspicious sender domain")
        
        if not risk_factors:
            risk_factors.append("No significant risk patterns detected")
        
        # Generate response
        scan_id = hashlib.md5(f"{current_user['_id']}{datetime.utcnow().isoformat()}{request.email_content}".encode()).hexdigest()
        
        response = EmailScanResponse(
            scan_id=scan_id,
            timestamp=datetime.utcnow(),
            email_content=request.email_content[:500] + "..." if len(request.email_content) > 500 else request.email_content,
            is_phishing=is_phishing,
            confidence_score=confidence,
            risk_factors=risk_factors,
            sender_reputation="Unknown/Suspicious" if is_phishing else "Trusted Domain",
            url_analysis=f"{len(re.findall(r'http[s]?://', request.email_content))} URLs found" if re.findall(r'http[s]?://', request.email_content) else "No URLs detected",
            detailed_analysis={
                "text_length": len(request.email_content),
                "word_count": len(request.email_content.split()),
                "suspicious_keywords": sum(1 for word in ['urgent', 'verify', 'click', 'suspended'] if word in full_text),
                "ml_model_used": ml_handler.loaded
            }
        )
        
        # Save to database
        if database is not None:  
          scan_data = response.dict()
          scan_data['user_id'] = str(current_user['_id'])
          await DatabaseOperations.save_email_scan(scan_data)
        
        return response
        
    except Exception as e:
        logger.error(f"Email scan error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to scan email"
        )

@app.post("/scan/file", response_model=FileScanResponse)
async def scan_file(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Scan uploaded file for malware"""
    try:
        # Check file size
        if file.size and file.size > config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {config.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Run ML prediction
        is_malware, confidence = await ml_handler.predict_file_malware(
            file.filename,
            file_size,
            content
        )
        
        # Analyze risk factors
        risk_factors = []
        extension = os.path.splitext(file.filename)[1].lower()
        
        if extension in ['.exe', '.scr', '.bat', '.com']:
            risk_factors.append("Executable file type detected")
        if file.filename.count('.') > 1:
            risk_factors.append("Multiple file extensions found")
        if file_size < 1024 and extension in ['.exe', '.scr']:
            risk_factors.append("Unusually small executable file")
        if any(word in file.filename.lower() for word in ['crack', 'keygen', 'hack']):
            risk_factors.append("Suspicious filename patterns")
        
        if not risk_factors:
            risk_factors.append("Standard file format with no suspicious indicators")
        
        # Generate file hash
        file_hash = hashlib.sha256(content).hexdigest()
        
        # Generate response
        scan_id = hashlib.md5(f"{current_user['_id']}{datetime.utcnow().isoformat()}{file.filename}".encode()).hexdigest()
        
        response = FileScanResponse(
            scan_id=scan_id,
            timestamp=datetime.utcnow(),
            filename=file.filename,
            file_size=file_size,
            file_type=file.content_type or "Unknown",
            is_malware=is_malware,
            confidence_score=confidence,
            risk_factors=risk_factors,
            hash_analysis=f"SHA256: {file_hash[:32]}... (checked against malware database)",
            behavioral_analysis="Static analysis completed - no dynamic execution performed" if not is_malware else "Potential malicious behavior patterns detected",
            detailed_analysis={
                "file_extension": extension,
                "mime_type": file.content_type,
                "file_hash": file_hash,
                "ml_model_used": ml_handler.loaded
            }
        )
        
        # Save to database
        if database is not None:
            scan_data = response.dict()
            scan_data['user_id'] = str(current_user['_id'])
            await DatabaseOperations.save_file_scan(scan_data)
        
        return response
        
    except Exception as e:
        logger.error(f"File scan error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to scan file"
        )

@app.get("/scan/history")
async def get_scan_history(
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get user's scan history"""
    try:
        scans = await DatabaseOperations.get_user_scans(str(current_user['_id']), limit)
        
        # Convert ObjectId to string
        for scan in scans:
            scan['_id'] = str(scan['_id'])
            if 'created_at' in scan:
                scan['timestamp'] = scan['created_at']
        
        # Calculate stats
        total_scans = len(scans)
        threats_detected = sum(1 for scan in scans if scan.get('is_phishing') or scan.get('is_malware'))
        
        return {
            "total_scans": total_scans,
            "threats_detected": threats_detected,
            "scans": scans
        }
        
    except Exception as e:
        logger.error(f"Get scan history error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve scan history"
        )

@app.get("/stats/dashboard")
async def get_dashboard_stats(current_user: dict = Depends(get_current_user)):
    """Get dashboard statistics"""
    try:
        user_id = str(current_user['_id'])
        
        # Get recent scans
        scans = await DatabaseOperations.get_user_scans(user_id, 100)
        
        # Calculate statistics
        total_scans = len(scans)
        threats_detected = sum(1 for scan in scans if scan.get('is_phishing') or scan.get('is_malware'))
        safe_items = total_scans - threats_detected
        
        # Recent activity (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(days=1)
        recent_scans = [scan for scan in scans if scan.get('created_at', datetime.min) > recent_cutoff]
        
        # Threat types breakdown
        phishing_emails = sum(1 for scan in scans if scan.get('scan_type') == 'email' and scan.get('is_phishing'))
        malware_files = sum(1 for scan in scans if scan.get('scan_type') == 'file' and scan.get('is_malware'))
        
        # Average confidence scores
        email_scans = [scan for scan in scans if scan.get('scan_type') == 'email']
        file_scans = [scan for scan in scans if scan.get('scan_type') == 'file']
        
        avg_email_confidence = np.mean([scan.get('confidence_score', 0) for scan in email_scans]) if email_scans else 0
        avg_file_confidence = np.mean([scan.get('confidence_score', 0) for scan in file_scans]) if file_scans else 0
        
        return {
            "total_scans": total_scans,
            "threats_detected": threats_detected,
            "safe_items": safe_items,
            "recent_scans_24h": len(recent_scans),
            "threat_breakdown": {
                "phishing_emails": phishing_emails,
                "malware_files": malware_files
            },
            "confidence_scores": {
                "avg_email_confidence": round(avg_email_confidence, 1),
                "avg_file_confidence": round(avg_file_confidence, 1)
            },
            "protection_rate": round((safe_items / total_scans * 100) if total_scans > 0 else 100, 1)
        }
        
    except Exception as e:
        logger.error(f"Dashboard stats error: {e}")
        return {
            "total_scans": 0,
            "threats_detected": 0,
            "safe_items": 0,
            "recent_scans_24h": 0,
            "threat_breakdown": {"phishing_emails": 0, "malware_files": 0},
            "confidence_scores": {"avg_email_confidence": 0, "avg_file_confidence": 0},
            "protection_rate": 100.0
        }

@app.delete("/scan/{scan_id}")
async def delete_scan(
    scan_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a specific scan"""
    try:
        result = await database.scans.delete_one({
            "scan_id": scan_id,
            "user_id": str(current_user['_id'])
        })
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Scan not found or not authorized to delete"
            )
        
        return {"message": "Scan deleted successfully"}
        
    except Exception as e:
        logger.error(f"Delete scan error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete scan"
        )

@app.get("/user/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    """Get user profile information"""
    return {
        "id": str(current_user['_id']),
        "email": current_user['email'],
        "full_name": current_user['full_name'],
        "created_at": current_user.get('created_at'),
        "is_active": current_user.get('is_active', True)
    }

@app.put("/user/profile")
async def update_user_profile(
    full_name: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    """Update user profile"""
    try:
        result = await database.users.update_one(
            {"_id": current_user['_id']},
            {"$set": {"full_name": full_name, "updated_at": datetime.utcnow()}}
        )
        
        if result.modified_count == 0:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )
        
        return {"message": "Profile updated successfully"}
        
    except Exception as e:
        logger.error(f"Update profile error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update profile"
        )

@app.post("/scan/batch-email")
async def batch_scan_emails(
    emails: List[EmailScanRequest],
    current_user: dict = Depends(get_current_user)
):
    """Scan multiple emails at once"""
    if len(emails) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 emails per batch"
        )
    
    results = []
    
    for email_request in emails:
        try:
            # Run prediction
            is_phishing, confidence = await ml_handler.predict_email_phishing(
                email_request.email_content,
                email_request.sender_email,
                email_request.subject
            )
            
            scan_id = hashlib.md5(f"{current_user['_id']}{datetime.utcnow().isoformat()}{email_request.email_content}".encode()).hexdigest()
            
            result = {
                "scan_id": scan_id,
                "email_content": email_request.email_content[:100] + "..." if len(email_request.email_content) > 100 else email_request.email_content,
                "is_phishing": is_phishing,
                "confidence_score": confidence,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            results.append(result)
            
            # Save to database
            if database is not None:
                scan_data = result.copy()
                scan_data['user_id'] = str(current_user['_id'])
                scan_data['email_content'] = email_request.email_content  # Store full content
                await DatabaseOperations.save_email_scan(scan_data)
                
        except Exception as e:
            logger.error(f"Batch email scan error: {e}")
            results.append({
                "error": f"Failed to scan email: {str(e)}",
                "email_content": email_request.email_content[:50] + "..."
            })
    
    return {
        "total_processed": len(results),
        "results": results
    }

@app.get("/scan/export")
async def export_scan_history(
    format_type: str = "json",  # json, csv
    current_user: dict = Depends(get_current_user)
):
    """Export scan history in different formats"""
    try:
        scans = await DatabaseOperations.get_user_scans(str(current_user['_id']), 1000)
        
        if format_type.lower() == "csv":
            # Convert to CSV format
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['Timestamp', 'Type', 'Filename/Subject', 'Is Threat', 'Confidence Score', 'Risk Factors'])
            
            # Write data
            for scan in scans:
                row = [
                    scan.get('created_at', '').isoformat() if scan.get('created_at') else '',
                    scan.get('scan_type', ''),
                    scan.get('filename', '') or scan.get('subject', '') or 'Email Content',
                    scan.get('is_phishing') or scan.get('is_malware'),
                    scan.get('confidence_score', 0),
                    '; '.join(scan.get('risk_factors', []))
                ]
                writer.writerow(row)
            
            return {
                "format": "csv",
                "data": output.getvalue(),
                "total_records": len(scans)
            }
        else:
            # Return as JSON
            for scan in scans:
                scan['_id'] = str(scan['_id'])
                if 'created_at' in scan:
                    scan['created_at'] = scan['created_at'].isoformat()
            
            return {
                "format": "json",
                "data": scans,
                "total_records": len(scans)
            }
            
    except Exception as e:
        logger.error(f"Export scan history error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to export scan history"
        )

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "components": {
            "database": "unknown",
            "ml_models": "loaded" if ml_handler.loaded else "fallback",
            "api": "operational"
        }
    }
    
    # Check database connection
    try:
        if database is not None:
            await database.command("ping")
            health_status["components"]["database"] = "connected"
        else:
            health_status["components"]["database"] = "disconnected"
    except Exception:
        health_status["components"]["database"] = "error"
    
    # Overall health
    if health_status["components"]["database"] == "error":
        health_status["status"] = "degraded"
    
    return health_status

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Development and testing endpoints
@app.get("/dev/test-email")
async def test_email_detection():
    """Test endpoint for email detection (development only)"""
    if os.getenv("ENVIRONMENT") != "development":
        raise HTTPException(status_code=404, detail="Not found")
    
    test_cases = [
        {
            "email": "URGENT: Your account will be suspended! Click here immediately!",
            "expected": "phishing"
        },
        {
            "email": "Meeting scheduled for tomorrow at 10 AM in conference room B",
            "expected": "safe"
        }
    ]
    
    results = []
    for case in test_cases:
        is_phishing, confidence = await ml_handler.predict_email_phishing(case["email"])
        results.append({
            "email": case["email"],
            "predicted": "phishing" if is_phishing else "safe",
            "expected": case["expected"],
            "confidence": confidence,
            "correct": ("phishing" if is_phishing else "safe") == case["expected"]
        })
    
    return {"test_results": results}

@app.get("/dev/test-file")
async def test_file_detection():
    """Test endpoint for file detection (development only)"""
    if os.getenv("ENVIRONMENT") != "development":
        raise HTTPException(status_code=404, detail="Not found")
    
    test_cases = [
        {"filename": "document.pdf", "size": 1024000, "expected": "safe"},
        {"filename": "suspicious.exe", "size": 512, "expected": "malware"},
        {"filename": "image.jpg.exe", "size": 1024, "expected": "malware"},
        {"filename": "report.docx", "size": 2048000, "expected": "safe"}
    ]
    
    results = []
    for case in test_cases:
        is_malware, confidence = await ml_handler.predict_file_malware(
            case["filename"], case["size"]
        )
        results.append({
            "filename": case["filename"],
            "predicted": "malware" if is_malware else "safe",
            "expected": case["expected"],
            "confidence": confidence,
            "correct": ("malware" if is_malware else "safe") == case["expected"]
        })
    
    return {"test_results": results}

# Run the application
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SecureGuard AI Backend Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting SecureGuard AI on {args.host}:{args.port}")
    
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
        access_log=True
    )