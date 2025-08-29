import React, { useState, useEffect } from 'react';
import { Shield, Mail, FileText, AlertTriangle, CheckCircle, Upload, Scan, User, LogOut, RefreshCw, TrendingUp, Eye, Clock, Activity } from 'lucide-react';
import APIService from './services/api';
import './App.css';

const App = () => {
  // State Management
  const [user, setUser] = useState(null);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [scanHistory, setScanHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState({ 
    totalScans: 0, 
    threatsBlocked: 0, 
    safeEmails: 0,
    recentScans24h: 0 
  });
  const [error, setError] = useState(null);
  
  // Email scanning state
  const [emailInput, setEmailInput] = useState('');
  const [senderEmail, setSenderEmail] = useState('');
  const [subject, setSubject] = useState('');
  const [emailScanResult, setEmailScanResult] = useState(null);
  
  // File scanning state
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileScanResult, setFileScanResult] = useState(null);

  // Login form state
  const [loginForm, setLoginForm] = useState({
    email: '',
    password: '',
    isLogin: true,
    fullName: ''
  });

  useEffect(() => {
    // Check for existing authentication
    const token = localStorage.getItem('access_token');
    const savedUser = localStorage.getItem('user');
    
    if (token && savedUser) {
      setUser(JSON.parse(savedUser));
      loadUserData();
    }
  }, []);

  const loadUserData = async () => {
    try {
      setLoading(true);
      
      // Load dashboard stats
      const statsData = await APIService.getDashboardStats();
      setStats({
        totalScans: statsData.total_scans,
        threatsBlocked: statsData.threats_detected,
        safeEmails: statsData.safe_items,
        recentScans24h: statsData.recent_scans_24h
      });

      // Load scan history
      const historyData = await APIService.getScanHistory(50);
      setScanHistory(historyData.scans || []);
      
    } catch (error) {
      console.error('Error loading user data:', error);
      setError('Failed to load user data');
    } finally {
      setLoading(false);
    }
  };

  const handleAuth = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      if (loginForm.isLogin) {
        // Login
        const response = await APIService.login({
          email: loginForm.email,
          password: loginForm.password
        });
        
        setUser(response.user);
        localStorage.setItem('user', JSON.stringify(response.user));
        await loadUserData();
      } else {
        // Register
        const response = await APIService.register({
          email: loginForm.email,
          password: loginForm.password,
          full_name: loginForm.fullName
        });
        
        setUser(response.user);
        localStorage.setItem('user', JSON.stringify(response.user));
        await loadUserData();
      }
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const scanEmail = async () => {
    if (!emailInput.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await APIService.scanEmail({
        email_content: emailInput,
        sender_email: senderEmail || null,
        subject: subject || null
      });
      
      setEmailScanResult(result);
      
      // Refresh dashboard stats and history
      await loadUserData();
      
    } catch (error) {
      console.error('Email scan failed:', error);
      setError('Failed to scan email: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const scanFile = async () => {
    if (!selectedFile) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await APIService.scanFile(selectedFile);
      setFileScanResult(result);
      
      // Refresh dashboard stats and history
      await loadUserData();
      
    } catch (error) {
      console.error('File scan failed:', error);
      setError('Failed to scan file: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    APIService.clearToken();
    setUser(null);
    setScanHistory([]);
    setStats({ totalScans: 0, threatsBlocked: 0, safeEmails: 0, recentScans24h: 0 });
    setEmailScanResult(null);
    setFileScanResult(null);
  };

  const deleteScan = async (scanId) => {
    try {
      await APIService.deleteScan(scanId);
      await loadUserData(); // Refresh data
    } catch (error) {
      console.error('Failed to delete scan:', error);
      setError('Failed to delete scan');
    }
  };

  const exportHistory = async (format = 'json') => {
    try {
      const data = await APIService.exportScanHistory(format);
      
      // Download the file
      const blob = new Blob([format === 'csv' ? data.data : JSON.stringify(data.data)], {
        type: format === 'csv' ? 'text/csv' : 'application/json'
      });
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `scan_history.${format}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Export failed:', error);
      setError('Failed to export scan history');
    }
  };

  // Login Screen
  if (!user) {
    return (
      <div className="login-container">
        <div className="login-card">
          <div className="logo-section">
            <Shield className="logo-icon" />
            <h1>SecureGuard AI</h1>
            <p>Advanced Phishing & Malware Detection</p>
          </div>
          
          <div className="login-content">
            <form onSubmit={handleAuth} className="auth-form">
              <h2>{loginForm.isLogin ? 'Sign In' : 'Create Account'}</h2>
              
              {error && <div className="error-message">{error}</div>}
              
              <div className="form-group">
                <input
                  type="email"
                  placeholder="Email"
                  value={loginForm.email}
                  onChange={(e) => setLoginForm({...loginForm, email: e.target.value})}
                  required
                />
              </div>
              
              {!loginForm.isLogin && (
                <div className="form-group">
                  <input
                    type="text"
                    placeholder="Full Name"
                    value={loginForm.fullName}
                    onChange={(e) => setLoginForm({...loginForm, fullName: e.target.value})}
                    required
                  />
                </div>
              )}
              
              <div className="form-group">
                <input
                  type="password"
                  placeholder="Password"
                  value={loginForm.password}
                  onChange={(e) => setLoginForm({...loginForm, password: e.target.value})}
                  required
                />
              </div>
              
              <button type="submit" className="login-btn" disabled={loading}>
                {loading ? <RefreshCw className="spinning" size={20} /> : <User size={20} />}
                {loading ? 'Please wait...' : (loginForm.isLogin ? 'Sign In' : 'Create Account')}
              </button>
              
              <button
                type="button"
                className="toggle-auth"
                onClick={() => setLoginForm({...loginForm, isLogin: !loginForm.isLogin})}
              >
                {loginForm.isLogin ? 'Need an account? Sign up' : 'Already have an account? Sign in'}
              </button>
            </form>
            
            <div className="features-preview">
              <div className="feature-item">
                <Mail size={16} />
                <span>Email Phishing Detection</span>
              </div>
              <div className="feature-item">
                <FileText size={16} />
                <span>Malware File Scanning</span>
              </div>
              <div className="feature-item">
                <TrendingUp size={16} />
                <span>Real-time Analytics</span>
              </div>
            </div>
          </div>
        </div>
        <div className="background-animation"></div>
      </div>
    );
  }

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <Shield className="logo" />
          <h1>SecureGuard AI</h1>
        </div>
        <div className="header-right">
          <div className="user-info">
            <div className="user-avatar">{user.full_name?.charAt(0) || user.name?.charAt(0) || 'U'}</div>
            <span>{user.full_name || user.name}</span>
          </div>
          <button className="logout-btn" onClick={handleLogout}>
            <LogOut size={18} />
          </button>
        </div>
      </header>

      <div className="main-content">
        {/* Sidebar */}
        <nav className="sidebar">
          <button
            className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            <Activity size={20} />
            Dashboard
          </button>
          <button
            className={`nav-item ${activeTab === 'email' ? 'active' : ''}`}
            onClick={() => setActiveTab('email')}
          >
            <Mail size={20} />
            Email Scanner
          </button>
          <button
            className={`nav-item ${activeTab === 'file' ? 'active' : ''}`}
            onClick={() => setActiveTab('file')}
          >
            <FileText size={20} />
            File Scanner
          </button>
          <button
            className={`nav-item ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            <Clock size={20} />
            Scan History
          </button>
        </nav>

        {/* Content Area */}
        <main className="content">
          {error && (
            <div className="error-banner">
              <AlertTriangle size={20} />
              <span>{error}</span>
              <button onClick={() => setError(null)}>×</button>
            </div>
          )}

          {/* Dashboard */}
          {activeTab === 'dashboard' && (
            <div className="dashboard">
              <div className="dashboard-header">
                <h2>Security Dashboard</h2>
                <button onClick={loadUserData} className="refresh-btn" disabled={loading}>
                  <RefreshCw className={loading ? 'spinning' : ''} size={16} />
                  Refresh
                </button>
              </div>
              
              <div className="stats-grid">
                <div className="stat-card">
                  <div className="stat-icon total">
                    <Scan size={24} />
                  </div>
                  <div className="stat-content">
                    <h3>{stats.totalScans}</h3>
                    <p>Total Scans</p>
                  </div>
                </div>
                <div className="stat-card">
                  <div className="stat-icon danger">
                    <AlertTriangle size={24} />
                  </div>
                  <div className="stat-content">
                    <h3>{stats.threatsBlocked}</h3>
                    <p>Threats Blocked</p>
                  </div>
                </div>
                <div className="stat-card">
                  <div className="stat-icon success">
                    <CheckCircle size={24} />
                  </div>
                  <div className="stat-content">
                    <h3>{stats.safeEmails}</h3>
                    <p>Safe Items</p>
                  </div>
                </div>
                <div className="stat-card">
                  <div className="stat-icon info">
                    <Clock size={24} />
                  </div>
                  <div className="stat-content">
                    <h3>{stats.recentScans24h}</h3>
                    <p>Recent (24h)</p>
                  </div>
                </div>
              </div>

              <div className="recent-activity">
                <h3>Recent Activity</h3>
                <div className="activity-list">
                  {scanHistory.slice(0, 5).map((scan) => (
                    <div key={scan.scan_id || scan._id} className="activity-item">
                      <div className={`activity-icon ${scan.is_phishing || scan.is_malware ? 'danger' : 'success'}`}>
                        {scan.scan_type === 'email' ? <Mail size={16} /> : <FileText size={16} />}
                      </div>
                      <div className="activity-content">
                        <p>{scan.scan_type === 'email' ? 'Email Scan' : 'File Scan'}</p>
                        <small>{new Date(scan.timestamp || scan.created_at).toLocaleString()}</small>
                      </div>
                      <div className={`activity-status ${scan.is_phishing || scan.is_malware ? 'danger' : 'success'}`}>
                        {scan.is_phishing || scan.is_malware ? 'Threat Detected' : 'Safe'}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Email Scanner */}
          {activeTab === 'email' && (
            <div className="scanner-section">
              <h2>Email Phishing Scanner</h2>
              <div className="scanner-card">
                <div className="input-section">
                  <div className="form-row">
                    <input
                      type="email"
                      value={senderEmail}
                      onChange={(e) => setSenderEmail(e.target.value)}
                      placeholder="Sender email (optional)"
                      className="email-meta-input"
                    />
                    <input
                      type="text"
                      value={subject}
                      onChange={(e) => setSubject(e.target.value)}
                      placeholder="Email subject (optional)"
                      className="email-meta-input"
                    />
                  </div>
                  <textarea
                    value={emailInput}
                    onChange={(e) => setEmailInput(e.target.value)}
                    placeholder="Paste suspicious email content here..."
                    rows="6"
                    className="email-input"
                  />
                  <button
                    onClick={scanEmail}
                    disabled={!emailInput.trim() || loading}
                    className="scan-btn"
                  >
                    {loading ? <RefreshCw className="spinning" size={20} /> : <Scan size={20} />}
                    {loading ? 'Analyzing...' : 'Scan Email'}
                  </button>
                </div>

                {emailScanResult && (
                  <div className="scan-result">
                    <div className={`result-header ${emailScanResult.is_phishing ? 'danger' : 'success'}`}>
                      {emailScanResult.is_phishing ? <AlertTriangle size={24} /> : <CheckCircle size={24} />}
                      <h3>{emailScanResult.is_phishing ? 'Phishing Detected!' : 'Email Appears Safe'}</h3>
                      <div className="confidence-score">
                        Confidence: {emailScanResult.confidence_score.toFixed(1)}%
                      </div>
                    </div>
                    
                    <div className="result-details">
                      <div className="detail-section">
                        <h4>Risk Analysis</h4>
                        <ul>
                          {emailScanResult.risk_factors.map((factor, index) => (
                            <li key={index}>{factor}</li>
                          ))}
                        </ul>
                      </div>
                      <div className="detail-section">
                        <h4>Sender Reputation</h4>
                        <p>{emailScanResult.sender_reputation}</p>
                      </div>
                      <div className="detail-section">
                        <h4>URL Analysis</h4>
                        <p>{emailScanResult.url_analysis}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* File Scanner */}
          {activeTab === 'file' && (
            <div className="scanner-section">
              <h2>File Malware Scanner</h2>
              <div className="scanner-card">
                <div className="file-upload-section">
                  <div 
                    className="upload-area" 
                    onClick={() => document.getElementById('file-input').click()}
                    onDrop={(e) => {
                      e.preventDefault();
                      const files = e.dataTransfer.files;
                      if (files.length > 0) {
                        setSelectedFile(files[0]);
                      }
                    }}
                    onDragOver={(e) => e.preventDefault()}
                  >
                    <Upload size={48} />
                    <h3>Drop files here or click to browse</h3>
                    <p>Supported formats: All file types (Max 50MB)</p>
                    {selectedFile && (
                      <div className="selected-file">
                        <FileText size={20} />
                        <span>{selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)</span>
                      </div>
                    )}
                  </div>
                  <input
                    id="file-input"
                    type="file"
                    onChange={(e) => setSelectedFile(e.target.files[0])}
                    style={{ display: 'none' }}
                  />
                  <button
                    onClick={scanFile}
                    disabled={!selectedFile || loading}
                    className="scan-btn"
                  >
                    {loading ? <RefreshCw className="spinning" size={20} /> : <Scan size={20} />}
                    {loading ? 'Scanning...' : 'Scan File'}
                  </button>
                </div>

                {fileScanResult && (
                  <div className="scan-result">
                    <div className={`result-header ${fileScanResult.is_malware ? 'danger' : 'success'}`}>
                      {fileScanResult.is_malware ? <AlertTriangle size={24} /> : <CheckCircle size={24} />}
                      <h3>{fileScanResult.is_malware ? 'Malware Detected!' : 'File Appears Safe'}</h3>
                      <div className="confidence-score">
                        Confidence: {fileScanResult.confidence_score.toFixed(1)}%
                      </div>
                    </div>
                    
                    <div className="result-details">
                      <div className="detail-section">
                        <h4>File Information</h4>
                        <p><strong>Name:</strong> {fileScanResult.filename}</p>
                        <p><strong>Size:</strong> {(fileScanResult.file_size / 1024).toFixed(2)} KB</p>
                        <p><strong>Type:</strong> {fileScanResult.file_type}</p>
                      </div>
                      <div className="detail-section">
                        <h4>Risk Factors</h4>
                        <ul>
                          {fileScanResult.risk_factors.map((factor, index) => (
                            <li key={index}>{factor}</li>
                          ))}
                        </ul>
                      </div>
                      <div className="detail-section">
                        <h4>Analysis Results</h4>
                        <p><strong>Hash Analysis:</strong> {fileScanResult.hash_analysis}</p>
                        <p><strong>Behavioral Analysis:</strong> {fileScanResult.behavioral_analysis}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* History */}
          {activeTab === 'history' && (
            <div className="history-section">
              <div className="history-header">
                <h2>Scan History</h2>
                <div className="history-actions">
                  <button onClick={() => exportHistory('json')} className="export-btn">
                    Export JSON
                  </button>
                  <button onClick={() => exportHistory('csv')} className="export-btn">
                    Export CSV
                  </button>
                </div>
              </div>

              <div className="history-list">
                {scanHistory.map((scan) => (
                  <div key={scan.scan_id || scan._id} className="history-item">
                    <div className="history-icon">
                      {scan.scan_type === 'email' ? <Mail size={20} /> : <FileText size={20} />}
                    </div>
                    <div className="history-content">
                      <h4>{scan.scan_type === 'email' ? 'Email Scan' : `File: ${scan.filename}`}</h4>
                      <p className="history-timestamp">{new Date(scan.timestamp || scan.created_at).toLocaleString()}</p>
                      {scan.email_content && (
                        <p className="history-preview">{scan.email_content.substring(0, 100)}...</p>
                      )}
                    </div>
                    <div className="history-result">
                      <div className={`confidence-badge ${scan.is_phishing || scan.is_malware ? 'danger' : 'success'}`}>
                        {scan.confidence_score.toFixed(1)}%
                      </div>
                      <span className={`status-text ${scan.is_phishing || scan.is_malware ? 'danger' : 'success'}`}>
                        {scan.is_phishing || scan.is_malware ? 'Threat' : 'Safe'}
                      </span>
                    </div>
                    <div className="history-actions">
                      <button className="view-details-btn">
                        <Eye size={16} />
                      </button>
                      <button 
                        className="delete-btn"
                        onClick={() => deleteScan(scan.scan_id)}
                      >
                        ×
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default App;