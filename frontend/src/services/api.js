// src/services/api.js
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://phishing-email-detector-wiom.onrender.com'
  : process.env.REACT_APP_API_URL || 'http://localhost:8000';

class APIService {
  constructor() {
    this.token = localStorage.getItem('access_token');
    this.loading = new Set();
    this.cache = new Map();
    this.cacheExpiry = new Map();
  }

  // Set authorization header
  getHeaders() {
    const headers = {
      'Content-Type': 'application/json',
    };
    
    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`;
    }
    
    return headers;
  }

  // Set token after login
  setToken(token) {
    this.token = token;
    localStorage.setItem('access_token', token);
  }

  // Clear token on logout
  clearToken() {
    this.token = null;
    localStorage.removeItem('access_token');
    localStorage.removeItem('user');
    this.cache.clear();
    this.cacheExpiry.clear();
  }

  // Loading state management
  setLoading(operation, isLoading) {
    if (isLoading) {
      this.loading.add(operation);
    } else {
      this.loading.delete(operation);
    }
    
    window.dispatchEvent(new CustomEvent('apiLoadingChange', {
      detail: { loading: this.loading.size > 0, operations: [...this.loading] }
    }));
  }

  // Fetch with timeout
  async fetchWithTimeout(url, options, timeout = 30000) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal
      });
      clearTimeout(id);
      return response;
    } catch (error) {
      clearTimeout(id);
      if (error.name === 'AbortError') {
        throw new Error('Request timeout. Please check your connection.');
      }
      throw error;
    }
  }

  // Retry logic for failed requests
  async makeRequestWithRetry(url, options, maxRetries = 3) {
    let lastError;
    
    for (let i = 0; i <= maxRetries; i++) {
      try {
        const response = await this.fetchWithTimeout(url, options);
        return response;
      } catch (error) {
        lastError = error;
        
        // Don't retry on client errors (4xx) except 408, 429
        if (error.status >= 400 && error.status < 500 && 
            error.status !== 408 && error.status !== 429) {
          throw error;
        }
        
        if (i < maxRetries) {
          // Exponential backoff: wait 1s, 2s, 4s
          await new Promise(resolve => setTimeout(resolve, Math.pow(2, i) * 1000));
        }
      }
    }
    
    throw lastError;
  }

  // Token refresh handling
  async refreshTokenIfNeeded(response) {
    if (response.status === 401) {
      this.clearToken();
      window.location.href = '/login';
      throw new Error('Session expired. Please login again.');
    }
    return response;
  }

  // Enhanced error handling
  async handleResponse(response) {
    await this.refreshTokenIfNeeded(response);
    
    if (!response.ok) {
      let errorMessage;
      try {
        const error = await response.json();
        errorMessage = error.detail || error.message;
      } catch {
        errorMessage = `HTTP error! status: ${response.status}`;
      }

      switch (response.status) {
        case 413:
          throw new Error('File too large. Maximum size is 50MB');
        case 429:
          throw new Error('Too many requests. Please wait and try again.');
        case 500:
          throw new Error('Server error. Please try again later.');
        default:
          throw new Error(errorMessage);
      }
    }
    return response.json();
  }

  // Caching utility
  async getCachedData(key, fetchFunction, cacheTime = 5 * 60 * 1000) {
    const now = Date.now();
    const expiry = this.cacheExpiry.get(key);
    
    if (this.cache.has(key) && expiry && now < expiry) {
      return this.cache.get(key);
    }
    
    const data = await fetchFunction();
    this.cache.set(key, data);
    this.cacheExpiry.set(key, now + cacheTime);
    return data;
  }

  // Auth endpoints
  async register(userData) {
    this.setLoading('register', true);
    try {
      const response = await this.makeRequestWithRetry(`${API_BASE_URL}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(userData),
      });
      
      return this.handleResponse(response);
    } finally {
      this.setLoading('register', false);
    }
  }

  async login(credentials) {
    this.setLoading('login', true);
    try {
      const response = await this.makeRequestWithRetry(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(credentials),
      });
      
      const data = await this.handleResponse(response);
      this.setToken(data.access_token);
      return data;
    } finally {
      this.setLoading('login', false);
    }
  }

  // Email scanning
  async scanEmail(emailData) {
    this.setLoading('scanEmail', true);
    try {
      const response = await this.makeRequestWithRetry(`${API_BASE_URL}/scan/email`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify(emailData),
      });
      
      return this.handleResponse(response);
    } finally {
      this.setLoading('scanEmail', false);
    }
  }

  // File scanning
  async scanFile(file) {
    this.setLoading('scanFile', true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await this.makeRequestWithRetry(`${API_BASE_URL}/scan/file`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.token}`,
        },
        body: formData,
      });
      
      return this.handleResponse(response);
    } finally {
      this.setLoading('scanFile', false);
    }
  }

  // Batch email scanning
  async batchScanEmails(emails) {
    this.setLoading('batchScan', true);
    try {
      const response = await this.makeRequestWithRetry(`${API_BASE_URL}/scan/batch-email`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify(emails),
      });
      
      return this.handleResponse(response);
    } finally {
      this.setLoading('batchScan', false);
    }
  }

  // Get scan history
  async getScanHistory(limit = 50) {
    return this.getCachedData(`scanHistory_${limit}`, async () => {
      const response = await this.makeRequestWithRetry(`${API_BASE_URL}/scan/history?limit=${limit}`, {
        method: 'GET',
        headers: this.getHeaders(),
      });
      
      return this.handleResponse(response);
    }, 2 * 60 * 1000); // Cache for 2 minutes
  }

  // Get dashboard stats
  async getDashboardStats() {
    return this.getCachedData('dashboardStats', async () => {
      const response = await this.makeRequestWithRetry(`${API_BASE_URL}/stats/dashboard`, {
        method: 'GET',
        headers: this.getHeaders(),
      });
      
      return this.handleResponse(response);
    });
  }

  // Get user profile
  async getUserProfile() {
    return this.getCachedData('userProfile', async () => {
      const response = await this.makeRequestWithRetry(`${API_BASE_URL}/user/profile`, {
        method: 'GET',
        headers: this.getHeaders(),
      });
      
      return this.handleResponse(response);
    });
  }

  // Update user profile
  async updateUserProfile(profileData) {
    this.setLoading('updateProfile', true);
    try {
      const formData = new FormData();
      formData.append('full_name', profileData.full_name);

      const response = await this.makeRequestWithRetry(`${API_BASE_URL}/user/profile`, {
        method: 'PUT',
        headers: {
          Authorization: `Bearer ${this.token}`,
        },
        body: formData,
      });
      
      // Clear cached profile data
      this.cache.delete('userProfile');
      this.cacheExpiry.delete('userProfile');
      
      return this.handleResponse(response);
    } finally {
      this.setLoading('updateProfile', false);
    }
  }

  // Delete scan
  async deleteScan(scanId) {
    this.setLoading('deleteScan', true);
    try {
      const response = await this.makeRequestWithRetry(`${API_BASE_URL}/scan/${scanId}`, {
        method: 'DELETE',
        headers: this.getHeaders(),
      });
      
      // Clear cached scan history
      Array.from(this.cache.keys())
        .filter(key => key.startsWith('scanHistory_'))
        .forEach(key => {
          this.cache.delete(key);
          this.cacheExpiry.delete(key);
        });
      
      return this.handleResponse(response);
    } finally {
      this.setLoading('deleteScan', false);
    }
  }

  // Export scan history
  async exportScanHistory(format = 'json') {
    this.setLoading('export', true);
    try {
      const response = await this.makeRequestWithRetry(`${API_BASE_URL}/scan/export?format_type=${format}`, {
        method: 'GET',
        headers: this.getHeaders(),
      });
      
      return this.handleResponse(response);
    } finally {
      this.setLoading('export', false);
    }
  }

  // Health check
  async healthCheck() {
    const response = await fetch(`${API_BASE_URL}/health`);
    return this.handleResponse(response);
  }

  // Utility methods
  isLoading(operation = null) {
    return operation ? this.loading.has(operation) : this.loading.size > 0;
  }

  clearCache() {
    this.cache.clear();
    this.cacheExpiry.clear();
  }
}

export default new APIService();