// src/services/api.js
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class APIService {
  constructor() {
    this.token = localStorage.getItem('access_token');
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
  }

  // Handle API errors
  async handleResponse(response) {
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }
    return response.json();
  }

  // Auth endpoints
  async register(userData) {
    const response = await fetch(`${API_BASE_URL}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(userData),
    });
    
    return this.handleResponse(response);
  }

  async login(credentials) {
    const response = await fetch(`${API_BASE_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(credentials),
    });
    
    const data = await this.handleResponse(response);
    this.setToken(data.access_token);
    return data;
  }

  // Email scanning
  async scanEmail(emailData) {
    const response = await fetch(`${API_BASE_URL}/scan/email`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(emailData),
    });
    
    return this.handleResponse(response);
  }

  // File scanning
  async scanFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/scan/file`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.token}`,
      },
      body: formData,
    });
    
    return this.handleResponse(response);
  }

  // Batch email scanning
  async batchScanEmails(emails) {
    const response = await fetch(`${API_BASE_URL}/scan/batch-email`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(emails),
    });
    
    return this.handleResponse(response);
  }

  // Get scan history
  async getScanHistory(limit = 50) {
    const response = await fetch(`${API_BASE_URL}/scan/history?limit=${limit}`, {
      method: 'GET',
      headers: this.getHeaders(),
    });
    
    return this.handleResponse(response);
  }

  // Get dashboard stats
  async getDashboardStats() {
    const response = await fetch(`${API_BASE_URL}/stats/dashboard`, {
      method: 'GET',
      headers: this.getHeaders(),
    });
    
    return this.handleResponse(response);
  }

  // Get user profile
  async getUserProfile() {
    const response = await fetch(`${API_BASE_URL}/user/profile`, {
      method: 'GET',
      headers: this.getHeaders(),
    });
    
    return this.handleResponse(response);
  }

  // Update user profile
  async updateUserProfile(profileData) {
    const formData = new FormData();
    formData.append('full_name', profileData.full_name);

    const response = await fetch(`${API_BASE_URL}/user/profile`, {
      method: 'PUT',
      headers: {
        Authorization: `Bearer ${this.token}`,
      },
      body: formData,
    });
    
    return this.handleResponse(response);
  }

  // Delete scan
  async deleteScan(scanId) {
    const response = await fetch(`${API_BASE_URL}/scan/${scanId}`, {
      method: 'DELETE',
      headers: this.getHeaders(),
    });
    
    return this.handleResponse(response);
  }

  // Export scan history
  async exportScanHistory(format = 'json') {
    const response = await fetch(`${API_BASE_URL}/scan/export?format_type=${format}`, {
      method: 'GET',
      headers: this.getHeaders(),
    });
    
    return this.handleResponse(response);
  }

  // Health check
  async healthCheck() {
    const response = await fetch(`${API_BASE_URL}/health`);
    return this.handleResponse(response);
  }
}

export default new APIService();