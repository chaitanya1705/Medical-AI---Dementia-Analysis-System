import React, { useState, useEffect, createContext, useContext } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import axios from 'axios';

// Components
import Login from './components/Login';
import Dashboard from './components/Dashboard';
import LoadingSpinner from './components/LoadingSpinner';
import Notification from './components/Notification';

// Styles
import './App.css';

// Auth Context
const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Notification Context
const NotificationContext = createContext();

export const useNotification = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotification must be used within a NotificationProvider');
  }
  return context;
};

// FIXED: Axios Configuration - Only set baseURL once
axios.defaults.baseURL = '/api';
axios.defaults.timeout = 30000; // 30 second timeout

// Set auth token for all requests
const setAuthToken = (token) => {
  if (token) {
    axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    localStorage.setItem('authToken', token);
  } else {
    delete axios.defaults.headers.common['Authorization'];
    localStorage.removeItem('authToken');
  }
};

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [notifications, setNotifications] = useState([]);
  const [authChecked, setAuthChecked] = useState(false);

  // Auth functions
  const login = async (credentials) => {
    try {
      console.log('Attempting login...', credentials);
      const response = await axios.post('/auth/login', credentials);
      const { token, user: userData } = response.data;
      
      setAuthToken(token);
      setUser(userData);
      showNotification('Login successful!', 'success');
      
      return { success: true };
    } catch (error) {
      console.error('Login error:', error);
      const message = error.response?.data?.error || 'Login failed';
      showNotification(message, 'error');
      return { success: false, error: message };
    }
  };

  const register = async (userData) => {
    try {
      console.log('Attempting registration...', userData);
      const response = await axios.post('/auth/register', userData);
      const { token, user: newUser } = response.data;
      
      setAuthToken(token);
      setUser(newUser);
      showNotification('Registration successful!', 'success');
      
      return { success: true };
    } catch (error) {
      console.error('Registration error:', error);
      const message = error.response?.data?.error || 'Registration failed';
      showNotification(message, 'error');
      return { success: false, error: message };
    }
  };

  const logout = () => {
    console.log('Logging out...');
    setAuthToken(null);
    setUser(null);
    showNotification('Logged out successfully', 'info');
  };

  const updateProfile = async (profileData) => {
    try {
      await axios.put('/auth/profile', profileData);
      
      // Update local user data
      setUser(prev => ({
        ...prev,
        ...profileData
      }));
      
      showNotification('Profile updated successfully!', 'success');
      return { success: true };
    } catch (error) {
      const message = error.response?.data?.error || 'Profile update failed';
      showNotification(message, 'error');
      return { success: false, error: message };
    }
  };

  // Notification functions
  const showNotification = (message, type = 'info') => {
    const id = Date.now();
    const notification = { id, message, type };
    
    setNotifications(prev => [...prev, notification]);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 5000);
  };

  const removeNotification = (id) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  // Check for existing auth token on app start
  useEffect(() => {
    const checkAuth = async () => {
      console.log('Checking authentication...');
      const token = localStorage.getItem('authToken');
      
      if (token) {
        setAuthToken(token);
        try {
          console.log('Verifying token...');
          const response = await axios.get('/auth/verify');
          console.log('Token verified, user:', response.data.user);
          setUser(response.data.user);
        } catch (error) {
          console.error('Token verification failed:', error);
          setAuthToken(null);
          showNotification('Session expired. Please login again.', 'warning');
        }
      } else {
        console.log('No token found');
      }
      
      setAuthChecked(true);
      setLoading(false);
    };

    checkAuth();
  }, []);

  // Add axios interceptor for handling auth errors
  useEffect(() => {
    const interceptor = axios.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          console.log('Unauthorized, logging out...');
          logout();
        }
        return Promise.reject(error);
      }
    );

    return () => {
      axios.interceptors.response.eject(interceptor);
    };
  }, []);

  if (loading || !authChecked) {
    return <LoadingSpinner />;
  }

  return (
    <AuthContext.Provider value={{ 
      user, 
      login, 
      register, 
      logout, 
      updateProfile,
      isAuthenticated: !!user 
    }}>
      <NotificationContext.Provider value={{ 
        showNotification, 
        removeNotification 
      }}>
        <Router>
          <div className="App">
            {/* Notification Container */}
            <div className="notification-container">
              {notifications.map(notification => (
                <Notification
                  key={notification.id}
                  {...notification}
                  onRemove={removeNotification}
                />
              ))}
            </div>

            <Routes>
              {/* Public Routes */}
              <Route path="/login" element={
                user ? <Navigate to="/dashboard" replace /> : <Login />
              } />

              {/* Protected Routes */}
              <Route path="/dashboard" element={
                user ? <Dashboard /> : <Navigate to="/login" replace />
              } />

              {/* Default redirect */}
              <Route path="/" element={
                <Navigate to={user ? "/dashboard" : "/login"} replace />
              } />

              {/* 404 fallback */}
              <Route path="*" element={
                <Navigate to={user ? "/dashboard" : "/login"} replace />
              } />
            </Routes>
          </div>
        </Router>
      </NotificationContext.Provider>
    </AuthContext.Provider>
  );
}

export default App;