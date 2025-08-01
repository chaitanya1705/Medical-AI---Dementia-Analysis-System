import React, { useState } from 'react';
import { useAuth } from '../App';

const Login = () => {
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  });
  const [isRegister, setIsRegister] = useState(false);
  const [registerData, setRegisterData] = useState({
    username: '',
    password: '',
    confirmPassword: '',
    email: '',
    firstName: '',
    lastName: '',
    specialty: '',
    hospital: '',
    license: ''
  });
  const [loading, setLoading] = useState(false);

  const { login, register } = useAuth();

  const handleLoginChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleRegisterChange = (e) => {
    setRegisterData({
      ...registerData,
      [e.target.name]: e.target.value
    });
  };

  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const result = await login(formData);
    
    setLoading(false);
    
    if (result.success) {
      // Navigation handled by App component
    }
  };

  const handleRegisterSubmit = async (e) => {
    e.preventDefault();
    
    if (registerData.password !== registerData.confirmPassword) {
      alert('Passwords do not match');
      return;
    }

    setLoading(true);

    const { confirmPassword, ...submitData } = registerData;
    const result = await register(submitData);
    
    setLoading(false);
    
    if (result.success) {
      // Navigation handled by App component
    }
  };

  const toggleMode = () => {
    setIsRegister(!isRegister);
    setFormData({ username: '', password: '' });
    setRegisterData({
      username: '',
      password: '',
      confirmPassword: '',
      email: '',
      firstName: '',
      lastName: '',
      specialty: '',
      hospital: '',
      license: ''
    });
  };

  return (
    <div className="login-container">
      <form 
        className={`login-form ${isRegister ? 'registration-form' : ''}`} 
        onSubmit={isRegister ? handleRegisterSubmit : handleLoginSubmit}
      >
        <div className="logo-section" style={{ justifyContent: 'center', marginBottom: '20px' }}>
          <img src="/logo.png" alt="Company Logo" className="logo-image" />
        </div>
        
        <h2>{isRegister ? 'Doctor Registration' : 'Doctor Login'}</h2>
        <p style={{ marginBottom: '20px', color: '#666', textAlign: 'center' }}>
          {isRegister ? 'Register as a medical professional' : 'Login to access the AI analysis system'}
        </p>

        {isRegister ? (
          // Doctor Registration Form
          <>
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="firstName">First Name:</label>
                <input
                  type="text"
                  id="firstName"
                  name="firstName"
                  value={registerData.firstName}
                  onChange={handleRegisterChange}
                  required
                  
                />
              </div>
              <div className="form-group">
                <label htmlFor="lastName">Last Name:</label>
                <input
                  type="text"
                  id="lastName"
                  name="lastName"
                  value={registerData.lastName}
                  onChange={handleRegisterChange}
                  required
                 
                />
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="username">Username:</label>
              <input
                type="text"
                id="username"
                name="username"
                value={registerData.username}
                onChange={handleRegisterChange}
                required
                
              />
            </div>

            <div className="form-group">
              <label htmlFor="email">Email:</label>
              <input
                type="email"
                id="email"
                name="email"
                value={registerData.email}
                onChange={handleRegisterChange}
               
              />
            </div>

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="specialty">Medical Specialty:</label>
                <input
                  type="text"
                  id="specialty"
                  name="specialty"
                  value={registerData.specialty}
                  onChange={handleRegisterChange}
                  
                />
              </div>
              <div className="form-group">
                <label htmlFor="hospital">Hospital/Clinic:</label>
                <input
                  type="text"
                  id="hospital"
                  name="hospital"
                  value={registerData.hospital}
                  onChange={handleRegisterChange}
                  
                />
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="license">Medical License Number:</label>
              <input
                type="text"
                id="license"
                name="license"
                value={registerData.license}
                onChange={handleRegisterChange}
               
              />
            </div>

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="password">Password:</label>
                <input
                  type="password"
                  id="password"
                  name="password"
                  value={registerData.password}
                  onChange={handleRegisterChange}
                  required
                  placeholder="Secure password"
                />
              </div>
              <div className="form-group">
                <label htmlFor="confirmPassword">Confirm Password:</label>
                <input
                  type="password"
                  id="confirmPassword"
                  name="confirmPassword"
                  value={registerData.confirmPassword}
                  onChange={handleRegisterChange}
                  required
                  placeholder="Confirm password"
                />
              </div>
            </div>
          </>
        ) : (
          // Doctor Login Form
          <>
            <div className="form-group">
              <label htmlFor="username">Username:</label>
              <input
                type="text"
                id="username"
                name="username"
                value={formData.username}
                onChange={handleLoginChange}
                required
                placeholder="Enter your username"
              />
            </div>
            <div className="form-group">
              <label htmlFor="password">Password:</label>
              <input
                type="password"
                id="password"
                name="password"
                value={formData.password}
                onChange={handleLoginChange}
                required
                placeholder="Enter your password"
              />
            </div>
          </>
        )}

        <button 
          type="submit" 
          className="login-btn"
          disabled={loading}
        >
          {loading ? 'Please wait...' : (isRegister ? 'Register as Doctor' : 'Login')}
        </button>

        <div style={{ marginTop: '20px', textAlign: 'center' }}>
          <button
            type="button"
            onClick={toggleMode}
            style={{
              background: 'none',
              border: 'none',
              color: '#667eea',
              textDecoration: 'underline',
              cursor: 'pointer',
              fontSize: '14px'
            }}
          >
            {isRegister ? 'Already registered? Login here' : "New doctor? Register here"}
          </button>
        </div>

        <div style={{ marginTop: '15px', padding: '10px', backgroundColor: '#f8f9fa', borderRadius: '5px', fontSize: '12px', color: '#666' }}>
          <strong>Note:</strong> This system is for licensed medical professionals only. 
          After login, you can manage patient profiles and perform AI-powered dementia analysis.
        </div>
      </form>
    </div>
  );
};

export default Login;