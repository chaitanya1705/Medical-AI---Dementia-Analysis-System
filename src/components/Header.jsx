import React from 'react';
import { useAuth } from '../App';

const Header = () => {
  const { user, logout } = useAuth();

  const handleLogout = () => {
    logout();
  };

  return (
    <div className="header">
      <div className="logo-section">
        <img src="/logo.png" alt="Company Logo" className="logo-image" />
        <div className="company-name">Medical AI - Dementia Analysis System</div>
      </div>
      <div className="user-info">
        <span id="welcomeUser">
          Welcome, Dr. {user?.firstName || user?.username}
        </span>
        <button className="logout-btn" onClick={handleLogout}>
          Logout
        </button>
      </div>
    </div>
  );
};

export default Header;