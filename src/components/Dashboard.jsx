import React, { useState } from 'react';
import { useAuth } from '../App';
import Header from './Header';
import Sidebar from './Sidebar';
import Profile from './Profile';
import Analysis from './Analysis';

const Dashboard = () => {
  const [activeSection, setActiveSection] = useState('profile');
  const { user } = useAuth();

  const showSection = (section) => {
    setActiveSection(section);
  };

  const renderContent = () => {
    switch (activeSection) {
      case 'profile':
        return <Profile />;
      case 'analysis':
        return <Analysis />;
      default:
        return <Profile />;
    }
  };

  return (
    <div className="dashboard">
      <div className="container">
        <Header />
        
        <div className="dashboard-content">
          <Sidebar 
            activeSection={activeSection} 
            onSectionChange={showSection} 
          />
          
          <div className="main-content">
            {renderContent()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;