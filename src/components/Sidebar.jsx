import React from 'react';

const Sidebar = ({ activeSection, onSectionChange }) => {
  const menuItems = [
    {
      id: 'profile',
      title: 'Patient Management',
      description: 'Add & manage patient information',
      icon: '👤'
    },
    {
      id: 'analysis',
      title: 'AI Dementia Analysis',
      description: 'Upload MRI & biomarker data for analysis',
      icon: '🧠'
    }
  ];

  return (
    <div className="sidebar">
      <div style={{ marginBottom: '20px', padding: '15px', background: 'rgba(102, 126, 234, 0.1)', borderRadius: '10px' }}>
        <h3 style={{ margin: '0 0 5px 0', color: '#667eea' }}>Medical AI Dashboard</h3>
        <p style={{ margin: '0', fontSize: '12px', color: '#666' }}>
          AI-powered dementia analysis system for healthcare professionals
        </p>
      </div>

      {menuItems.map((item) => (
        <div
          key={item.id}
          className={`sidebar-item ${activeSection === item.id ? 'active' : ''}`}
          onClick={() => onSectionChange(item.id)}
          style={{ cursor: 'pointer' }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '20px' }}>{item.icon}</span>
            <div>
              <strong>{item.title}</strong>
              <div style={{ fontSize: '12px', color: activeSection === item.id ? 'rgba(255,255,255,0.8)' : '#666', marginTop: '5px' }}>
                {item.description}
              </div>
            </div>
          </div>
        </div>
      ))}

      <div style={{ marginTop: '30px', padding: '15px', background: 'rgba(255, 193, 7, 0.1)', borderRadius: '10px' }}>
        <h4 style={{ margin: '0 0 10px 0', color: '#856404', fontSize: '14px' }}>Quick Guide:</h4>
        <ul style={{ margin: '0', paddingLeft: '15px', fontSize: '12px', color: '#666', lineHeight: '1.4' }}>
          <li>Add patient details in Patient Management</li>
          <li>Upload MRI images and biomarker CSV/Excel files</li>
          <li>Get AI predictions and generate care plans</li>
          <li>Export comprehensive PDF reports</li>
        </ul>
      </div>
    </div>
  );
};

export default Sidebar;