import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNotification } from '../App';

const PatientSelector = ({ selectedPatient, onPatientSelect }) => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const { showNotification } = useNotification();

  useEffect(() => {
    fetchPatients();
  }, []);

  const fetchPatients = async () => {
    setLoading(true);
    try {
      // FIXED: Remove /api/ prefix since it's already in axios.defaults.baseURL
      const response = await axios.get('/patients', {
        params: { limit: 50 } // Get more patients for selection
      });
      
      if (response.data.success) {
        setPatients(response.data.data);
        
        // Auto-select first patient if none selected and patients exist
        if (!selectedPatient && response.data.data.length > 0) {
          onPatientSelect(response.data.data[0]);
        }
      }
    } catch (error) {
      console.error('Error fetching patients:', error);
      showNotification('Failed to load patients. Please check if the server is running.', 'error');
    } finally {
      setLoading(false);
    }
  };

  const formatPatientDisplay = (patient) => {
    const age = patient.date_of_birth 
      ? new Date().getFullYear() - new Date(patient.date_of_birth).getFullYear()
      : 'Unknown';
    
    const gender = patient.gender ? ` • ${patient.gender}` : '';
    const condition = patient.medical_condition ? ` • ${patient.medical_condition}` : '';
    const patientId = patient.patient_id_number ? ` (ID: ${patient.patient_id_number})` : '';
    
    return {
      name: `${patient.first_name} ${patient.last_name}`,
      details: `Age: ${age}${gender}${condition}${patientId}`,
      fullDisplay: `${patient.first_name} ${patient.last_name} - Age: ${age}${gender}${condition}${patientId}`
    };
  };

  const handlePatientSelect = (patient) => {
    onPatientSelect(patient);
    setShowDropdown(false);
  };

  const createNewPatient = () => {
    // You can implement this to open a patient creation modal
    showNotification('Please create a new patient in the Patients section first', 'info');
  };

  return (
    <div className="patient-selector">
      <div className="selector-header">
        <h3>👤 Select Patient for Analysis</h3>
        <p>Choose the patient whose MRI and biomarker data will be analyzed</p>
      </div>

      <div className="patient-dropdown-container">
        <div 
          className={`patient-dropdown-trigger ${showDropdown ? 'active' : ''}`}
          onClick={() => setShowDropdown(!showDropdown)}
        >
          {selectedPatient ? (
            <div className="selected-patient">
              <div className="patient-name">
                {formatPatientDisplay(selectedPatient).name}
              </div>
              <div className="patient-details">
                {formatPatientDisplay(selectedPatient).details}
              </div>
            </div>
          ) : (
            <div className="no-selection">
              <div className="placeholder-text">Select a patient...</div>
            </div>
          )}
          <div className="dropdown-arrow">
            {showDropdown ? '▲' : '▼'}
          </div>
        </div>

        {showDropdown && (
          <div className="patient-dropdown-menu">
            {loading ? (
              <div className="loading-item">Loading patients...</div>
            ) : patients.length === 0 ? (
              <div className="no-patients">
                <div className="no-patients-text">No patients found</div>
                <button 
                  className="create-patient-btn"
                  onClick={createNewPatient}
                >
                  + Create New Patient
                </button>
              </div>
            ) : (
              patients.map((patient) => {
                const formatted = formatPatientDisplay(patient);
                const isSelected = selectedPatient?.id === patient.id;
                
                return (
                  <div
                    key={patient.id}
                    className={`patient-option ${isSelected ? 'selected' : ''}`}
                    onClick={() => handlePatientSelect(patient)}
                  >
                    <div className="patient-option-name">
                      {formatted.name}
                    </div>
                    <div className="patient-option-details">
                      {formatted.details}
                    </div>
                    {isSelected && <div className="selected-indicator">✓</div>}
                  </div>
                );
              })
            )}
          </div>
        )}
      </div>

      {selectedPatient && (
        <div className="selected-patient-summary">
          <div className="summary-title">Selected Patient:</div>
          <div className="summary-content">
            <strong>{formatPatientDisplay(selectedPatient).name}</strong>
            <br />
            {formatPatientDisplay(selectedPatient).details}
          </div>
        </div>
      )}

      <style>{`
        .patient-selector {
          background: #f8f9fa;
          border: 1px solid #e0e0e0;
          border-radius: 12px;
          padding: 20px;
          margin-bottom: 25px;
        }

        .selector-header h3 {
          margin: 0 0 8px 0;
          color: #333;
          font-size: 18px;
        }

        .selector-header p {
          margin: 0 0 20px 0;
          color: #666;
          font-size: 14px;
        }

        .patient-dropdown-container {
          position: relative;
        }

        .patient-dropdown-trigger {
          background: white;
          border: 2px solid #ddd;
          border-radius: 8px;
          padding: 15px;
          cursor: pointer;
          display: flex;
          justify-content: space-between;
          align-items: center;
          min-height: 60px;
          transition: all 0.2s ease;
        }

        .patient-dropdown-trigger:hover {
          border-color: #667eea;
        }

        .patient-dropdown-trigger.active {
          border-color: #667eea;
          box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .selected-patient {
          flex: 1;
        }

        .patient-name {
          font-weight: 600;
          font-size: 16px;
          color: #333;
          margin-bottom: 4px;
        }

        .patient-details {
          font-size: 14px;
          color: #666;
        }

        .no-selection .placeholder-text {
          color: #999;
          font-style: italic;
        }

        .dropdown-arrow {
          color: #667eea;
          font-weight: bold;
          margin-left: 10px;
        }

        .patient-dropdown-menu {
          position: absolute;
          top: 100%;
          left: 0;
          right: 0;
          background: white;
          border: 1px solid #ddd;
          border-radius: 8px;
          box-shadow: 0 4px 12px rgba(0,0,0,0.15);
          max-height: 300px;
          overflow-y: auto;
          z-index: 1000;
          margin-top: 4px;
        }

        .loading-item {
          padding: 20px;
          text-align: center;
          color: #666;
        }

        .no-patients {
          padding: 20px;
          text-align: center;
        }

        .no-patients-text {
          color: #666;
          margin-bottom: 15px;
        }

        .create-patient-btn {
          background: #667eea;
          color: white;
          border: none;
          padding: 10px 20px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
        }

        .create-patient-btn:hover {
          background: #5a6fd8;
        }

        .patient-option {
          padding: 15px;
          border-bottom: 1px solid #f0f0f0;
          cursor: pointer;
          display: flex;
          justify-content: space-between;
          align-items: center;
          transition: background-color 0.2s ease;
        }

        .patient-option:hover {
          background-color: #f8f9fa;
        }

        .patient-option.selected {
          background-color: #e8f0fe;
          border-left: 4px solid #667eea;
        }

        .patient-option:last-child {
          border-bottom: none;
        }

        .patient-option-name {
          font-weight: 600;
          color: #333;
          margin-bottom: 4px;
        }

        .patient-option-details {
          font-size: 13px;
          color: #666;
        }

        .selected-indicator {
          color: #667eea;
          font-weight: bold;
          font-size: 18px;
        }

        .selected-patient-summary {
          margin-top: 15px;
          padding: 15px;
          background: #e8f5e8;
          border: 1px solid #c8e6c9;
          border-radius: 8px;
        }

        .summary-title {
          font-weight: 600;
          color: #2e7d32;
          margin-bottom: 8px;
        }

        .summary-content {
          color: #1b5e20;
          font-size: 14px;
        }
      `}</style>
    </div>
  );
};

export default PatientSelector;