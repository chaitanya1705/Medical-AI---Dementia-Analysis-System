import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useAuth, useNotification } from '../App';

const Profile = () => {
  const { user, updateProfile } = useAuth();
  const { showNotification } = useNotification();
  
  // Doctor's own profile data
  const [doctorProfile, setDoctorProfile] = useState({
    firstName: '',
    lastName: '',
    email: '',
    specialty: '',
    hospital: '',
    license: ''
  });

  // Current patient data being managed
  const [currentPatient, setCurrentPatient] = useState({
    firstName: '',
    lastName: '',
    email: '',
    gender: '',
    dateOfBirth: '',
    medicalCondition: '',
    patientIdNumber: '',
    notes: ''
  });

  const [activeTab, setActiveTab] = useState('doctor'); // 'doctor' or 'patient'
  const [loading, setLoading] = useState(false);
  const [existingPatients, setExistingPatients] = useState([]);

  // Load doctor data when component mounts
  useEffect(() => {
    if (user) {
      setDoctorProfile({
        firstName: user.firstName || '',
        lastName: user.lastName || '',
        email: user.email || '',
        specialty: user.specialty || '',
        hospital: user.hospital || '',
        license: user.license || ''
      });
    }
  }, []);

  // Load existing patients when patient tab is active
  useEffect(() => {
    if (activeTab === 'patient') {
      fetchExistingPatients();
    }
  }, [activeTab]);

  const fetchExistingPatients = async () => {
    try {
      const response = await axios.get('/patients', {
        params: { limit: 10 }
      });
      
      if (response.data.success) {
        setExistingPatients(response.data.data);
      }
    } catch (error) {
      console.error('Error fetching patients:', error);
    }
  };

  const handleDoctorChange = (e) => {
    setDoctorProfile({
      ...doctorProfile,
      [e.target.name]: e.target.value
    });
  };

  const handlePatientChange = (e) => {
    setCurrentPatient({
      ...currentPatient,
      [e.target.name]: e.target.value
    });
  };

  const handleDoctorSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const result = await updateProfile(doctorProfile);
    
    setLoading(false);
    
    if (result.success) {
      showNotification('Doctor profile updated successfully!', 'success');
    }
  };

  const handlePatientSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      // FIXED: Save to database via API instead of localStorage
      const patientData = {
        firstName: currentPatient.firstName,
        lastName: currentPatient.lastName,
        email: currentPatient.email,
        gender: currentPatient.gender,
        medicalCondition: currentPatient.medicalCondition,
        patientIdNumber: currentPatient.patientIdNumber || `PAT_${Date.now()}`,
        dateOfBirth: currentPatient.dateOfBirth
      };

      console.log('Saving patient to database:', patientData);

      const response = await axios.post('/patients', patientData);

      if (response.data.success) {
        showNotification(`Patient ${currentPatient.firstName} ${currentPatient.lastName} saved successfully!`, 'success');
        
        // Reset form
        setCurrentPatient({
          firstName: '',
          lastName: '',
          email: '',
          gender: '',
          dateOfBirth: '',
          medicalCondition: '',
          patientIdNumber: '',
          notes: ''
        });

        // Refresh the existing patients list
        fetchExistingPatients();

      } else {
        throw new Error(response.data.error || 'Failed to save patient');
      }

    } catch (error) {
      console.error('Error saving patient:', error);
      const errorMessage = error.response?.data?.error || error.message || 'Failed to save patient information';
      showNotification(errorMessage, 'error');
    }

    setLoading(false);
  };

  const deletePatient = async (patientId) => {
    if (!window.confirm('Are you sure you want to delete this patient?')) {
      return;
    }

    try {
      await axios.delete(`/patients/${patientId}`);
      showNotification('Patient deleted successfully', 'success');
      fetchExistingPatients(); // Refresh list
    } catch (error) {
      console.error('Error deleting patient:', error);
      showNotification('Failed to delete patient', 'error');
    }
  };

  const loadPatientData = (patient) => {
    setCurrentPatient({
      firstName: patient.first_name,
      lastName: patient.last_name,
      email: patient.email || '',
      gender: patient.gender || '',
      dateOfBirth: patient.date_of_birth || '',
      medicalCondition: patient.medical_condition || '',
      patientIdNumber: patient.patient_id_number || '',
      notes: ''
    });
    showNotification('Patient data loaded for editing', 'info');
  };

  return (
    <div className="profile-section">
      <div style={{ marginBottom: '30px' }}>
        <h2>Medical Dashboard</h2>
        <p style={{ color: '#666', marginBottom: '20px' }}>
          Manage your professional profile and patient information for AI dementia analysis
        </p>

        {/* Tab Navigation */}
        <div className="profile-tabs" style={{ display: 'flex', marginBottom: '30px', borderBottom: '2px solid #e0e0e0' }}>
          <button
            className={`tab-btn ${activeTab === 'doctor' ? 'active' : ''}`}
            onClick={() => setActiveTab('doctor')}
            style={{
              padding: '12px 24px',
              border: 'none',
              background: activeTab === 'doctor' ? '#667eea' : 'transparent',
              color: activeTab === 'doctor' ? 'white' : '#666',
              cursor: 'pointer',
              borderRadius: '8px 8px 0 0',
              fontWeight: '600'
            }}
          >
            👨‍⚕️ Your Profile
          </button>
          <button
            className={`tab-btn ${activeTab === 'patient' ? 'active' : ''}`}
            onClick={() => setActiveTab('patient')}
            style={{
              padding: '12px 24px',
              border: 'none',
              background: activeTab === 'patient' ? '#667eea' : 'transparent',
              color: activeTab === 'patient' ? 'white' : '#666',
              cursor: 'pointer',
              borderRadius: '8px 8px 0 0',
              fontWeight: '600',
              marginLeft: '5px'
            }}
          >
            👤 Patient Management 
          </button>
        </div>
      </div>

      {/* Doctor Profile Tab */}
      {activeTab === 'doctor' && (
        <form className="profile-form" onSubmit={handleDoctorSubmit}>
          <h3 style={{ marginBottom: '20px', color: '#333' }}>Doctor Profile Information</h3>
          
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="firstName">First Name:</label>
              <input
                type="text"
                id="firstName"
                name="firstName"
                value={doctorProfile.firstName}
                onChange={handleDoctorChange}
              />
            </div>
            <div className="form-group">
              <label htmlFor="lastName">Last Name:</label>
              <input
                type="text"
                id="lastName"
                name="lastName"
                value={doctorProfile.lastName}
                onChange={handleDoctorChange}
              />
            </div>
          </div>
          
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="email">Email:</label>
              <input
                type="email"
                id="email"
                name="email"
                value={doctorProfile.email}
                onChange={handleDoctorChange}
              />
            </div>
            <div className="form-group">
              <label htmlFor="specialty">Medical Specialty:</label>
              <input
                type="text"
                id="specialty"
                name="specialty"
                value={doctorProfile.specialty}
                onChange={handleDoctorChange}
              />
            </div>
          </div>
          
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="hospital">Hospital/Clinic:</label>
              <input
                type="text"
                id="hospital"
                name="hospital"
                value={doctorProfile.hospital}
                onChange={handleDoctorChange}
              />
            </div>
            <div className="form-group">
              <label htmlFor="license">Medical License:</label>
              <input
                type="text"
                id="license"
                name="license"
                value={doctorProfile.license}
                onChange={handleDoctorChange}
              />
            </div>
          </div>
          
          <button 
            type="submit" 
            className="save-btn"
            disabled={loading}
          >
            {loading ? 'Updating...' : 'Update Doctor Profile'}
          </button>
        </form>
      )}

      {/* Patient Management Tab */}
      {activeTab === 'patient' && (
        <div>
          {/* Existing Patients List */}
          {existingPatients.length > 0 && (
            <div style={{ marginBottom: '30px' }}>
              <h3 style={{ marginBottom: '15px', color: '#333' }}>Existing Patients ({existingPatients.length}) </h3>
              <div className="patients-grid">
                {existingPatients.map((patient) => (
                  <div key={patient.id} className="patient-card">
                    <div className="patient-info">
                      <h4>{patient.first_name} {patient.last_name}</h4>
                      <p>ID: {patient.patient_id_number || patient.id}</p>
                      <p>Age: {patient.date_of_birth ? new Date().getFullYear() - new Date(patient.date_of_birth).getFullYear() : 'Unknown'}</p>
                      <p>Gender: {patient.gender || 'Not specified'}</p>
                      <p>Condition: {patient.medical_condition || 'None specified'}</p>
                    </div>
                    <div className="patient-actions">
                      <button 
                        onClick={() => loadPatientData(patient)}
                        className="edit-btn"
                      >
                         Edit
                      </button>
                      <button 
                        onClick={() => deletePatient(patient.id)}
                        className="delete-btn"
                      >
                         Delete
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Patient Form */}
          <form className="profile-form" onSubmit={handlePatientSubmit}>
            <h3 style={{ marginBottom: '20px', color: '#333' }}>
              {currentPatient.firstName ? `Edit Patient: ${currentPatient.firstName} ${currentPatient.lastName}` : 'Add New Patient'}
            </h3>
            <p style={{ marginBottom: '20px', color: '#666', fontSize: '14px' }}>
              Enter patient details to save them to the database. These patients will be available for AI analysis.
            </p>
            
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="firstName">Patient First Name:</label>
                <input
                  type="text"
                  id="firstName"
                  name="firstName"
                  value={currentPatient.firstName}
                  onChange={handlePatientChange}
                  required
                />
              </div>
              <div className="form-group">
                <label htmlFor="lastName">Patient Last Name:</label>
                <input
                  type="text"
                  id="lastName"
                  name="lastName"
                  value={currentPatient.lastName}
                  onChange={handlePatientChange}
                  required
                />
              </div>
            </div>
            
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="patientIdNumber">Patient ID:</label>
                <input
                  type="text"
                  id="patientIdNumber"
                  name="patientIdNumber"
                  value={currentPatient.patientIdNumber}
                  onChange={handlePatientChange}
                  placeholder="Leave empty for auto-generation"
                />
              </div>
              <div className="form-group">
                <label htmlFor="dateOfBirth">Date of Birth:</label>
                <input
                  type="date"
                  id="dateOfBirth"
                  name="dateOfBirth"
                  value={currentPatient.dateOfBirth}
                  onChange={handlePatientChange}
                />
              </div>
            </div>
            
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="gender">Gender:</label>
                <select
                  id="gender"
                  name="gender"
                  value={currentPatient.gender}
                  onChange={handlePatientChange}
                  style={{ 
                    width: '100%',
                    padding: '14px 18px',
                    border: '2px solid #e0e0e0',
                    borderRadius: '12px',
                    fontSize: '16px'
                  }}
                >
                  <option value="">Select Gender</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
              <div className="form-group">
                <label htmlFor="email">Patient Email (Optional):</label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={currentPatient.email}
                  onChange={handlePatientChange}
                />
              </div>
            </div>
            
            <div className="form-group">
              <label htmlFor="medicalCondition">Primary Medical Condition/Symptoms:</label>
              <input
                type="text"
                id="medicalCondition"
                name="medicalCondition"
                value={currentPatient.medicalCondition}
                onChange={handlePatientChange}
                placeholder="e.g., Early Stage Dementia, Memory Loss, etc."
              />
            </div>
            
            <button 
              type="submit" 
              className="save-btn"
              disabled={loading}
              style={{ background: 'linear-gradient(135deg, #28a745, #20c997)' }}
            >
              {loading ? 'Saving...' : 'Save Patient to Database'}
            </button>

            {currentPatient.firstName && (
              <button 
                type="button"
                onClick={() => setCurrentPatient({
                  firstName: '', lastName: '', email: '', gender: '', 
                  dateOfBirth: '', medicalCondition: '', patientIdNumber: '', notes: ''
                })}
                style={{ 
                  marginLeft: '10px',
                  background: '#6c757d',
                  color: 'white',
                  border: 'none',
                  padding: '12px 20px',
                  borderRadius: '8px',
                  cursor: 'pointer'
                }}
              >
                Clear Form
              </button>
            )}

            <div style={{ marginTop: '20px', padding: '15px', background: '#e3f2fd', borderRadius: '8px', fontSize: '14px' }}>
              <strong>Next Step:</strong> After saving patients here, go to "AI Dementia Analysis" to select a patient and upload their MRI images and biomarker data for analysis.
            </div>
          </form>
        </div>
      )}

      <style jsx>{`
        .patients-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
          gap: 15px;
          margin-bottom: 30px;
        }

        .patient-card {
          background: white;
          border: 1px solid #e0e0e0;
          border-radius: 8px;
          padding: 15px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .patient-info h4 {
          margin: 0 0 10px 0;
          color: #333;
        }

        .patient-info p {
          margin: 5px 0;
          color: #666;
          font-size: 14px;
        }

        .patient-actions {
          display: flex;
          gap: 10px;
          margin-top: 15px;
        }

        .edit-btn, .delete-btn {
          padding: 8px 12px;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-size: 12px;
        }

        .edit-btn {
          background: #007bff;
          color: white;
        }

        .delete-btn {
          background: #dc3545;
          color: white;
        }

        .edit-btn:hover {
          background: #0056b3;
        }

        .delete-btn:hover {
          background: #c82333;
        }
      `}</style>
    </div>
  );
};

export default Profile;