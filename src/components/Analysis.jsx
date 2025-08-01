import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNotification } from '../App';
import FileUpload from './FileUpload';
import ResultsDisplay from './ResultsDisplay';
import LoadingSpinner from './LoadingSpinner';
import PatientSelector from './PatientSelector';

const Analysis = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [uploadedMRI, setUploadedMRI] = useState(null);
  const [uploadedBiomarkers, setUploadedBiomarkers] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [carePlan, setCarePlan] = useState(null);
  const [loading, setLoading] = useState(false);
  const [backendConnected, setBackendConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState({
    connected: false,
    modelLoaded: false,
    tokenizerLoaded: false,
    geminiConfigured: false
  });

  const { showNotification } = useNotification();

  // Check backend connection on component mount
  useEffect(() => {
    checkBackendConnection();
  }, []);

  const checkBackendConnection = async () => {
    try {
      console.log('Checking backend connection...');
      
      // FIXED: Remove /api/ prefix since it's already in axios.defaults.baseURL
      const healthResponse = await axios.get('/health');
      console.log('Node.js backend health:', healthResponse.data);
      
      if (healthResponse.data.status === 'healthy') {
        setBackendConnected(true);
        setConnectionStatus({
          connected: true,
          modelLoaded: healthResponse.data.vlm_model_loaded || false,
          tokenizerLoaded: healthResponse.data.tokenizer_loaded || false,
          geminiConfigured: healthResponse.data.gemini_configured || false
        });
        
        // Try to check Python backend
        try {
          const pythonResponse = await axios.get('/analysis/health/python');
          console.log('Python backend status:', pythonResponse.data);
        } catch (pythonError) {
          console.log('Python backend not available:', pythonError.message);
        }
      }
    } catch (error) {
      console.error('Backend connection failed:', error);
      setBackendConnected(false);
      setConnectionStatus({
        connected: false,
        modelLoaded: false,
        tokenizerLoaded: false,
        geminiConfigured: false
      });
    }
  };

  const handlePatientSelect = (patient) => {
    setSelectedPatient(patient);
    console.log('Selected patient:', patient);
  };

  const handleMRIUpload = (file) => {
    console.log('MRI file uploaded:', file.name);
    setUploadedMRI(file);
    showNotification(`MRI uploaded successfully: ${file.name}`, 'success');
  };

  const handleBiomarkerUpload = (file) => {
    console.log('Biomarker file uploaded:', file.name);
    setUploadedBiomarkers(file);
    showNotification(`Biomarker data uploaded: ${file.name}`, 'success');
  };

  const canAnalyze = () => {
    return selectedPatient && uploadedMRI && uploadedBiomarkers && backendConnected;
  };

  const performAnalysis = async () => {
    if (!canAnalyze()) {
      if (!selectedPatient) {
        showNotification('Please select a patient first', 'error');
      } else if (!uploadedMRI || !uploadedBiomarkers) {
        showNotification('Please upload both MRI image and biomarker data', 'error');
      } else {
        showNotification('Backend not connected', 'error');
      }
      return;
    }

    console.log('Starting analysis for patient:', selectedPatient.first_name, selectedPatient.last_name);
    setLoading(true);
    
    try {
      const formData = new FormData();
      formData.append('mri_image', uploadedMRI);
      formData.append('biomarker_data', uploadedBiomarkers);
      formData.append('patient_id', selectedPatient.id); // Add patient ID to the request
      
      console.log('Sending analysis request to /analysis/analyze');
      
      // FIXED: Remove /api/ prefix since it's already in axios.defaults.baseURL
      const response = await axios.post('/analysis/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 300000, // 5 minute timeout
      });
      
      console.log('Analysis response:', response.data);
      
      if (response.data.success) {
        // Add patient info to results for PDF generation
        const resultsWithPatient = {
          ...response.data.results,
          patient: selectedPatient
        };
        
        setAnalysisResults(resultsWithPatient);
        setActiveTab('results');
        showNotification(
          `Analysis completed for ${selectedPatient.first_name} ${selectedPatient.last_name}!`, 
          'success'
        );
        
        if (response.data.warning) {
          showNotification(response.data.warning, 'warning');
        }
      } else {
        throw new Error(response.data.error || 'Analysis failed');
      }
      
    } catch (error) {
      console.error('Analysis error:', error);
      
      let errorMessage = 'Analysis failed';
      
      if (error.response) {
        // Server responded with error status
        errorMessage = error.response.data?.error || `Server error: ${error.response.status}`;
        console.log('Server error response:', error.response.data);
      } else if (error.request) {
        // Request was made but no response received
        errorMessage = 'No response from server. Check if backend is running.';
        console.log('No response received:', error.request);
      } else {
        // Something else happened
        errorMessage = error.message;
        console.log('Request error:', error.message);
      }
      
      showNotification(`Analysis failed: ${errorMessage}`, 'error');
      
      // For development, show mock results if backend fails
      if (error.response?.status === 404 || error.code === 'ECONNREFUSED') {
        console.log('Using mock results for development');
        const mockResults = {
          predicted_stage: 'VMD (Very Mild Dementia)',
          stage_index: 1,
          confidence: 0.85,
          progression_months: 12.5,
          prediction_method: 'Mock Analysis (Backend unavailable)',
          biomarkers: {
            'P-Tau 181': 2.1,
            'P-Tau 217': 1.5,
            'P-Tau 231': 3.2,
            'Amyloid Beta 42': 450.0,
            'Amyloid Beta 40': 5200.0,
            'AB42/AB40': 0.087
          },
          all_probabilities: {
            'ND (No Dementia)': 0.05,
            'VMD (Very Mild Dementia)': 0.85,
            'MD (Mild Dementia)': 0.08,
            'MOD (Moderate Dementia)': 0.02
          },
          enhanced_features_count: 16,
          patient: selectedPatient // Add patient info to mock results too
        };
        
        setAnalysisResults(mockResults);
        setActiveTab('results');
        showNotification('Using mock results for development', 'warning');
      }
      
    } finally {
      setLoading(false);
    }
  };

  const generateCarePlan = async () => {
    if (!analysisResults) {
      showNotification('No analysis results available', 'error');
      return;
    }

    console.log('Generating care plan for patient:', selectedPatient?.first_name, selectedPatient?.last_name);
    setLoading(true);
    
    try {
      // FIXED: Remove /api/ prefix since it's already in axios.defaults.baseURL
      const response = await axios.post('/care-plans/generate', {
        stage: analysisResults.predicted_stage,
        biomarkers: analysisResults.biomarkers,
        confidence: analysisResults.confidence,
        analysisId: analysisResults.analysis_id,
        patientId: selectedPatient?.id
      });
      
      console.log('Care plan response:', response.data);
      
      if (response.data.success) {
        setCarePlan(response.data.care_plan);
        const source = response.data.source === 'AI-Generated' ? 'AI-powered' : 'Clinical';
        showNotification(`${source} care plan generated for ${selectedPatient?.first_name}!`, 'success');
      } else {
        throw new Error(response.data.error || 'Care plan generation failed');
      }
      
    } catch (error) {
      console.error('Care plan generation error:', error);
      
      // Use fallback care plan
      const fallbackPlan = getFallbackCarePlan(analysisResults.predicted_stage);
      setCarePlan(fallbackPlan);
      
      const message = error.response?.data?.error || error.message || 'Care plan generation failed';
      showNotification(`Using fallback care plan: ${message}`, 'warning');
    } finally {
      setLoading(false);
    }
  };

  const resetAnalysis = () => {
    setActiveTab('upload');
    setUploadedMRI(null);
    setUploadedBiomarkers(null);
    setAnalysisResults(null);
    setCarePlan(null);
    // Keep selected patient for convenience
  };

  const getConnectionStatusDisplay = () => {
    const { connected, modelLoaded, tokenizerLoaded, geminiConfigured } = connectionStatus;
    
    if (connected) {
      return {
        className: 'status-success',
        message: `Node.js Backend Connected - Analysis Available`
      };
    } else {
      return {
        className: 'status-error',
        message: 'Backend Disconnected - Please start the Node.js server'
      };
    }
  };

  const getFallbackCarePlan = (stage) => {
    const fallbackPlans = {
      'ND (No Dementia)': [
        "Maintain regular aerobic exercise routine 30-45 minutes daily",
        "Follow Mediterranean diet with weekly fish consumption",
        "Engage in challenging cognitive activities 3x weekly",
        "Maintain consistent sleep schedule 7-8 hours nightly",
        "Schedule annual comprehensive cognitive evaluations"
      ],
      'VMD (Very Mild Dementia)': [
        "Establish structured daily routines with visual cues",
        "Begin supervised cognitive rehabilitation program",
        "Implement home safety modifications",
        "Participate in social engagement activities 2-3x weekly",
        "Coordinate quarterly medical monitoring"
      ],
      'MD (Mild Dementia)': [
        "Provide structured supervision for complex activities",
        "Implement comprehensive safety protocols",
        "Establish consistent caregiver routines",
        "Engage in simplified cognitive and physical activities",
        "Coordinate multidisciplinary care team"
      ],
      'MOD (Moderate Dementia)': [
        "Ensure continuous supervision and assistance",
        "Create calming environment with familiar objects",
        "Implement person-centered care approaches",
        "Establish comprehensive safety measures",
        "Provide extensive caregiver support"
      ]
    };
    
    return fallbackPlans[stage] || fallbackPlans['VMD (Very Mild Dementia)'];
  };

  const statusDisplay = getConnectionStatusDisplay();

  return (
    <div className="analysis-section">
      {/* Analysis Tab Navigation */}
      <div className="analysis-tabs">
        <div 
          className={`tab-button ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
           Upload Data
        </div>
        {analysisResults && (
          <div 
            className={`tab-button ${activeTab === 'results' ? 'active' : ''}`}
            onClick={() => setActiveTab('results')}
          >
             Analysis Results
          </div>
        )}
      </div>

      {/* Upload Tab Content */}
      {activeTab === 'upload' && (
        <div className="tab-content active">
          <h2>🧠 AI Dementia Analysis</h2>
          <p style={{ marginBottom: '30px', color: '#666' }}>
            Select a patient and upload their MRI scan and biomarker data to get AI-powered dementia stage prediction and care recommendations.
          </p>
          
          {/* Patient Selection */}
          <PatientSelector 
            selectedPatient={selectedPatient}
            onPatientSelect={handlePatientSelect}
          />
          
          <FileUpload
            type="mri"
            accept=".jpg,.jpeg,.png"
            onUpload={handleMRIUpload}
            uploadedFile={uploadedMRI}
            title="Upload Patient MRI Scan"
            description="Drag and drop the patient's MRI image here or click to browse"
          />

          <FileUpload
            type="biomarker"
            accept=".csv,.xlsx"
            onUpload={handleBiomarkerUpload}
            uploadedFile={uploadedBiomarkers}
            title="Upload Biomarker Data"
            description="Upload Excel (.xlsx) or CSV file with patient biomarker values"
          />

          {/* Backend Connection Status */}
          <div className={`connection-status ${statusDisplay.className}`}>
            <div className="status-indicator">
              {statusDisplay.message}
            </div>
          </div>

          <div style={{ textAlign: 'center' }}>
            <button 
              className="analyze-btn"
              onClick={performAnalysis}
              disabled={!canAnalyze() || loading}
            >
              {loading ? ' Analyzing Patient Data...' : 
               !selectedPatient ? '👤 Select Patient First' :
               !backendConnected ? '❌ Backend Not Connected' : 
               !uploadedMRI ? '📁 Upload MRI Image First' :
               !uploadedBiomarkers ? '📁 Upload Biomarker Data First' :
               ` Analyze ${selectedPatient.first_name}'s Data`}
            </button>
          </div>

          {loading && <LoadingSpinner message={`Analyzing ${selectedPatient?.first_name}'s data...`} />}
        </div>
      )}

      {/* Results Tab Content */}
      {activeTab === 'results' && analysisResults && (
        <div className="tab-content active">
          <div className="results-header">
            <button className="back-btn" onClick={() => setActiveTab('upload')}>
              ← Back to Upload
            </button>
            <h2> Analysis Results - {selectedPatient?.first_name} {selectedPatient?.last_name}</h2>
            <button className="reset-btn" onClick={resetAnalysis}>
               New Analysis
            </button>
          </div>

          <ResultsDisplay
            results={analysisResults}
            carePlan={carePlan}
            mriImage={uploadedMRI}
            selectedPatient={selectedPatient}
            onGenerateCarePlan={generateCarePlan}
            loading={loading}
          />
        </div>
      )}
    </div>
  );
};

export default Analysis;