import React from 'react';
import PDFGenerator from './PDFGenerator';

const ResultsDisplay = ({ 
  results, 
  carePlan, 
  mriImage, 
  selectedPatient,
  onGenerateCarePlan, 
  loading 
}) => {
  
  const getBiomarkerUnit = (biomarker) => {
    const units = {
      'P-Tau 181': 'pg/mL',
      'P-Tau 217': 'pg/mL',
      'P-Tau 231': 'pg/mL',
      'Amyloid Beta 42': 'pg/mL',
      'Amyloid Beta 40': 'pg/mL',
      'AB42/AB40': 'ratio'
    };
    return units[biomarker] || '';
  };

  const getMRIDisplay = () => {
    if (!mriImage) return <p>No MRI image available</p>;
    
    const imageUrl = URL.createObjectURL(mriImage);
    return (
      <img 
        src={imageUrl} 
        alt="MRI Scan" 
        style={{ 
          maxWidth: '100%', 
          borderRadius: '8px',
          boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
        }}
        onLoad={() => {
          // Clean up object URL after image loads
          setTimeout(() => URL.revokeObjectURL(imageUrl), 1000);
        }}
      />
    );
  };

  const formatPatientInfo = (patient) => {
    if (!patient) return 'Unknown Patient';
    
    const name = `${patient.first_name || ''} ${patient.last_name || ''}`.trim();
    const age = patient.date_of_birth 
      ? new Date().getFullYear() - new Date(patient.date_of_birth).getFullYear()
      : 'Unknown';
    const gender = patient.gender || 'Not specified';
    const patientId = patient.patient_id_number || patient.id || 'N/A';
    
    return {
      name: name || 'Unknown Patient',
      age,
      gender,
      patientId,
      condition: patient.medical_condition || 'Dementia Analysis'
    };
  };

  const patientInfo = formatPatientInfo(selectedPatient);

  return (
    <div className="results-container">
      {/* Patient Info Header */}
      {selectedPatient && (
        <div className="patient-info-header">
          <div className="patient-avatar">
            {patientInfo.name.split(' ').map(n => n[0]).join('').toUpperCase()}
          </div>
          <div className="patient-details">
            <h3>{patientInfo.name}</h3>
            <div className="patient-meta">
              Age: {patientInfo.age} • Gender: {patientInfo.gender} • ID: {patientInfo.patientId}
            </div>
            <div className="analysis-date">
              Analysis Date: {new Date().toLocaleDateString()}
            </div>
          </div>
        </div>
      )}

      <div className="results-grid">
        {/* Predicted Stage */}
        <div className="result-card predicted-stage">
          <div>Predicted STAGE</div>
          <div style={{ fontSize: '36px', marginTop: '10px' }}>
            {results.predicted_stage}
          </div>
          <div style={{ fontSize: '16px', marginTop: '10px', opacity: 0.9 }}>
            Confidence: {(results.confidence * 100).toFixed(1)}%
          </div>
        </div>
        
        {/* MRI Image */}
        <div className="result-card">
          <h3>MRI Image</h3>
          <div className="mri-display">
            {getMRIDisplay()}
          </div>
        </div>

        {/* Biomarkers Values */}
        <div className="result-card">
          <h3>Biomarkers Values</h3>
          <table className="biomarkers-table">
            <thead>
              <tr>
                <th>Biomarker</th>
                <th>Value</th>
                <th>Unit</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(results.biomarkers).map(([biomarker, value]) => (
                <tr key={biomarker}>
                  <td>{biomarker}</td>
                  <td>{value.toFixed(3)}</td>
                  <td>{getBiomarkerUnit(biomarker)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Progression Time */}
        <div className="result-card progression-time">
          Estimated Progression Time: {results.progression_months.toFixed(1)} months to next stage
        </div>

        {/* Plan of Care */}
        {carePlan && (
          <div className="result-card plan-of-care">
            <h3>Plan of Care for {patientInfo.name}</h3>
            <ul className="care-plan-list">
              {carePlan.map((item, index) => (
                <li key={index}>{item}</li>
              ))}
            </ul>
            <div className="pdf-generation-section">
              <PDFGenerator 
                results={results}
                carePlan={carePlan}
                mriImage={mriImage}
                selectedPatient={selectedPatient}
              />
              <span className="pdf-note">
                Generate a comprehensive medical report for {patientInfo.name}
              </span>
            </div>
          </div>
        )}
      </div>
      
      {/* Generate Care Plan Button */}
      {!carePlan && (
        <div style={{ textAlign: 'center' }}>
          <button 
            className="generate-plan-btn"
            onClick={onGenerateCarePlan}
            disabled={loading}
          >
            {loading ? 'Generating AI Care Plan...' : `Generate Care Plan for ${patientInfo.name}`}
          </button>
        </div>
      )}

      <style>{`
        .results-container {
          max-width: 1200px;
          margin: 0 auto;
          padding: 20px;
        }

        .patient-info-header {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 20px;
          border-radius: 12px;
          margin-bottom: 25px;
          display: flex;
          align-items: center;
          box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        }

        .patient-avatar {
          width: 60px;
          height: 60px;
          border-radius: 50%;
          background: rgba(255, 255, 255, 0.2);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 24px;
          font-weight: bold;
          margin-right: 20px;
          border: 2px solid rgba(255, 255, 255, 0.3);
        }

        .patient-details h3 {
          margin: 0 0 8px 0;
          font-size: 24px;
          font-weight: 600;
        }

        .patient-meta {
          font-size: 16px;
          opacity: 0.9;
          margin-bottom: 4px;
        }

        .analysis-date {
          font-size: 14px;
          opacity: 0.8;
        }

        .results-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 20px;
          margin-bottom: 30px;
        }

        .result-card {
          background: white;
          border: 1px solid #e0e0e0;
          border-radius: 12px;
          padding: 20px;
          box-shadow: 0 2px 10px rgba(0,0,0,0.1);
          transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .result-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }

        .result-card h3 {
          margin: 0 0 15px 0;
          color: #333;
          font-size: 18px;
          font-weight: 600;
        }

        .predicted-stage {
          background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
          color: white;
          text-align: center;
          font-weight: bold;
        }

        .predicted-stage:hover {
          transform: translateY(-2px);
        }

        .progression-time {
          background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
          color: white;
          text-align: center;
          font-weight: bold;
          font-size: 18px;
          padding: 30px 20px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .plan-of-care {
          grid-column: 1 / -1;
          background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
          border: 2px solid #dee2e6;
        }

        .biomarkers-table {
          width: 100%;
          border-collapse: collapse;
          margin-top: 10px;
        }

        .biomarkers-table th,
        .biomarkers-table td {
          padding: 12px;
          text-align: left;
          border-bottom: 1px solid #e0e0e0;
        }

        .biomarkers-table th {
          background: #f8f9fa;
          font-weight: 600;
          color: #333;
        }

        .biomarkers-table tr:hover {
          background: #f8f9fa;
        }

        .care-plan-list {
          list-style: none;
          padding: 0;
          margin: 15px 0;
        }

        .care-plan-list li {
          padding: 15px 0;
          border-bottom: 1px solid #f0f0f0;
          position: relative;
          padding-left: 30px;
          font-size: 14px;
          line-height: 1.6;
        }

        .care-plan-list li:before {
          content: "✓";
          color: #28a745;
          font-weight: bold;
          position: absolute;
          left: 0;
          top: 15px;
          font-size: 16px;
        }

        .care-plan-list li:last-child {
          border-bottom: none;
        }

        .pdf-generation-section {
          display: flex;
          align-items: center;
          justify-content: center;
          margin-top: 25px;
          padding-top: 25px;
          border-top: 2px solid #e0e0e0;
        }

        .pdf-note {
          margin-left: 15px;
          color: #666;
          font-size: 14px;
          font-style: italic;
        }

        .generate-plan-btn {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border: none;
          padding: 15px 30px;
          border-radius: 8px;
          font-size: 16px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.3s ease;
          box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        }

        .generate-plan-btn:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        }

        .generate-plan-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
          transform: none;
        }

        .mri-display {
          text-align: center;
          margin-top: 10px;
        }

        .mri-display img {
          max-height: 200px;
          object-fit: contain;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
          .results-grid {
            grid-template-columns: 1fr;
          }
          
          .patient-info-header {
            flex-direction: column;
            text-align: center;
          }
          
          .patient-avatar {
            margin-right: 0;
            margin-bottom: 15px;
          }
          
          .pdf-generation-section {
            flex-direction: column;
            gap: 15px;
          }
          
          .pdf-note {
            margin-left: 0;
            text-align: center;
          }
        }

        /* Animation for loading states */
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.5; }
          100% { opacity: 1; }
        }

        .generate-plan-btn:disabled {
          animation: pulse 2s infinite;
        }
      `}</style>
    </div>
  );
};

export default ResultsDisplay;