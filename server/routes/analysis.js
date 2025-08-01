import express from 'express';
import multer from 'multer';
import axios from 'axios';
import FormData from 'form-data';
import { authenticateToken } from './auth.js';
import { dbHelpers } from '../database/init.js';

const router = express.Router();

// Configure multer for file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB limit
  },
  fileFilter: (req, file, cb) => {
    if (file.fieldname === 'mri_image') {
      if (file.mimetype.startsWith('image/')) {
        cb(null, true);
      } else {
        cb(new Error('Only image files are allowed for MRI'), false);
      }
    } else if (file.fieldname === 'biomarker_data') {
      const allowedTypes = [
        'text/csv',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
      ];
      if (allowedTypes.includes(file.mimetype)) {
        cb(null, true);
      } else {
        cb(new Error('Only CSV and Excel files are allowed for biomarkers'), false);
      }
    } else {
      cb(new Error('Unknown field'), false);
    }
  }
});

// Analyze patient data endpoint - Fixed version
router.post('/analyze', authenticateToken, upload.fields([
  { name: 'mri_image', maxCount: 1 },
  { name: 'biomarker_data', maxCount: 1 }
]), async (req, res) => {
  try {
    const mriFile = req.files['mri_image']?.[0];
    const biomarkerFile = req.files['biomarker_data']?.[0];
    const requestedPatientId = req.body.patient_id; // Get patient ID from form data

    if (!mriFile || !biomarkerFile) {
      return res.status(400).json({ 
        error: 'Both MRI image and biomarker data are required' 
      });
    }

    console.log('📁 Files received:', {
      mri: mriFile.originalname,
      biomarker: biomarkerFile.originalname,
      requestedPatientId: requestedPatientId
    });

    // Determine which patient to use
    let patientId = requestedPatientId;
    
    if (requestedPatientId) {
      // Verify the requested patient exists and belongs to this user
      const existingPatient = await dbHelpers.get(
        'SELECT id FROM patients WHERE id = ? AND user_id = ?',
        [requestedPatientId, req.user.id]
      );
      
      if (!existingPatient) {
        return res.status(400).json({ 
          error: 'Invalid patient ID or patient does not belong to your account' 
        });
      }
      
      console.log('👤 Using requested patient ID:', requestedPatientId);
    } else {
      // Create or get default patient if no specific patient requested
      try {
        let defaultPatient = await dbHelpers.get(
          'SELECT id FROM patients WHERE user_id = ? AND patient_id_number = ?',
          [req.user.id, 'DEFAULT']
        );

        if (!defaultPatient) {
          // Create default patient if doesn't exist
          const newPatient = await dbHelpers.run(
            `INSERT INTO patients (user_id, first_name, last_name, patient_id_number, medical_condition)
             VALUES (?, ?, ?, ?, ?)`,
            [req.user.id, 'Default', 'Patient', 'DEFAULT', 'Dementia Analysis']
          );
          patientId = newPatient.id;
          console.log('Created default patient with ID:', patientId);
        } else {
          patientId = defaultPatient.id;
          console.log('Using existing default patient ID:', patientId);
        }
      } catch (patientError) {
        console.error('Patient creation error:', patientError);
        
        // Emergency fallback: Create patient with unique timestamp
        try {
          const fallbackPatient = await dbHelpers.run(
            `INSERT INTO patients (user_id, first_name, last_name, patient_id_number, medical_condition)
             VALUES (?, ?, ?, ?, ?)`,
            [req.user.id, 'Emergency', 'Patient', `TEMP_${Date.now()}`, 'Dementia Analysis']
          );
          patientId = fallbackPatient.id;
          console.log('Created emergency fallback patient with ID:', patientId);
        } catch (fallbackError) {
          console.error('Patient creation failed:', fallbackError);
          return res.status(500).json({ 
            error: 'Unable to create patient record for analysis. Please create a patient first.' 
          });
        }
      }
    }

    // Forward request to Python AI backend
    const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
    
    console.log('Forwarding to Python backend:', `${pythonBackendUrl}/api/analyze`);

    try {
      // Create FormData for Python backend
      const formData = new FormData();
      formData.append('mri_image', mriFile.buffer, {
        filename: mriFile.originalname,
        contentType: mriFile.mimetype
      });
      formData.append('biomarker_data', biomarkerFile.buffer, {
        filename: biomarkerFile.originalname,
        contentType: biomarkerFile.mimetype
      });

      console.log('Sending request to Python AI backend...');

      const response = await axios.post(`${pythonBackendUrl}/api/analyze`, formData, {
        headers: {
          ...formData.getHeaders(),
        },
        timeout: 300000, // 5 minute timeout for AI processing
        maxContentLength: Infinity,
        maxBodyLength: Infinity
      });

      console.log('Python backend response received');

      if (response.data.success) {
        const results = response.data.results;

        console.log('AI Analysis Results:', {
          stage: results.predicted_stage,
          confidence: results.confidence,
          progression: results.progression_months
        });

        // Store analysis results in database
        try {
          await dbHelpers.beginTransaction();

          const analysisResult = await dbHelpers.run(
            `INSERT INTO analysis_results (
              patient_id, user_id, predicted_stage, stage_index, confidence,
              progression_months, prediction_method, mri_image_data, mri_image_type,
              biomarkers_data, enhanced_features_count, all_probabilities
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
            [
              patientId, // Use the determined patient ID
              req.user.id,
              results.predicted_stage,
              results.stage_index,
              results.confidence,
              results.progression_months,
              results.prediction_method,
              mriFile.buffer,
              mriFile.mimetype,
              JSON.stringify(results.biomarkers),
              results.enhanced_features_count,
              JSON.stringify(results.all_probabilities)
            ]
          );

          // Insert individual biomarkers
          for (const [biomarker, value] of Object.entries(results.biomarkers)) {
            await dbHelpers.run(
              `INSERT INTO biomarkers (analysis_id, biomarker_name, biomarker_value, biomarker_unit)
               VALUES (?, ?, ?, ?)`,
              [
                analysisResult.id,
                biomarker,
                value,
                getBiomarkerUnit(biomarker)
              ]
            );
          }

          await dbHelpers.commit();
          console.log('Analysis results saved to database');

          // Return results with database ID and patient ID
          res.json({
            success: true,
            results: {
              ...results,
              analysis_id: analysisResult.id,
              patient_id: patientId
            }
          });

        } catch (dbError) {
          await dbHelpers.rollback();
          console.error('Database error:', dbError);
          
          // Still return the analysis results even if DB save fails
          res.json({
            success: true,
            results: {
              ...results,
              patient_id: patientId
            },
            warning: 'Analysis completed but database save failed - ' + dbError.message
          });
        }

      } else {
        throw new Error(response.data.error || 'Python backend analysis failed');
      }

    } catch (pythonError) {
      console.error('Python backend error:', pythonError.message);
      
      if (pythonError.code === 'ECONNREFUSED') {
        console.log('Python backend not running, using mock data...');
        
        // Enhanced mock results that look realistic
        const mockResults = {
          predicted_stage: 'VMD (Very Mild Dementia)',
          stage_index: 1,
          confidence: 0.87,
          progression_months: 14.2,
          prediction_method: 'Mock Analysis (Python AI backend not available)',
          biomarkers: {
            'P-Tau 181': 2.34,
            'P-Tau 217': 1.67,
            'P-Tau 231': 3.45,
            'Amyloid Beta 42': 423.7,
            'Amyloid Beta 40': 4876.3,
            'AB42/AB40': 0.087
          },
          all_probabilities: {
            'ND (No Dementia)': 0.03,
            'VMD (Very Mild Dementia)': 0.87,
            'MD (Mild Dementia)': 0.08,
            'MOD (Moderate Dementia)': 0.02
          },
          enhanced_features_count: 16
        };

        // Save mock results to database too
        try {
          const analysisResult = await dbHelpers.run(
            `INSERT INTO analysis_results (
              patient_id, user_id, predicted_stage, stage_index, confidence,
              progression_months, prediction_method, mri_image_data, mri_image_type,
              biomarkers_data, enhanced_features_count, all_probabilities
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
            [
              patientId, req.user.id, mockResults.predicted_stage, mockResults.stage_index,
              mockResults.confidence, mockResults.progression_months, mockResults.prediction_method,
              mriFile.buffer, mriFile.mimetype, JSON.stringify(mockResults.biomarkers),
              mockResults.enhanced_features_count, JSON.stringify(mockResults.all_probabilities)
            ]
          );

          mockResults.analysis_id = analysisResult.id;
          mockResults.patient_id = patientId;
          console.log(' Mock analysis results saved to database');
        } catch (dbError) {
          console.warn('Mock data DB save failed:', dbError.message);
          mockResults.patient_id = patientId;
        }

        return res.json({
          success: true,
          results: mockResults,
          warning: 'Python AI backend not available - using mock results. Please start Python backend on port 5001.'
        });
      }
      
      if (pythonError.response) {
        return res.status(pythonError.response.status).json({ 
          error: pythonError.response.data?.error || 'Python backend error',
          details: pythonError.response.data
        });
      }
      
      throw pythonError;
    }

  } catch (error) {
    console.error('Analysis error:', error);
    res.status(500).json({ 
      error: error.message || 'Internal server error',
      details: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  }
});

// Generate care plan - FIXED VERSION with proper patient ID
router.post('/generate_care_plan', authenticateToken, async (req, res) => {
  try {
    const { stage, biomarkers, confidence, analysisId, patientId } = req.body;

    if (!stage) {
      return res.status(400).json({ error: 'Stage is required for care plan generation' });
    }

    console.log('🔄 Generating care plan for stage:', stage, 'patientId:', patientId);

    const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';

    try {
      const response = await axios.post(`${pythonBackendUrl}/api/generate_care_plan`, {
        stage,
        biomarkers: biomarkers || {},
        confidence: confidence || 0
      }, {
        timeout: 30000 // 30 second timeout
      });

      if (response.data.success) {
        const carePlan = response.data.care_plan;
        const planSource = response.data.source || 'AI-Generated';

        console.log(`Care plan generated: ${planSource} (${carePlan.length} points)`);

        // Save care plan to database with CORRECT patient ID
        if (analysisId || patientId) {
          try {
            await dbHelpers.run(
              `INSERT INTO care_plans 
               (analysis_id, patient_id, user_id, care_plan_points, plan_source, stage)
               VALUES (?, ?, ?, ?, ?, ?)`,
              [
                analysisId || null,
                patientId || null, // Use the actual patient ID, not hardcoded 1
                req.user.id,
                JSON.stringify(carePlan),
                planSource,
                stage
              ]
            );
            console.log('Care plan saved to database with patient ID:', patientId);
          } catch (dbError) {
            console.warn('Care plan DB save failed:', dbError.message);
          }
        }

        res.json({
          success: true,
          care_plan: carePlan,
          source: planSource,
          stage: stage,
          recommendations_count: carePlan.length
        });

      } else {
        throw new Error(response.data.error || 'Care plan generation failed');
      }

    } catch (pythonError) {
      console.warn('Python backend care plan failed, using fallback');
      
      // Use fallback care plan
      const fallbackPlan = getFallbackCarePlan(stage);
      
      // Save fallback plan to database too
      if (analysisId || patientId) {
        try {
          await dbHelpers.run(
            `INSERT INTO care_plans 
             (analysis_id, patient_id, user_id, care_plan_points, plan_source, stage)
             VALUES (?, ?, ?, ?, ?, ?)`,
            [
              analysisId || null,
              patientId || null, // Use the actual patient ID
              req.user.id,
              JSON.stringify(fallbackPlan),
              'Fallback',
              stage
            ]
          );
          console.log('Fallback care plan saved to database');
        } catch (dbError) {
          console.warn('Fallback care plan DB save failed:', dbError.message);
        }
      }
      
      res.json({
        success: true,
        care_plan: fallbackPlan,
        source: 'Fallback',
        stage: stage,
        recommendations_count: fallbackPlan.length,
        warning: 'Python backend not available - using clinical fallback care plan'
      });
    }

  } catch (error) {
    console.error('Care plan generation error:', error);
    res.status(500).json({ error: 'Failed to generate care plan' });
  }
});

// Check Python backend health
router.get('/health/python', async (req, res) => {
  try {
    const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
    const response = await axios.get(`${pythonBackendUrl}/api/health`, {
      timeout: 5000
    });

    res.json({
      success: true,
      python_backend: {
        status: 'connected',
        url: pythonBackendUrl,
        ...response.data
      }
    });

  } catch (error) {
    res.status(503).json({
      success: false,
      error: 'Python AI backend not available',
      details: error.message,
      python_backend: {
        status: 'disconnected',
        url: process.env.PYTHON_BACKEND_URL || 'http://localhost:5001'
      }
    });
  }
});

// Get analysis history
router.get('/history', authenticateToken, async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;
    const offset = (page - 1) * limit;

    const analyses = await dbHelpers.all(
      `SELECT 
        ar.id, ar.predicted_stage, ar.stage_index, ar.confidence,
        ar.progression_months, ar.prediction_method, ar.created_at,
        COUNT(b.id) as biomarker_count,
        p.first_name, p.last_name
       FROM analysis_results ar
       LEFT JOIN biomarkers b ON ar.id = b.analysis_id
       LEFT JOIN patients p ON ar.patient_id = p.id
       WHERE ar.user_id = ?
       GROUP BY ar.id
       ORDER BY ar.created_at DESC
       LIMIT ? OFFSET ?`,
      [req.user.id, limit, offset]
    );

    const total = await dbHelpers.get(
      'SELECT COUNT(*) as count FROM analysis_results WHERE user_id = ?',
      [req.user.id]
    );

    res.json({
      success: true,
      data: analyses,
      pagination: {
        page,
        limit,
        total: total.count,
        pages: Math.ceil(total.count / limit)
      }
    });

  } catch (error) {
    console.error('History fetch error:', error);
    res.status(500).json({ error: 'Failed to fetch analysis history' });
  }
});

// Helper function to get biomarker units
function getBiomarkerUnit(biomarker) {
  const units = {
    'P-Tau 181': 'pg/mL',
    'P-Tau 217': 'pg/mL',
    'P-Tau 231': 'pg/mL',
    'Amyloid Beta 42': 'pg/mL',
    'Amyloid Beta 40': 'pg/mL',
    'AB42/AB40': 'ratio'
  };
  return units[biomarker] || '';
}

// Fallback care plan function
function getFallbackCarePlan(stage) {
  const fallbackPlans = {
    'ND (No Dementia)': [
      "Maintain regular aerobic exercise routine 30-45 minutes daily to enhance neuroplasticity",
      "Follow Mediterranean diet with weekly fish consumption for brain health",
      "Engage in challenging cognitive activities 3x weekly including puzzles and reading",
      "Maintain consistent sleep schedule 7-8 hours nightly for amyloid clearance",
      "Schedule annual comprehensive cognitive evaluations and biomarker monitoring"
    ],
    'VMD (Very Mild Dementia)': [
      "Establish structured daily routines with visual cues and calendars for memory support",
      "Begin supervised cognitive rehabilitation program focusing on memory strategies",
      "Implement home safety modifications including improved lighting and grab bars",
      "Participate in social engagement activities 2-3x weekly for cognitive stimulation",
      "Coordinate quarterly medical monitoring with neurologist for progression tracking"
    ],
    'MD (Mild Dementia)': [
      "Provide structured supervision for complex activities while maintaining basic independence",
      "Implement comprehensive safety protocols including medical alert systems",
      "Establish consistent caregiver routines with clear communication strategies",
      "Engage in simplified cognitive and physical activities tailored to abilities",
      "Coordinate multidisciplinary care team including neurology and occupational therapy"
    ],
    'MOD (Moderate Dementia)': [
      "Ensure continuous supervision and assistance with all daily living activities",
      "Create calming environment with familiar objects and consistent routines",
      "Implement person-centered care approaches using validation and music therapy",
      "Establish comprehensive safety measures including wandering prevention systems",
      "Provide extensive caregiver support including respite care and support groups"
    ]
  };

  return fallbackPlans[stage] || fallbackPlans['VMD (Very Mild Dementia)'];
}

export default router;