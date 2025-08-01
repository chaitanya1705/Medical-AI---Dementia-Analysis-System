import express from 'express';
import axios from 'axios';
import { authenticateToken } from './auth.js';
import { dbHelpers } from '../database/init.js';

const router = express.Router();

// Generate care plan
router.post('/generate', authenticateToken, async (req, res) => {
  try {
    const { stage, biomarkers, confidence, analysisId, patientId } = req.body;

    if (!stage) {
      return res.status(400).json({ error: 'Stage is required for care plan generation' });
    }

    let carePlan = [];
    let planSource = 'Fallback';

    // Try to generate care plan using Python backend first
    try {
      // FIXED: Use the correct Python backend URL (port 5001)
      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:5001';
      
      console.log(`Attempting to generate care plan via Python backend: ${pythonBackendUrl}/api/generate_care_plan`);
      
      const response = await axios.post(`${pythonBackendUrl}/api/generate_care_plan`, {
        stage,
        biomarkers: biomarkers || {},
        confidence: confidence || 0
      }, {
        timeout: 30000, // 30 second timeout
        headers: {
          'Content-Type': 'application/json'
        }
      });

      console.log('Python backend care plan response:', response.data);

      if (response.data.success) {
        carePlan = response.data.care_plan;
        planSource = response.data.source || 'AI-Generated';
        console.log(`Successfully generated ${planSource} care plan with ${carePlan.length} recommendations`);
      } else {
        throw new Error(response.data.error || 'Python backend care plan generation failed');
      }

    } catch (error) {
      console.warn('Python backend care plan generation failed:', error.message);
      
      // Log more details about the error
      if (error.response) {
        console.log('Error status:', error.response.status);
        console.log('Error data:', error.response.data);
      }
      
      // Use fallback care plan
      carePlan = getFallbackCarePlan(stage);
      planSource = 'Fallback';
      console.log(`Using fallback care plan for stage: ${stage}`);
    }

    // Validate care plan
    if (!Array.isArray(carePlan) || carePlan.length === 0) {
      console.warn('Invalid care plan received, using fallback');
      carePlan = getFallbackCarePlan(stage);
      planSource = 'Fallback';
    }

    // Ensure exactly 5 recommendations
    while (carePlan.length < 5) {
      const fallback = getFallbackCarePlan(stage);
      carePlan.push(fallback[carePlan.length] || 'Follow physician recommendations for continued care');
    }
    carePlan = carePlan.slice(0, 5);

    // Save care plan to database if we have analysis/patient IDs
    if (analysisId || patientId) {
      try {
        await dbHelpers.run(
          `INSERT INTO care_plans 
           (analysis_id, patient_id, user_id, care_plan_points, plan_source, stage)
           VALUES (?, ?, ?, ?, ?, ?)`,
          [
            analysisId || null,
            patientId || 1, // Default patient ID
            req.user.id,
            JSON.stringify(carePlan),
            planSource,
            stage
          ]
        );
        console.log('Care plan saved to database successfully');
      } catch (dbError) {
        console.warn('Failed to save care plan to database:', dbError.message);
        // Continue anyway - return the care plan even if DB save fails
      }
    }

    console.log(`Care plan generation completed: ${planSource} (${carePlan.length} recommendations)`);

    res.json({
      success: true,
      care_plan: carePlan,
      source: planSource,
      stage: stage,
      recommendations_count: carePlan.length
    });

  } catch (error) {
    console.error('Care plan generation error:', error);
    
    // Return fallback care plan on any error
    const fallbackPlan = getFallbackCarePlan(req.body.stage || 'VMD (Very Mild Dementia)');
    
    res.json({
      success: true,
      care_plan: fallbackPlan,
      source: 'Fallback',
      stage: req.body.stage || 'VMD (Very Mild Dementia)',
      recommendations_count: fallbackPlan.length,
      warning: 'Used fallback care plan due to service unavailability'
    });
  }
});

// Get care plans for user
router.get('/', authenticateToken, async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;
    const offset = (page - 1) * limit;

    const carePlans = await dbHelpers.all(
      `SELECT cp.*, ar.predicted_stage, ar.confidence
       FROM care_plans cp
       LEFT JOIN analysis_results ar ON cp.analysis_id = ar.id
       WHERE cp.user_id = ?
       ORDER BY cp.generated_at DESC
       LIMIT ? OFFSET ?`,
      [req.user.id, limit, offset]
    );

    const total = await dbHelpers.get(
      'SELECT COUNT(*) as count FROM care_plans WHERE user_id = ?',
      [req.user.id]
    );

    // Parse care plan points
    const formattedCarePlans = carePlans.map(plan => ({
      ...plan,
      care_plan_points: JSON.parse(plan.care_plan_points)
    }));

    res.json({
      success: true,
      data: formattedCarePlans,
      pagination: {
        page,
        limit,
        total: total.count,
        pages: Math.ceil(total.count / limit)
      }
    });

  } catch (error) {
    console.error('Care plans fetch error:', error);
    res.status(500).json({ error: 'Failed to fetch care plans' });
  }
});

// Get specific care plan
router.get('/:id', authenticateToken, async (req, res) => {
  try {
    const carePlanId = req.params.id;

    const carePlan = await dbHelpers.get(
      `SELECT cp.*, ar.predicted_stage, ar.confidence
       FROM care_plans cp
       LEFT JOIN analysis_results ar ON cp.analysis_id = ar.id
       WHERE cp.id = ? AND cp.user_id = ?`,
      [carePlanId, req.user.id]
    );

    if (!carePlan) {
      return res.status(404).json({ error: 'Care plan not found' });
    }

    res.json({
      success: true,
      data: {
        ...carePlan,
        care_plan_points: JSON.parse(carePlan.care_plan_points)
      }
    });

  } catch (error) {
    console.error('Care plan fetch error:', error);
    res.status(500).json({ error: 'Failed to fetch care plan' });
  }
});

// Delete care plan
router.delete('/:id', authenticateToken, async (req, res) => {
  try {
    const carePlanId = req.params.id;

    // Verify care plan exists and belongs to user
    const existingPlan = await dbHelpers.get(
      'SELECT id FROM care_plans WHERE id = ? AND user_id = ?',
      [carePlanId, req.user.id]
    );

    if (!existingPlan) {
      return res.status(404).json({ error: 'Care plan not found' });
    }

    // Delete care plan
    await dbHelpers.run(
      'DELETE FROM care_plans WHERE id = ?',
      [carePlanId]
    );

    res.json({
      success: true,
      message: 'Care plan deleted successfully'
    });

  } catch (error) {
    console.error('Care plan deletion error:', error);
    res.status(500).json({ error: 'Failed to delete care plan' });
  }
});

// Fallback care plan function
function getFallbackCarePlan(stage) {
  const fallbackPlans = {
    'ND (No Dementia)': [
      "Maintain regular aerobic exercise routine 30-45 minutes daily to enhance neuroplasticity and reduce dementia risk",
      "Follow Mediterranean diet with weekly fish consumption targeting brain health and inflammation reduction",
      "Engage in challenging cognitive activities 3x weekly including puzzles, reading, and learning new skills",
      "Maintain consistent sleep schedule 7-8 hours nightly to optimize amyloid clearance and cognitive function",
      "Schedule annual comprehensive cognitive evaluations and biomarker monitoring for early detection"
    ],
    'VMD (Very Mild Dementia)': [
      "Establish structured daily routines with visual cues and calendars to support memory and reduce confusion",
      "Begin supervised cognitive rehabilitation program focusing on memory strategies and executive function",
      "Implement home safety modifications including improved lighting, grab bars, and clear pathways",
      "Participate in social engagement activities 2-3x weekly to maintain cognitive stimulation and emotional wellbeing",
      "Coordinate quarterly medical monitoring with neurologist and biomarker tracking for progression assessment"
    ],
    'MD (Mild Dementia)': [
      "Provide structured supervision for complex activities while encouraging maintained independence in basic tasks",
      "Implement comprehensive safety protocols including medical alert systems and environment modifications",
      "Establish consistent caregiver routines with clear communication strategies and behavioral management techniques",
      "Engage in simplified cognitive and physical activities tailored to current abilities and interests",
      "Coordinate multidisciplinary care team including neurology, social work, and occupational therapy services"
    ],
    'MOD (Moderate Dementia)': [
      "Ensure continuous supervision and assistance with all activities of daily living and personal care",
      "Create calming environment with familiar objects, consistent routines, and minimal overwhelming stimuli",
      "Implement person-centered care approaches using validation therapy and music/art therapy interventions",
      "Establish comprehensive safety measures including wandering prevention and 24-hour monitoring systems",
      "Provide extensive caregiver support including respite care, support groups, and professional care consultation"
    ]
  };

  return fallbackPlans[stage] || fallbackPlans['VMD (Very Mild Dementia)'];
}

export default router;