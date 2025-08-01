import express from 'express';
import { authenticateToken } from './auth.js';
import { dbHelpers } from '../database/init.js';

const router = express.Router();

// Create new patient
router.post('/', authenticateToken, async (req, res) => {
  try {
    const {
      firstName,
      lastName,
      email,
      gender,
      medicalCondition,
      patientIdNumber,
      dateOfBirth
    } = req.body;

    // Validate required fields
    if (!firstName || !lastName) {
      return res.status(400).json({ error: 'First name and last name are required' });
    }

    // Create patient
    const result = await dbHelpers.run(
      `INSERT INTO patients 
       (user_id, first_name, last_name, email, gender, medical_condition, patient_id_number, date_of_birth)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      [req.user.id, firstName, lastName, email, gender, medicalCondition, patientIdNumber, dateOfBirth]
    );

    res.status(201).json({
      success: true,
      message: 'Patient created successfully',
      patient: {
        id: result.id,
        firstName,
        lastName,
        email,
        gender,
        medicalCondition,
        patientIdNumber,
        dateOfBirth
      }
    });

  } catch (error) {
    console.error('Patient creation error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get all patients for user
router.get('/', authenticateToken, async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;
    const offset = (page - 1) * limit;

    const patients = await dbHelpers.all(
      `SELECT * FROM patients 
       WHERE user_id = ? 
       ORDER BY created_at DESC 
       LIMIT ? OFFSET ?`,
      [req.user.id, limit, offset]
    );

    const total = await dbHelpers.get(
      'SELECT COUNT(*) as count FROM patients WHERE user_id = ?',
      [req.user.id]
    );

    res.json({
      success: true,
      data: patients,
      pagination: {
        page,
        limit,
        total: total.count,
        pages: Math.ceil(total.count / limit)
      }
    });

  } catch (error) {
    console.error('Patients fetch error:', error);
    res.status(500).json({ error: 'Failed to fetch patients' });
  }
});

// Get specific patient
router.get('/:id', authenticateToken, async (req, res) => {
  try {
    const patientId = req.params.id;

    const patient = await dbHelpers.get(
      'SELECT * FROM patients WHERE id = ? AND user_id = ?',
      [patientId, req.user.id]
    );

    if (!patient) {
      return res.status(404).json({ error: 'Patient not found' });
    }

    res.json({
      success: true,
      data: patient
    });

  } catch (error) {
    console.error('Patient fetch error:', error);
    res.status(500).json({ error: 'Failed to fetch patient' });
  }
});

// Update patient
router.put('/:id', authenticateToken, async (req, res) => {
  try {
    const patientId = req.params.id;
    const {
      firstName,
      lastName,
      email,
      gender,
      medicalCondition,
      patientIdNumber,
      dateOfBirth
    } = req.body;

    // Verify patient exists and belongs to user
    const existingPatient = await dbHelpers.get(
      'SELECT id FROM patients WHERE id = ? AND user_id = ?',
      [patientId, req.user.id]
    );

    if (!existingPatient) {
      return res.status(404).json({ error: 'Patient not found' });
    }

    // Update patient
    await dbHelpers.run(
      `UPDATE patients SET 
        first_name = ?, last_name = ?, email = ?, gender = ?,
        medical_condition = ?, patient_id_number = ?, date_of_birth = ?,
        updated_at = CURRENT_TIMESTAMP
       WHERE id = ?`,
      [firstName, lastName, email, gender, medicalCondition, patientIdNumber, dateOfBirth, patientId]
    );

    res.json({
      success: true,
      message: 'Patient updated successfully'
    });

  } catch (error) {
    console.error('Patient update error:', error);
    res.status(500).json({ error: 'Failed to update patient' });
  }
});

// Delete patient
router.delete('/:id', authenticateToken, async (req, res) => {
  try {
    const patientId = req.params.id;

    // Verify patient exists and belongs to user
    const existingPatient = await dbHelpers.get(
      'SELECT id FROM patients WHERE id = ? AND user_id = ?',
      [patientId, req.user.id]
    );

    if (!existingPatient) {
      return res.status(404).json({ error: 'Patient not found' });
    }

    // Delete patient (cascade will handle related records)
    await dbHelpers.run(
      'DELETE FROM patients WHERE id = ?',
      [patientId]
    );

    res.json({
      success: true,
      message: 'Patient deleted successfully'
    });

  } catch (error) {
    console.error('Patient deletion error:', error);
    res.status(500).json({ error: 'Failed to delete patient' });
  }
});

export default router;