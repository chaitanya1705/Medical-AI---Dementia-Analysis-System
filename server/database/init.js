import sqlite3 from 'sqlite3';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const dbDir = path.join(__dirname, '../data');
if (!fs.existsSync(dbDir)) {
  fs.mkdirSync(dbDir, { recursive: true });
}

const dbPath = path.join(dbDir, 'medical_ai.db');

const db = new sqlite3.Database(dbPath, (err) => {
  if (err) {
    console.error(' Error opening database:', err.message);
  } else {
    console.log(' Connected to SQLite database at:', dbPath);
  }
});

// Enable foreign keys
db.run('PRAGMA foreign_keys = ON');

// Initialize database tables
export async function initializeDatabase() {
  return new Promise((resolve, reject) => {
    db.serialize(() => {
      // Users table for authentication
      db.run(`
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT UNIQUE NOT NULL,
          password_hash TEXT NOT NULL,
          email TEXT,
          first_name TEXT,
          last_name TEXT,
          specialty TEXT,
          hospital TEXT,
          license_number TEXT,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
      `);

      // Patients table
      db.run(`
        CREATE TABLE IF NOT EXISTS patients (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL,
          first_name TEXT NOT NULL,
          last_name TEXT NOT NULL,
          email TEXT,
          gender TEXT,
          medical_condition TEXT,
          patient_id_number TEXT,
          date_of_birth DATE,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
      `);

      // Analysis results table
      db.run(`
        CREATE TABLE IF NOT EXISTS analysis_results (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          patient_id INTEGER NOT NULL,
          user_id INTEGER NOT NULL,
          predicted_stage TEXT NOT NULL,
          stage_index INTEGER NOT NULL,
          confidence REAL NOT NULL,
          progression_months REAL NOT NULL,
          prediction_method TEXT NOT NULL,
          mri_image_data BLOB,
          mri_image_type TEXT,
          biomarkers_data TEXT NOT NULL, -- JSON string
          enhanced_features_count INTEGER,
          all_probabilities TEXT, -- JSON string
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE,
          FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
      `);

      // Care plans table
      db.run(`
        CREATE TABLE IF NOT EXISTS care_plans (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          analysis_id INTEGER NOT NULL,
          patient_id INTEGER NOT NULL,
          user_id INTEGER NOT NULL,
          care_plan_points TEXT NOT NULL, -- JSON array
          plan_source TEXT NOT NULL, -- 'AI-Generated' or 'Fallback'
          stage TEXT NOT NULL,
          generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (analysis_id) REFERENCES analysis_results (id) ON DELETE CASCADE,
          FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE,
          FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
      `);

      // Biomarkers table for detailed storage
      db.run(`
        CREATE TABLE IF NOT EXISTS biomarkers (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          analysis_id INTEGER NOT NULL,
          biomarker_name TEXT NOT NULL,
          biomarker_value REAL NOT NULL,
          biomarker_unit TEXT,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (analysis_id) REFERENCES analysis_results (id) ON DELETE CASCADE
        )
      `);

      // Audit log table
      db.run(`
        CREATE TABLE IF NOT EXISTS audit_log (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER,
          action TEXT NOT NULL,
          table_name TEXT,
          record_id INTEGER,
          old_values TEXT, -- JSON
          new_values TEXT, -- JSON
          ip_address TEXT,
          user_agent TEXT,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
        )
      `);

      // Create indexes for better performance
      db.run('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)');
      db.run('CREATE INDEX IF NOT EXISTS idx_patients_user_id ON patients(user_id)');
      db.run('CREATE INDEX IF NOT EXISTS idx_analysis_patient_id ON analysis_results(patient_id)');
      db.run('CREATE INDEX IF NOT EXISTS idx_analysis_user_id ON analysis_results(user_id)');
      db.run('CREATE INDEX IF NOT EXISTS idx_care_plans_analysis_id ON care_plans(analysis_id)');
      db.run('CREATE INDEX IF NOT EXISTS idx_biomarkers_analysis_id ON biomarkers(analysis_id)');
      db.run('CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id)');
      db.run('CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at)');

      console.log(' Database tables initialized successfully');
      resolve();
    });
  });
}

// Database helper functions
export const dbHelpers = {
  // Run a query
  run: (sql, params = []) => {
    return new Promise((resolve, reject) => {
      db.run(sql, params, function(err) {
        if (err) {
          reject(err);
        } else {
          resolve({ id: this.lastID, changes: this.changes });
        }
      });
    });
  },

  // Get single row
  get: (sql, params = []) => {
    return new Promise((resolve, reject) => {
      db.get(sql, params, (err, row) => {
        if (err) {
          reject(err);
        } else {
          resolve(row);
        }
      });
    });
  },

  // Get all rows
  all: (sql, params = []) => {
    return new Promise((resolve, reject) => {
      db.all(sql, params, (err, rows) => {
        if (err) {
          reject(err);
        } else {
          resolve(rows);
        }
      });
    });
  },

  // Begin transaction
  beginTransaction: () => {
    return new Promise((resolve, reject) => {
      db.run('BEGIN TRANSACTION', (err) => {
        if (err) reject(err);
        else resolve();
      });
    });
  },

  // Commit transaction
  commit: () => {
    return new Promise((resolve, reject) => {
      db.run('COMMIT', (err) => {
        if (err) reject(err);
        else resolve();
      });
    });
  },

  // Rollback transaction
  rollback: () => {
    return new Promise((resolve, reject) => {
      db.run('ROLLBACK', (err) => {
        if (err) reject(err);
        else resolve();
      });
    });
  },

  // Close database connection
  close: () => {
    return new Promise((resolve, reject) => {
      db.close((err) => {
        if (err) reject(err);
        else resolve();
      });
    });
  }
};

export default db;