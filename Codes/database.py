import sqlite3
import pandas as pd
from datetime import datetime

class Database:
    def __init__(self, db_name="scam_detector.db"):
        """Initialize database connection"""
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.setup_database()
    
    def setup_database(self):
        """Create necessary tables if they don't exist"""
        # Table for storing detection history
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            prediction TEXT NOT NULL,
            probability REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Table for storing user feedback
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER,
            is_correct BOOLEAN NOT NULL,
            correct_label TEXT,
            feedback_text TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (detection_id) REFERENCES detection_history (id)
        )
        ''')
        self.conn.commit()
    
    def save_detection(self, text, prediction, probability):
        """Save a detection to the history table"""
        query = '''
        INSERT INTO detection_history (text, prediction, probability, timestamp)
        VALUES (?, ?, ?, ?)
        '''
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(query, (text, prediction, probability, current_time))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def save_feedback(self, detection_id, is_correct, correct_label=None, feedback_text=None):
        """Save user feedback on a detection"""
        query = '''
        INSERT INTO user_feedback (detection_id, is_correct, correct_label, feedback_text, timestamp)
        VALUES (?, ?, ?, ?, ?)
        '''
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(query, (detection_id, is_correct, correct_label, feedback_text, current_time))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_recent_detections(self, limit=10):
        """Get recent detection history"""
        query = '''
        SELECT id, text, prediction, probability, timestamp 
        FROM detection_history 
        ORDER BY timestamp DESC
        LIMIT ?
        '''
        self.cursor.execute(query, (limit,))
        rows = self.cursor.fetchall()
        return [{"id": row[0], "text": row[1], "prediction": row[2], 
                "probability": row[3], "timestamp": row[4]} for row in rows]
    
    def get_feedback_stats(self):
        """Get statistics about user feedback"""
        query = '''
        SELECT 
            COUNT(*) as total_feedback,
            SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
            SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) as incorrect_predictions
        FROM user_feedback
        '''
        self.cursor.execute(query)
        result = self.cursor.fetchone()
        if result[0] == 0:  # No feedback yet
            return {"total_feedback": 0, "correct_rate": 0, "incorrect_rate": 0}
        
        total = result[0]
        correct = result[1]
        incorrect = result[2]
        
        return {
            "total_feedback": total,
            "correct_rate": (correct / total) * 100 if total > 0 else 0,
            "incorrect_rate": (incorrect / total) * 100 if total > 0 else 0
        }
    
    def export_feedback_data(self):
        """Export user feedback data for model retraining"""
        query = '''
        SELECT 
            h.text, 
            COALESCE(f.correct_label, h.prediction) as label
        FROM detection_history h
        LEFT JOIN user_feedback f ON h.id = f.detection_id
        WHERE f.id IS NOT NULL
        '''
        return pd.read_sql_query(query, self.conn)
    
    def close(self):
        """Close the database connection"""
        self.conn.close()