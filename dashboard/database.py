import sqlite3
import pandas as pd
import json

class DatabaseManager:
    """
    Manages SQLite database for storing generated datasets, 
    training runs, and metrics for the dashboard.
    """
    def __init__(self, db_path='project_data.db'):
        self.db_path = db_path
        self._init_db()
        
    def _get_conn(self):
        return sqlite3.connect(self.db_path)
        
    def _init_db(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # Experiments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    model_type TEXT,
                    epsilon REAL,
                    epochs INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    metric_name TEXT,
                    metric_value REAL,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                )
            ''')
            
            conn.commit()
            
    def log_experiment(self, name, model_type, epsilon, epochs):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO experiments (name, model_type, epsilon, epochs) VALUES (?, ?, ?, ?)",
                (name, model_type, epsilon, epochs)
            )
            return cursor.lastrowid
            
    def log_metrics(self, experiment_id, metrics_dict):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            for k, v in metrics_dict.items():
                # Handle numpy types
                val = float(v) if hasattr(v, 'item') else v
                cursor.execute(
                    "INSERT INTO metrics (experiment_id, metric_name, metric_value) VALUES (?, ?, ?)",
                    (experiment_id, k, val)
                )
                
    def get_experiments(self):
        with self._get_conn() as conn:
            return pd.read_sql("SELECT * FROM experiments", conn)
            
    def get_metrics(self, experiment_id):
        with self._get_conn() as conn:
            return pd.read_sql(f"SELECT * FROM metrics WHERE experiment_id = {experiment_id}", conn)
