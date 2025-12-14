"""
Day 4: Database Schema Design
Weather Data Pipeline Project

This script creates the SQLite database and required tables:
1. cities
2. weather_data
3. pipeline_logs
"""

import sqlite3


def create_database():
    # Connect to SQLite database (creates file if not exists)
    conn = sqlite3.connect("weather_data.db")
    cursor = conn.cursor()

    # -------------------------------
    # Table 1: cities
    # -------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cities (
            city_id INTEGER PRIMARY KEY AUTOINCREMENT,
            city_name TEXT NOT NULL,
            country TEXT,
            latitude REAL,
            longitude REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # -------------------------------
    # Table 2: weather_data
    # -------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_data (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            city_id INTEGER NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            temperature_c REAL,
            humidity INTEGER,
            pressure_hpa REAL,
            wind_speed_mps REAL,
            weather_condition TEXT,
            FOREIGN KEY (city_id) REFERENCES cities(city_id)
        )
    """)

    # -------------------------------
    # Table 3: pipeline_logs
    # -------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT,
            message TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("Database and tables created successfully.")


if __name__ == "__main__":
    create_database()
