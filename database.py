"""
Create SQLite database and store the sentiment and category analysis results.
"""

import logging

import sqlite3


# Configure logging
logging.basicConfig(
    filename="analyze.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


def create_table(database_name:str):
    """
    Create the news_articles table if it doesn't exist.
    """
    try:
        conn = sqlite3.connect(database_name)
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT,
                url_Timestamp TEXT,
                text TEXT,
                category TEXT,
                sentiment INTEGER,
                sentiment_score REAL,
                politics_score REAL,
                business_score REAL,
                technology_score REAL,
                science_score REAL,
                health_score REAL,
                sports_score REAL,
                entertainment_score REAL,
                lifestyle_score REAL,
                education_score REAL,
                environment_score REAL,
                crime_score REAL,
                weather_score REAL,
                economy_score REAL,
                real_estate_score REAL,
                automotive_score REAL,
                travel_score REAL
            )
        ''')
        cur.execute('''
            CREATE UNIQUE INDEX IF NOT EXISTS idx_url_timestamp
            ON news_articles (url, url_Timestamp)
        ''')
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        logging.error(f"Error creating table: {e}")


def insert_record(cur, record):
    """
    Insert a record into the news_articles table.
    """
    try:
        # Insert record into table
        cur.execute('''
            INSERT OR IGNORE INTO news_articles (url, url_Timestamp, text, category, sentiment, sentiment_score,
                politics_score, business_score, technology_score, science_score,
                health_score, sports_score, entertainment_score, lifestyle_score,
                education_score, environment_score, crime_score, weather_score,
                economy_score, real_estate_score, automotive_score, travel_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (record['url'], record['url_Timestamp'], record['text'], record['category'],
              record['sentiment'],
              record['sentiment_score'], record['politics_score'], record['business_score'],
              record['technology_score'], record['science_score'], record['health_score'],
              record['sports_score'], record['entertainment_score'], record['lifestyle_score'],
              record['education_score'], record['environment_score'], record['crime_score'],
              record['weather_score'], record['economy_score'], record['real_estate_score'],
              record['automotive_score'], record['travel_score']))
    except sqlite3.Error as e:
        logging.error(f"Error inserting record: {e}")
