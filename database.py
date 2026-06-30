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
                category_score REAL,
                sentiment INTEGER,
                sentiment_score REAL
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
            INSERT OR IGNORE INTO news_articles (url, url_Timestamp, text, category, category_score,
                sentiment, sentiment_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (record['url'], record['url_Timestamp'], record['text'], record['category'],
              record['category_score'], record['sentiment'], record['sentiment_score']))
    except sqlite3.Error as e:
        logging.error(f"Error inserting record: {e}")
