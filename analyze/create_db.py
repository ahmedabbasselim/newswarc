import sqlite3

# Connect to SQLite database
conn = sqlite3.connect("newswarc.db")
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS news_articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        url_timestamp DATETIME NOT NULL,
        category TEXT NOT NULL,
        sentiment TEXT NOT NULL,
        score REAL NOT NULL
    )
''')

# Commit and close
conn.commit()
conn.close()

print("Table 'news_articles' created successfully!")
