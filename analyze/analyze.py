import argparse
import os
from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from transformers import pipeline
import sqlite3
import re
import time
import logging

logging.basicConfig(filename="analyze.log", 
                    format='%(asctime)s: %(levelname)s: %(message)s', 
                    level=logging.INFO)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--warc_dir', type=str, default='/mnt/commoncrawl', help='Base directory for WARC files')
parser.add_argument('--db_name', type=str, default='newswarc.db', help='SQLite database name')
args = parser.parse_args()

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

# Define the candidate labels
candidate_labels_1 = ["positive news", "negative news"]
candidate_labels_2 = [
    'Politics', 'Business',
    'Technology', 'Science', 'Health', 'Sports', 'Entertainment', 
    'Lifestyle', 'Education', 'Environment', 'Crime', 
    'Weather', 'Economy', 'Real Estate', 'Automotive', 'Travel'
]

# Function to determine if a page is in English
def is_english_page(content):
    try:
        soup = BeautifulSoup(content, 'html.parser')
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang') == 'en':
            return True
        
        meta_tag = soup.find('meta', {'http-equiv': 'Content-Language'})
        if meta_tag and meta_tag.get('content') == 'en':
            return True
        
        text = soup.get_text()
        english_words = re.findall(r'\b\w+\b', text.lower())
        english_count = sum(1 for word in english_words if word in set([
            "the", "and", "is", "in", "it", "you", "that", "he", "was", "for", 
            "on", "are", "with", "as", "I", "his", "they", "be", "at", "one", 
            "have", "this", "from", "or", "had", "by", "not", "word", "but", 
            "what", "some", "we", "can", "out", "other", "were", "all", "there", 
            "when", "up", "use", "your", "how", "said", "an", "each", "she"]))
        return english_count / len(english_words) > 0.5 if english_words else False
    except Exception as e:
        logging.error(f"An error occurred while determining page language: {e}")
        #print(f"An error occurred while determining page language: {e}")
        return False

# Function to insert a record into the SQLite database
def insert_record(cur, record, conn):
    try:
        cur.execute('''
            INSERT INTO news_articles (url, url_Timestamp, category, sentiment, score)
            VALUES (?, ?, ?, ?, ?)
        ''', (record['url'], record['timestamp'], record['category'], record['predicted'], record['score']))
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Error inserting record: {e}")
        #print(f"Error inserting record: {e}")

# Function to process WARC records
def process_warc_records(warc_file_path, cur, conn):
    count = 0
    start_time = time.time()
    try:
        with open(warc_file_path, 'rb') as stream:
            for record in ArchiveIterator(stream):
                warc_date = record.rec_headers.get_header("WARC-Date")
                if warc_date:
                    try:
                        content = record.content_stream().read() if record.content_stream() else b''
                        content = content.decode(errors='ignore')

                        soup = BeautifulSoup(content, 'html.parser')
                        title = soup.title.string if soup.title else ''
                        headers = ' '.join([h.get_text() for h in soup.find_all(['h1', 'h2', 'h3'])])
                        combined_content = f"{title} {headers}"
                        first_15_words = ' '.join(combined_content.split()[:15])
                        first_40_words = ' '.join(combined_content.split()[:40])
                        if is_english_page(content):
                            url = record.rec_headers.get_header("WARC-Target-URI")
                            
                            result_1 = classifier(first_15_words, candidate_labels_1)
                            label_1 = result_1['labels'][0]
                            score_1 = result_1['scores'][0]
                            result_2 = classifier(first_40_words, candidate_labels_2)
                            label_2 = result_2['labels'][0]
                            score_2 = result_2['scores'][0]
                            
                            iso_string = warc_date
                            dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
                            formatted_dt = dt.strftime('%Y-%m-%d %H:%M:%S')
                            label_number = 1 if label_1 == 'positive news' else 0
                            
                            record_to_insert = {
                                'url': url,
                                'timestamp': formatted_dt,
                                'content': first_15_words,
                                'predicted': label_number,
                                'score': score_1,
                                'category': label_2
                            }
                            insert_record(cur, record_to_insert, conn)
                            count += 1

                    except ValueError:
                        #print(f"Error parsing date: {warc_date}")
                        logging.error(f"Error parsing date: {warc_date}")

            end_time = time.time()
            elapsed_seconds = end_time - start_time
            elapsed_time = str(timedelta(seconds=elapsed_seconds))
            #print(f"Processed {count} WARC records in {elapsed_time}")
            logging.info(f"Processed {count} WARC records in {elapsed_time}")

    except FileNotFoundError:
        #print(f"File not found: {warc_file_path}")
        logging.error(f"File not found: {warc_file_path}")
    except Exception as e:
        #print(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")


# Function to process all WARC files in the specified directory
def process_directory(warc_dir, conn):
    cur = conn.cursor()

    for root, dirs, files in os.walk(warc_dir):
        for file in files:
            if file.endswith(".warc.gz"):
                warc_file_path = os.path.join(root, file)
                logging.info(f"Processing {warc_file_path}...")
                #print(f"Processing {warc_file_path}...")
                process_warc_records(warc_file_path, cur, conn)

# Connect to SQLite using the provided database name
try:
    conn = sqlite3.connect(args.db_name)
except sqlite3.Error as e:
    #print(f"Error connecting to SQLite: {e}")
    logging.error(f"Error connecting to SQLite: {e}")
    sys.exit(1)

# Process the directory
process_directory(args.warc_dir, conn)

# Close the database connection
conn.close()
