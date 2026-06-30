"""
Classify WARC records using a zero-shot classifier model to build a dataset.
"""

import os
import sys
import logging
import argparse
import sqlite3

from download import download_all_paths, download_warc_file
from database import create_table

# Configure logging
logging.basicConfig(
    filename="analyze.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, help='Directory of WARC files to process.')
    parser.add_argument('--year', type=str, default=None, help='e.g. 2025.')
    parser.add_argument('--start_month', type=int, default=None, help= 'int from 1 to 12')
    parser.add_argument('--end_month', type=int, default=None, help= 'int from 1 to 12')
    args = parser.parse_args()

    directory = args.directory
    year = args.year
    start_month = args.start_month
    end_month = args.end_month

    # Ensure that either directory is provided or all of year, start_month, and end_month, but not both
    year_month_provided = year is not None or start_month is not None or end_month is not None
    if directory and year_month_provided or not directory and not year_month_provided:
        logging.error("Please provide either a directory path or all of year, start month, and end month")
        return
    if year_month_provided and (not year or not start_month or not end_month):
        logging.error("Please provide all of year, start month, and end month")
        return

    # Create database and table
    database_name = year + ".db" if not directory else os.path.basename(directory) + ".db"
    try:
        create_table(database_name)
    except sqlite3.Error as e:
        logging.error("Error creating database table: %s", e)

    from analyze import process_warc_files

    paths_file = "paths.txt"
    if directory:
        if os.path.exists(paths_file) and os.path.getsize(paths_file) > 0:
            logging.info("Found existing paths file")
        else:
            with open(paths_file, 'w', encoding='utf-8') as f:
                for filename in os.listdir(directory):
                    if filename.endswith(".warc.gz"):
                        warc_file = os.path.join(directory, filename)
                        f.write(warc_file + "\n")
    else:
        # If no directory is provided, download paths file from Common
        # Crawl for the specified year and month range. Skip downloading
        # if paths.txt already exists with content.
        if os.path.exists(paths_file) and os.path.getsize(paths_file) > 0:
            logging.info("Found existing paths file")
        else:
            if not download_all_paths(year, start_month, end_month):
                logging.error("Failed to download paths file")
                return

    with open(paths_file, 'r', encoding='utf-8') as f:
        paths = [line.strip() for line in f if line.strip()]
        # Iterate over the paths
        for path in paths:
            warc_file_path = path
            if not directory:
                warc_file_path = download_warc_file(path)
                if not warc_file_path:
                    continue
            # Connect to SQLite database
            try:
                conn = sqlite3.connect(database_name)
            except sqlite3.Error as e:
                logging.error("Error connecting to database: %s", e)
                sys.exit(1)

            # Process the file
            process_warc_files(warc_file_path, conn)

            # Close the database connection
            conn.close()


if __name__ == '__main__':
    # This block is only executed in the "parent" process
    main()
