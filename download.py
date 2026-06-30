"""
Download WARC files for a specified range of months.

See also: https://github.com/LukasKriesch/CommonCrawlNewsDataSet/blob/main/Project_Scripts/01_download_newscrawl.py
"""

import gzip
import os
import time
import logging

import requests
import urllib3

# Configure logging
logging.basicConfig(
    filename="analyze.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # Suppress only the InsecureRequestWarning from urllib3


def download_with_retries(url, local_path, retries=5, backoff=10, verify_ssl=False):
    """
    Download file including resume, retries with exponential backoff and security option.
    """
    wait_time = backoff

    for attempt in range(1, retries + 1):
        # Check how many bytes we already have
        resume_byte_pos = os.path.getsize(local_path) if os.path.exists(local_path) else 0
        headers = {}

        if resume_byte_pos > 0:
            headers["Range"] = f"bytes={resume_byte_pos}-"

        try:
            logging.info("Downloading (attempt %d): %s", attempt, url)
            # Stream download
            with requests.get(
                url,
                headers=headers,
                stream=True,
                verify=verify_ssl
                ) as response:

                if response.status_code == 416:
                    logging.info("File already fully downloaded: %s", local_path)
                    return True, local_path

                response.raise_for_status()

                # If server doesn't support resume, restart from scratch
                if resume_byte_pos > 0 and response.status_code != 206:
                    resume_byte_pos = 0

                if resume_byte_pos > 0:
                    mode = "ab"
                    logging.info("Resuming from byte: %d", resume_byte_pos)
                else:
                    mode = "wb"
                downloaded = resume_byte_pos
                with open(local_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                logging.info("Download complete (%d bytes): %s", downloaded, local_path)
                return True, local_path
        except Exception as e:
            logging.warning("Error downloading %s: %s. Retrying in %ds...", url, e, wait_time)
            time.sleep(wait_time)
            wait_time *= 2
    logging.error("Failed to download %s after %d attempts", url, retries)
    return False, None


def download_all_paths(year: str, start: int, end: int):
    """
    Download warc.paths.gz for all months in the year.
    """
    warc_paths_file_base_url = "https://data.commoncrawl.org/crawl-data/CC-NEWS/"
    successful_months = []
    failed_months = []

    for month in range(start, (end+1)):  # Try from start to end
        month_str = f"{month:02d}"

        year_dir = os.path.join(r".", year)
        os.makedirs(year_dir, exist_ok=True)

        month_dir = os.path.join(year_dir, f"{year}_{month_str}")
        os.makedirs(month_dir, exist_ok=True)

        warc_paths_file = os.path.join(month_dir, "warc.paths.gz")
        url = warc_paths_file_base_url + f"{year}/{month_str}/warc.paths.gz"

        if not download_with_retries (url, warc_paths_file)[0]:
            failed_months.append(f"{year}/{month_str}")
            continue

        # Extract paths of this month
        with gzip.open(warc_paths_file, 'rt') as f:
            paths_month = [line.strip() for line in f if line.strip()]
        logging.info("Extracted %d paths", len(paths_month))

        # Append new paths to paths_year_file (deduplicated)
        paths_year_file = "paths.txt"
        existing_paths = set()
        if os.path.exists(paths_year_file):
            with open(paths_year_file, "r") as f:
                existing_paths = {line.rstrip("\n") for line in f}
        with open(paths_year_file, 'a') as f:
            for path in paths_month:
                if path not in existing_paths:
                    f.write(path + '\n')
                    existing_paths.add(path)

        logging.info("Appended extracted paths to %s\n", paths_year_file)

        successful_months.append(f"{year}/{month_str}")
    # Summary
    logging.info("="*50)
    logging.info("Download summary for year %s:\nSuccessful months: %s, Failed months: %s", year, successful_months, failed_months)
    logging.info("="*50)
    return len(successful_months) > 0


def download_warc_file(path):
    """
    Download a single WARC file.
    """
    warc_file_base_url = "https://data.commoncrawl.org/"
    url = warc_file_base_url + path
    warc_file_path = os.path.basename(url)
    if not download_with_retries(url, warc_file_path)[0]:
        logging.error("Failed to download WARC file: %s", url)
        return None
    return warc_file_path
