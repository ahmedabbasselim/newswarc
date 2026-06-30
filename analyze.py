"""
Iterate over WARC files and process records.
"""

import os
import warnings
import logging
import time
from datetime import datetime
import multiprocessing as mp

import torch
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from langdetect import detect
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator
from warcio.exceptions import ArchiveLoadFailed

from database import insert_record
from classifier import get_classifier


warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", message=".*huggingface/tokenizers.*")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    filename="analyze.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Get number of available CPU cores (respects SLURM)
num_cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()

model = None  # pylint: disable=invalid-name
tokenizer = None  # pylint: disable=invalid-name
device = None  # pylint: disable=invalid-name


def log_gpu_mem(stage=""):
    """
    Log GPU memory usage for debugging and choosing the correct batch size.
    """
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated() / 1024**3
    logging.info("[GPU %s] Allocated: %.2f GB | Free: %.2f / %.2f GB",
                 stage, alloc, free/1024**3, total/1024**3)


def check_model_loaded():
    """
    Check if the model is loaded. If not, load it. This ensures that the model is loaded only once.
    """
    global model, tokenizer, device  # pylint: disable=global-statement
    if model is None:
        model, tokenizer, device = get_classifier()
        #log_gpu_mem("after model loaded")


def generate_html(warc_file_path):
    """
    Generate HTML content, URL, and timestamp from WARC records.
    """
    with open(warc_file_path, 'rb') as stream:
        iterator = ArchiveIterator(stream)
        while True:
            try:
                record = next(iterator)
            except StopIteration:
                break
            except ArchiveLoadFailed as e:
                #logging.warning("Skipping corrupt record")
                continue
            if record.rec_type != "response":
                continue
            html_content = record.content_stream().read().split(b'\r\n\r\n', 1)[-1].decode('utf-8', errors='ignore')
            url = record.rec_headers.get_header('WARC-Target-URI')
            timestamp = record.rec_headers.get_header('WARC-Date')
            if not timestamp:
                continue
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                logging.warning("Skipping record with invalid timestamp: %s", timestamp)
                continue
            yield html_content, url, timestamp


def parse_html(args, max_words=40):
    """
    Parse generated HTML to extract text and filter EN text (runs on pool of workers).
    """
    html_content, url, timestamp = args
    soup = BeautifulSoup(html_content, "html.parser")

    # Check HTML lang before extracting text
    html_tag = soup.find('html')
    if html_tag and html_tag.get('lang') and html_tag.get('lang') != 'en':
        return None
    meta_tag = soup.find('meta', {'http-equiv': 'Content-Language'})
    if meta_tag and meta_tag.get('content') and meta_tag.get('content') != 'en':
        return None

    title = soup.title.get_text() if soup.title else ''
    headers = ' '.join([h.get_text() for h in soup.find_all(['h1', 'h2', 'h3'])])
    text = f"{title} {headers}".strip()
    if not text or len(text.split()) < 5:
        return None
    truncated = " ".join(text.split()[:max_words])
    try:
        if detect(truncated) != 'en':
            return None
    except Exception:
        return None
    return {"text": truncated, "url": url, "timestamp": timestamp}


def feed_queue(warc_file_path, output_queue):
    """
    Feed the queue with the parsed records from the WARC file. Process
    the output from the generator (generate_html) using the parser
    (parse_html). Processing runs in parallel on CPU to keep GPU fed
    with data without overwhelming its RAM. The queue has a maxsize to
    apply backpressure on CPU RAM if GPU is slower.
    """
    with mp.Pool(num_cores) as pool:
        for result in pool.imap(parse_html, generate_html(warc_file_path), chunksize=75):
            if result is not None:
                output_queue.put(result)
    output_queue.put(None)  # Signal that parsing is done


def process_warc_records(warc_file_path, cur, conn, batch=600, category_batch=200):
    """
    WARC files
        ↓
    Generate HTML content, URL, and timestamp from WARC records
        ↓
    Parse HTML to extract text and filter EN text, with multiprocessing
        ↓
    Feed queue
        ↓
    Generate batches from the queue for tokenization and classification
        ↓
    GPU zero-shot classification inference (category pass first, then sentiment pass)
        ↓
    Store in database
    """
    check_model_loaded()

    def generate_batch(queue, batch_size):
        """
        Generate batches from the queue for tokenization and classification.
        """
        batch = []

        while True:
            item = queue.get()

            if item is None:
                break

            batch.append(item)

            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    # Define the candidate labels
    candidate_labels_1 = ["positive news", "negative news"]
    candidate_labels_2 = [
        'Politics', 'Business',
        'Technology', 'Science', 'Health', 'Sports', 'Entertainment',
        'Lifestyle', 'Education', 'Environment', 'Crime',
        'Weather', 'Economy', 'Real Estate', 'Automotive', 'Travel']
    hypotheses_1 = [f"This example is {l}." for l in candidate_labels_1]
    hypotheses_2 = [f"This example is about {l}." for l in candidate_labels_2]

    def classify_batch(texts):
        """
        Classify a batch of texts using zero-shot model.
        """
        # Category pass first: heavier (16 hypotheses, longer texts) — runs on clean unfragmented VRAM
        all_entail_2 = []
        for i in range(0, len(texts), category_batch):
            chunk = texts[i:i + category_batch]
            pairs_2 = []
            for text in chunk:
                truncated = " ".join(text.split()[:40])
                for hyp in hypotheses_2:
                    pairs_2.append((truncated, hyp))
            # Convert the text into tokens that the model can understand
            inputs_2 = tokenizer(
                [p[0] for p in pairs_2],
                [p[1] for p in pairs_2],
                padding=True, truncation=True, max_length=60, return_tensors="pt"
            )
            inputs_2 = {k: v.to(device) for k, v in inputs_2.items()}
            #log_gpu_mem(f"after category subbatch loaded")
            # Use inference mode to disable gradient computation (gradients are not needed for inference)
            with torch.inference_mode():
                # Extract the logits
                logits_2 = model(**inputs_2).logits
            # Extract entailment scores from the logits
            all_entail_2.append(logits_2[:, 2].view(len(chunk), len(candidate_labels_2)))
        # Concatenate entailment scores for all chunks
        entail_2 = torch.cat(all_entail_2, dim=0)  # pylint: disable=no-member
        # Convert entailment scores to probabilities using softmax
        probs_2 = torch.softmax(entail_2, dim=1)  # pylint: disable=no-member
        # Keep all category probabilities for each text
        all_category_probs = probs_2.cpu()
        del inputs_2, logits_2, all_entail_2, entail_2, probs_2
        #log_gpu_mem("after category pass")
        # Sentiment pass second: lightweight (2 hypotheses, shorter texts) — fits in fragmented VRAM
        pairs_1 = []
        for text in texts:
            truncated = " ".join(text.split()[:15])
            for hyp in hypotheses_1:
                pairs_1.append((truncated, hyp))

        inputs_1 = tokenizer(
            [p[0] for p in pairs_1],
            [p[1] for p in pairs_1],
            padding=True, truncation=True, max_length=25, return_tensors="pt"
        )
        inputs_1 = {k: v.to(device) for k, v in inputs_1.items()}
        #log_gpu_mem(f"after sentiment batch loaded")

        with torch.inference_mode():
            logits_1 = model(**inputs_1).logits
        entail_1 = logits_1[:, 2].view(len(texts), len(candidate_labels_1))
        # Convert logits to probabilities
        probs_1 = torch.softmax(entail_1, dim=1)  # pylint: disable=no-member
        # Get the predicted class using probabilities
        predicted_class_1 = probs_1.argmax(dim=1)
        # Get the maximum probability score for the predicted class for each text
        max_scores_1 = probs_1.max(dim=1).values
        del inputs_1, logits_1, entail_1, probs_1
        torch.cuda.empty_cache()
        #log_gpu_mem("after sentiment pass")
        # Map the predicted class to the label and include all category scores
        category_score_keys = [
            'politics_score', 'business_score', 'technology_score', 'science_score',
            'health_score', 'sports_score', 'entertainment_score', 'lifestyle_score',
            'education_score', 'environment_score', 'crime_score', 'weather_score',
            'economy_score', 'real_estate_score', 'automotive_score', 'travel_score']
        results = []
        for i in range(len(texts)):
            sentiment = 1 if predicted_class_1[i] == 0 else 0
            sentiment_score = max_scores_1[i].item()
            top_category = candidate_labels_2[all_category_probs[i].argmax().item()]
            cat_scores = {k: all_category_probs[i, j].item()
                          for j, k in enumerate(category_score_keys)}
            results.append((sentiment, sentiment_score, top_category, cat_scores))
        return results

    # Create workers and queue. When the queue has 1200 items,
    # output_queue.put() in feed_queue waits until the GPU consumer
    # pulls an item out. This prevents the parser from running too far
    # ahead and filling up RAM with thousands of parsed records that the
    # GPU hasn't processed yet. The fast parser slows down to match the
    # slower GPU consumer.
    queue = mp.Queue(maxsize=1200)
    workers = []
    p = mp.Process(target=feed_queue, args=(warc_file_path, queue))
    p.start()
    workers.append(p)

    logging.info("Processing batches started for WARC file: %s", os.path.basename(warc_file_path))
    record_count = 0

    # Classify and store batches
    for generated_batch in tqdm(generate_patch(queue, batch_size=batch), desc="Classifying"):
        preds = classify_batch([item["text"] for item in generated_batch])

        # Store record in database
        for item, (sentiment, sentiment_score, top_category, cat_scores) in zip(generated_batch,
                                                                                preds):
            record = {
                "url": item["url"],
                "url_Timestamp": item["timestamp"],
                "text": item["text"],
                "category": top_category,
                "sentiment": sentiment,
                "sentiment_score": sentiment_score,
            }
            record.update(cat_scores)
            insert_record(cur, record)
        conn.commit()
        record_count += len(generated_batch)

    for w in workers:
        w.join()

    return record_count


def process_warc_files(warc_file_path, conn):
    """
    Iterate over the WARC files path list.
    """
    cur = conn.cursor()

    start_time = time.time()
    # Process the WARC file
    record_count = process_warc_records(warc_file_path, cur, conn)
    elapsed = int(time.time() - start_time)
    h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
    logging.info("Processed %d records in %dh %dm %ds", record_count, h, m, s)

    if record_count:
        # Add processed WARC file path to processed_paths.txt
        processed_file_log = "processed_paths.txt"
        with open(processed_file_log, 'a', encoding='utf-8') as f:
            f.write(os.path.basename(warc_file_path) + '\n')
        logging.info("Added %s to processed_paths.txt", warc_file_path)
        # Remove processed warc_file_path line from paths.txt after processing is completed
        paths_file = "paths.txt"
        if os.path.exists(paths_file):
            try:
                lines = []
                with open(paths_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                updated_lines = [line for line in lines
                                 if os.path.basename(line.strip()) != os.path.basename(warc_file_path)]
                with open(paths_file, 'w', encoding='utf-8') as f:
                    f.writelines(updated_lines)
                    logging.info("Removed %s from %s", warc_file_path, paths_file)
            except Exception as e:
                logging.error("Error updating %s: %s", paths_file, e)
        else:
            logging.info("%s not found, skipping removal", paths_file)
        # Delete the WARC file after saving to database and adding its name to processed list
        try:
            os.remove(warc_file_path)
            logging.info("Deleted %s after processing", warc_file_path)
        except OSError as e:
            logging.error("Error deleting %s: %s", warc_file_path, e)
