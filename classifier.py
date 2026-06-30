"""
Load and save classifier model and tokenizer.
Set up device configuration for GPU or CPU usage.
"""

import warnings
import logging
import ssl
import torch


ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings("ignore", category=FutureWarning)

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    gpu_names = [f"GPU {i}: {torch.cuda.get_device_name(i)}" for i in range(gpu_count)]
    logging.info("%d GPU(s) allocated: %s", gpu_count, ', '.join(gpu_names))
    device = torch.device("cuda")
else:
    logging.info("CPUs are allocated for use")
    device = torch.device("cpu")


def get_classifier():
    """
    Load the model and tokenizer and save them locally if not already saved.
    """
    import os
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "FacebookAI/roberta-large-mnli"
    local_dir = model_name.replace("/", "_")

    if not os.path.exists(local_dir):
        _model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model.save_pretrained(local_dir)
        _tokenizer.save_pretrained(local_dir)

    logging.info("Loading model to %s...", device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    import time
    t0 = time.time()
    if device.type == "cuda":
        model = AutoModelForSequenceClassification.from_pretrained(local_dir,
                                                                   torch_dtype=dtype,
                                                                   device_map="cuda",
                                                                   max_memory="0.7GB")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(local_dir,
                                                                   torch_dtype=dtype)
    logging.info("Model loaded in %.1fs", time.time() - t0)
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    return model, tokenizer, device
