import hashlib
import logging
from logging.handlers import TimedRotatingFileHandler

import requests
from dotenv import dotenv_values

CONFIG = dotenv_values(".env")


def setup_logger(name):
    """
    Sets up a logger with the specified name, configuration, and handlers.

    Parameters:
        name (str): The name of the logger, typically the module's __name__.

    Returns:
        logger (logging.Logger): Configured logger instance.
    """

    # Create or get a logger with the specified name.
    logger = logging.getLogger(name)
    log_level = CONFIG["LOG_LEVEL"]

    # Set the logger's level based on the log_level value.
    if log_level == "DEBUG":
        logger.setLevel("DEBUG")
    elif log_level == "INFO":
        logger.setLevel("INFO")
    else:
        raise ValueError(f"Invalid {log_level}.")

    # Create a TimedRotatingFileHandler to manage log files.
    # - "app.log": Base filename for log files.
    # - when="W6": Rotate logs weekly, specifically on Sunday (W6).
    # - backupCount=0: Keep all log files, no deletions.
    file_handler = TimedRotatingFileHandler(
        "app.log", encoding="utf-8", when="W6", backupCount=0
    )
    logger.addHandler(file_handler)

    formatter = logging.Formatter(
        fmt="{asctime} | {name} | {levelname} | {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Apply the formatter to the handler so that all logs follow this format.
    file_handler.setFormatter(formatter)

    return logger


def get_embedding(text):
    """
    Retrieves an embedding for the given text using a specified embedding model.

    This function sends a POST request to a configured embedding model API with the provided text as input.
    The model generates an embedding, which is returned as a result if the request is successful.

    Parameters:
    text (str): The text input for which the embedding is to be generated.

    Returns:
    list or None: The embedding of the input text as a list of numbers if the request is successful.
                  Returns None if the request fails.
    """

    # Parameters for the POST request
    params = {
        "model": CONFIG["EMBEDDING_MODEL"],  # Model to use
        "prompt": text,  # Text to prompt the model with
    }
    # Add more parameters as needed

    # Making POST request to Embedding Model
    response = requests.post(CONFIG["EMBED_MODEL_URL"], json=params)

    # Check if request was successful (status code 200)
    if response.status_code == 200:
        # Parse JSON response
        results = response.json()
        # Do something with the results
        return results["embedding"]
    else:
        # Request was not successful
        print("Error:", response.status_code)
        return None


def compute_sha256(file_path):
    """
    Computes the SHA-256 hash of a file.

    This function reads a file in binary mode and computes its SHA-256 hash using the hashlib library.
    The file is read in chunks to efficiently handle large files without loading them entirely into memory.

    Parameters:
    file_path (str): The path to the file whose SHA-256 hash is to be computed.

    Returns:
    str: The hexadecimal representation of the SHA-256 hash of the file.
    """

    sha256_hash = hashlib.sha256()

    # Read the file in chunks to avoid loading the entire file into memory
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()
