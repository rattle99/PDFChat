import hashlib

import requests
from dotenv import dotenv_values

config = dotenv_values(".env")


def get_embedding(text):
    # Parameters for the POST request
    params = {
        "model": config["EMBEDDING_MODEL"],  # Model to use
        "prompt": text,  # Text to prompt the model with
    }
    # Add more parameters as needed

    # Making POST request to Embedding Model
    response = requests.post(config["EMBED_MODEL_URL"], json=params)

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
    sha256_hash = hashlib.sha256()

    # Read the file in chunks to avoid loading the entire file into memory
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()
