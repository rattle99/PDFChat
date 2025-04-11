# PDFChat

This repository contains a Retrieval-Augmented Generation (RAG) based chatbot. The project utilizes a vector database for document retrieval, an embedding model for vectorizing text, and a large language model (LLM) for generation. The following instructions will help you set up the project on your local machine.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation & Set Up](#Installation-&-Set-Up)
  - [1. Set Up Model Hosting with Ollama](#1-set-up-model-hosting-with-ollama)
  - [2. Configuration](#2-configuration)
  - [3. Create Guides Folder](#3-create-guides-folder)
  - [4. Set Up Python Virtual Environment](#4-set-up-python-virtual-environment)
  - [5. Set Up Weaviate](#5-set-up-weaviate)
  - [6. Run Document Parser](#6-run-document-parser)
  - [7. Launch Chatbot UI](#7-launch-chatbot-ui)
- [Environment Variables](#environment-variables)
- [Docker Compose for Weaviate](#docker-compose-for-weaviate)
- [Usage](#usage)

---

## Prerequisites

- [Ollama](https://ollama.ai/) - for hosting models via a REST API.
- Python 3.12.8 installed.
- Docker and Docker Compose installed.

---

## Installation & Set Up

### 1. Set Up Model Hosting with Ollama

Install and configure [Ollama](https://ollama.ai/) following the official instructions. Once installed:

- **Download the Embedding Model**:  
  Download the `minilm` model for embeddings.

- **Download the Chat Generation Model**:  
  Download a lightweight generation model (e.g., `phi4 mini` or similar, as per your requirements).

Ensure that both models are available and accessible via the REST API endpoints specified later in the configuration.

---

### 2. Configuration

Edit the `.env` file in the root directory of your project.

---


## 3. Create Guides Folder

Create a folder named `Guides` in the root directory. Place all your PDF documents inside this folder. These documents will be used by the document parser to create embeddings and store them in the vector database.

```bash
mkdir Guides
```

---

## 4. Set Up Python Virtual Environment

Create and activate a new Python virtual environment, then install the necessary dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Linux/Mac)
source venv/bin/activate

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Set Up Weaviate

This project uses Weaviate for storing document embeddings. Use the provided Docker Compose file to set up Weaviate:

```bash
docker-compose up -d
```

This command will start Weaviate on port `8080`.

---

## 6. Run Document Parser

Run the document parser script to chunk PDFs from the `Guides` folder, generate embeddings using the embedding model, and store them in your vector database:

```bash
python docparser.py
```

Ensure that the document parser is properly configured to read from the `Guides` folder and use the vector store directory specified in the `.env` file.

---

## 7. Launch Chatbot UI

Finally, launch the chatbot UI by running:

```bash
python demo_experiment.py
```

This will start the UI in your browser, allowing you to interact with the chatbot. Ask questions and explore its capabilities.

---

## Environment Variables

The project uses several environment variables to control its configuration. Set these in your `.env` file:

- `COLLECTION_NAME`: Name of the collection in your vector database.
- `PERSIST_DIRECTORY`: Directory where the vector store is persisted.
- `LLM_URL`: URL endpoint for the chat generation model.
- `EMBED_MODEL_URL`: URL endpoint for the embedding model.
- `EMBEDDING_MODEL`: Model identifier for the embedding model (e.g., `all-minilm`).
- `CHAT_MODEL`: Model identifier for the generation model (e.g., `llama3.1:8b`).
- `RETRIEVAL`: Retrieval method type (options: `keyword`, `vector`, or `hybrid`).
- `LOG_LEVEL`: Log level for debugging (e.g., `DEBUG`, `INFO`).

Ensure these variables are correctly set for the project to run smoothly.

---

## Docker Compose for Weaviate

The provided `docker-compose.yml` file configures and launches a Weaviate instance with anonymous access enabled. This setup is necessary for storing and retrieving document embeddings:

---

## Setup (Summary)

1. **Install and configure Ollama**: Host models for embeddings and generation via a REST API.
2. **Configure Environment**: Set up the `.env` file with correct endpoints and model identifiers.
3. **Prepare Documents**: Place PDFs in the `Guides` directory.
4. **Virtual Environment**: Create and activate the virtual environment, then install dependencies.
5. **Weaviate Setup**: Start the Weaviate database with Docker Compose.
6. **Document Parsing**: Run the document parser to embed and store documents.
7. **Launch Chatbot**: Execute `demo_experiment.py` to start the chatbot UI in your browser.

---

