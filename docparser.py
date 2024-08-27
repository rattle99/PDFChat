import os
import re
from glob import glob

import nltk
import pdfplumber
import weaviate
from dotenv import dotenv_values
from tqdm import tqdm
from weaviate.classes.config import DataType, Property
from weaviate.util import generate_uuid5

from utils import compute_sha256, get_embedding

nltk.download("punkt")

config = dotenv_values(".env")
client = weaviate.connect_to_local()
if not client.collections.exists(config["COLLECTION_NAME"]):
    print(
        f'Collection does not exist, creating collection {config["COLLECTION_NAME"]}.'
    )
    client.collections.create(
        name=config["COLLECTION_NAME"],
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="doc_hash", data_type=DataType.TEXT),
            Property(name="file_name", data_type=DataType.TEXT),
        ],
    )
    print("Done")
collection = client.collections.get(name=config["COLLECTION_NAME"])


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\.{2,}", ".", text)
    return text.strip()


def convert_to_dict_format(table):
    result = {}
    current_key = None

    for row in table:
        if len(row) > 1:
            if row[0]:
                current_key = row[0]
            combined_value = ":".join(
                [str(cell) if cell is not None else "" for cell in row[1:]]
            )
            result[current_key] = combined_value
        elif len(row) == 1 and row[0]:  # Handle single-element rows
            current_key = row[0]
            result[current_key] = ""

    return result


def format_dict_as_string(dictionary):
    return "\n".join([f"'{key}': '{value}'" for key, value in dictionary.items()])


def table_to_string(table):
    converted_dict = convert_to_dict_format(table)
    formatted_string = format_dict_as_string(converted_dict)

    return formatted_string


def extract_pdf_content(pdf_path):
    content = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            tables = page.extract_tables()
            page_content = text if text else ""

            for table in tables:
                table_str = table_to_string(table)
                page_content += "\n\n" + table_str + "\n\n"

            content += page_content + "\n\n"

    return content


def word_count_chunk_sentences(text, chunk_size=150):
    sentences = [
        [sentence.strip(), pair[1], pair[2]]
        for pair in text
        for sentence in nltk.sent_tokenize(pair[0])
    ]
    chunks = []
    current_chunk = []
    word_count = 0

    for i in range(len(sentences)):
        words = sentences[i][0].split()
        if word_count + len(words) <= chunk_size:
            current_chunk.extend(words)
            word_count += len(words)
        else:
            chunks.append([" ".join(current_chunk), sentences[i][1], sentences[i][2]])
            current_chunk = sentences[i - 1][0].split() + words
            word_count = len(current_chunk)

    if current_chunk:
        chunks.append([" ".join(current_chunk), sentences[-1][1], sentences[-1][2]])
    return chunks


PDF_DIRECTORY = "./Guides"
# PDF_DIRECTORY = "./PDF"
pdf_paths = glob(PDF_DIRECTORY + "/**/*.pdf", recursive=True)
all_texts = []
for pdf_path in tqdm(pdf_paths, desc="Chunking Documents ", colour="#bd4ced"):
    file_name = os.path.basename(pdf_path)
    filehash = compute_sha256(pdf_path)
    extracted_text = extract_pdf_content(pdf_path)
    y = clean_text(extracted_text)
    all_texts.append([y, filehash, file_name])


documents = word_count_chunk_sentences(all_texts)

fails = 0
for idx, document in enumerate(
    tqdm(documents, desc="Parsing Chunks ", colour="#bd4ced")
):
    embedding = get_embedding(document[0])
    data_object = {
        "content": document[0],
        "doc_hash": document[1],
        "file_name": document[2],
    }
    try:
        collection.data.insert(
            properties=data_object,
            uuid=generate_uuid5(data_object['content']),
            vector=embedding,
        )
    except Exception as e:
        # Handle the specific exception
        if "already exists" in str(e):
            # Optional: You can take additional actions here, like updating the object instead.
            fails += 1
        else:
            print("An unexpected error occurred.")
            print(e)

client.close()
print(f"Failed to add {fails} chunks.")
