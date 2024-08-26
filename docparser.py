import re
from glob import glob

import nltk
import PyPDF2
import weaviate
from dotenv import dotenv_values
from tqdm import tqdm
from weaviate.classes.config import DataType, Property
from weaviate.util import generate_uuid5

from utils import get_embedding

# from sklearn.metrics.pairwise import cosine_similarity

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
        ],
    )
    print("Done")
collection = client.collections.get(name=config["COLLECTION_NAME"])


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\.{2,}", ".", text)
    return text.strip()


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""

        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text


def word_count_chunk_sentences(text, chunk_size=100, overlap=10):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    word_count = 0

    for sentence in sentences:
        words = sentence.split()
        if word_count + len(words) <= chunk_size:
            current_chunk.extend(words)
            word_count += len(words)
        else:
            chunks.append(" ".join(current_chunk))
            overlap_words = current_chunk[-overlap:]
            current_chunk = overlap_words + words
            word_count = len(current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


PDF_DIRECTORY = "./Guides"
pdf_paths = glob(PDF_DIRECTORY + "/**/*.pdf", recursive=True)
all_texts = ""
for pdf_path in tqdm(pdf_paths, desc="Chunking Documents ", colour="#bd4ced"):
    extracted_text = extract_text_from_pdf(pdf_path)
    y = clean_text(extracted_text)
    all_texts += y


documents = word_count_chunk_sentences(all_texts)

fails = 0
for idx, document in enumerate(
    tqdm(documents, desc="Parsing Chunks ", colour="#bd4ced")
):
    embedding = get_embedding(document)
    data_object = {"content": document}
    try:
        collection.data.insert(
            properties=data_object,
            uuid=generate_uuid5(data_object),
            vector=embedding,
        )
    except Exception as e:
        # Handle the specific exception
        # print(f"Failed to add object: {e}")
        if "already exists" in str(e):
            # print("The object with this ID already exists.")
            # Optional: You can take additional actions here, like updating the object instead.
            fails += 1
        else:
            print("An unexpected error occurred.")
            print(e)

client.close()
print(f"Failed to add {fails} chunks.")

