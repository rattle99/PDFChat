import chromadb
import requests
from tqdm import tqdm
import PyPDF2
import re
import nltk
from dotenv import dotenv_values
from utils import get_embedding
from glob import glob
# from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")

config = dotenv_values(".env")
client = chromadb.PersistentClient(path=config["PERSIST_DIRECTORY"])
collection = client.get_or_create_collection(name=config["COLLECTION_NAME"])


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


# Example usage
PDF_DIRECTORY = "./PDF"
pdf_paths = glob("./Guides/**/*.pdf", recursive=True)
all_texts = ""
for pdf_path in tqdm(pdf_paths, desc="Chunking Documents ", colour="#bd4ced"):
    extracted_text = extract_text_from_pdf(pdf_path)
    y = clean_text(extracted_text)
    all_texts += y


documents = word_count_chunk_sentences(all_texts)


for idx, document in enumerate(
    tqdm(documents, desc="Parsing Chunks ", colour="#bd4ced")
):
    embedding = get_embedding(document)
    save_id = f"id{idx}"
    collection.upsert(documents=document, ids=save_id, embeddings=embedding)


## Semantic Chunking Below

# def semantic_chunking_with_word_limit(text, chunk_size=100, overlap=10, breakpoint_percentile_threshold=95):
# sentences = nltk.sent_tokenize(text)
# sentences = [{'sentence': x, 'index': i} for i, x in enumerate(sentences)]
# buffer_size = 1
#
# for i in range(len(sentences)):
# combined_sentence = ''
# for j in range(i - buffer_size, i):
# if j >= 0:
# combined_sentence += sentences[j]['sentence'] + ' '
# combined_sentence += sentences[i]['sentence']
# for j in range(i + 1, i + 1 + buffer_size):
# if j < len(sentences):
# combined_sentence += ' ' + sentences[j]['sentence']
# sentences[i]['combined_sentence'] = combined_sentence
#
# embeddings1 = [get_embedding([x['combined_sentence']) for x in sentences]]
# for i, sentence in enumerate(sentences):
# sentence['combined_sentence_embedding'] = embeddings1[i]
#
# distances = []
# for i in range(len(sentences) - 1):
# embedding_current = sentences[i]['combined_sentence_embedding']
# embedding_next = sentences[i + 1]['combined_sentence_embedding']
# similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
# distance = 1 - similarity
# distances.append(distance)
# sentences[i]['distance_to_next'] = distance
#
# breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
# indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]
#
# start_index = 0
# chunks = []
# current_chunk = []
# current_word_count = 0
#
# for index in indices_above_thresh:
# end_index = index
# group = sentences[start_index:end_index + 1]
# combined_text = ' '.join([d['sentence'] for d in group])
#
# # Split combined_text into words and ensure each chunk has max chunk_size words
# words = combined_text.split()
# for word in words:
# if current_word_count < chunk_size:
# current_chunk.append(word)
# current_word_count += 1
# else:
# chunks.append(' '.join(current_chunk))
# current_chunk = current_chunk[-overlap:]  # Overlap
# current_chunk.append(word)
# current_word_count = len(current_chunk)
#
# start_index = index + 1
#
# # The last group, if any sentences remain
# if current_chunk:
# chunks.append(' '.join(current_chunk))
#
# return chunks
#
#
# chunks = semantic_chunking_with_word_limit(y, chunk_size=300, overlap=10)
