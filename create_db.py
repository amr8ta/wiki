import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import re

# === CONFIG ===
CONFLUENCE_BASE_URL = "https://yourcompany.atlassian.net/wiki"
API_USERNAME = "your.email@company.com"
API_TOKEN = "your_api_token"
SPACE_KEY = "SPACEKEY"
MAX_PAGES = 1000

# === AUTH ===
auth = (API_USERNAME, API_TOKEN)
headers = {"Accept": "application/json"}

# === EMBEDDING MODEL ===
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
    return embeddings.squeeze().numpy()

# === CLEANING ===
def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")

# === CHUNKING ===
def chunk_text(text, max_words=100, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        if chunk:
            chunks.append(chunk)
    return chunks

# === FETCH FROM CONFLUENCE ===
def fetch_pages():
    pages = []
    start = 0
    limit = 50

    while True:
        url = (
            f"{CONFLUENCE_BASE_URL}/rest/api/content"
            f"?type=page&spaceKey={SPACE_KEY}&expand=body.storage"
            f"&limit={limit}&start={start}"
        )
        response = requests.get(url, headers=headers, auth=auth)
        response.raise_for_status()
        data = response.json()
        pages.extend(data.get("results", []))

        if "size" in data and (start + limit >= MAX_PAGES or start + limit >= data["size"]):
            break
        start += limit

    return pages

# === PROCESS TO VECTOR DB ===
def build_vector_db():
    print("Fetching pages...")
    pages = fetch_pages()
    all_text_chunks = []
    metadata = []

    for page in pages:
        title = page["title"]
        page_id = page["id"]
        url = f"{CONFLUENCE_BASE_URL}/pages/viewpage.action?pageId={page_id}"
        html = page["body"]["storage"]["value"]
        text = clean_html(html)
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            all_text_chunks.append(chunk)
            metadata.append({"title": title, "chunk_id": i, "url": url})

    print(f"Embedding {len(all_text_chunks)} chunks...")
    embeddings = np.array([get_embedding(chunk) for chunk in all_text_chunks]).astype("float32")

    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, all_text_chunks, metadata

# === SEARCH INTERFACE ===
def search(query, index, chunks, metadata, k=5):
    print(f"Searching for: {query}")
    query_vec = get_embedding(query).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vec, k)

    results = []
    for i in indices[0]:
        results.append({
            "text": chunks[i],
            "title": metadata[i]["title"],
            "url": metadata[i]["url"]
        })

    return results

# === MAIN ===
if __name__ == "__main__":
    index, chunks, metadata = build_vector_db()

    while True:
        query = input("\nEnter search query (or 'exit'): ")
        if query.lower() == "exit":
            break
        results = search(query, index, chunks, metadata)
        for i, res in enumerate(results, 1):
            print(f"\nResult {i}: {res['title']} — {res['url']}\n{res['text']}")
