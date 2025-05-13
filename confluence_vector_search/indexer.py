"""
Handles extraction, embedding, and FAISS
"""
import requests
from bs4 import BeautifulSoup
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from config import *

# Load tokenizer/model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

def clean_html(html):
    return BeautifulSoup(html, "html.parser").get_text(separator="\n")

def chunk_text(text, max_words=100, overlap=20):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words - overlap)]

def fetch_pages():
    auth = (API_USERNAME, API_TOKEN)
    headers = {"Accept": "application/json"}
    pages, start, limit = [], 0, 50

    while True:
        url = (f"{CONFLUENCE_BASE_URL}/rest/api/content"
               f"?type=page&spaceKey={SPACE_KEY}&expand=body.storage"
               f"&limit={limit}&start={start}")
        res = requests.get(url, headers=headers, auth=auth)
        res.raise_for_status()
        data = res.json()
        pages.extend(data.get("results", []))
        if len(data["results"]) < limit or len(pages) >= MAX_PAGES:
            break
        start += limit
    return pages

def build_index():
    print("Fetching Confluence pages...")
    pages = fetch_pages()
    texts, meta = [], []

    for page in pages:
        title = page["title"]
        pid = page["id"]
        url = f"{CONFLUENCE_BASE_URL}/pages/viewpage.action?pageId={pid}"
        content = clean_html(page["body"]["storage"]["value"])
        for i, chunk in enumerate(chunk_text(content)):
            texts.append(chunk)
            meta.append({"title": title, "url": url, "chunk_id": i})

    print(f"Embedding {len(texts)} chunks...")
    vectors = np.array([get_embedding(t) for t in texts]).astype("float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, texts, meta
