import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
import time
import json
from tqdm import tqdm
import os

# === CONFIG ===
CONFLUENCE_BASE_URL = "https://yourcompany.atlassian.net/wiki"
API_USERNAME = "your.email@company.com"
API_TOKEN = "your_api_token"
MAX_PAGES_PER_SPACE = 1000
CHUNK_OUTPUT_FILE = "confluence_chunks.json"

# === INIT ===
auth = (API_USERNAME, API_TOKEN)
headers = {"Accept": "application/json"}
nltk.download('punkt')

# === CLEAN HTML TO PLAIN TEXT ===
def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")

# === CHUNK TEXT ===
def chunk_text(text, max_sentences=5, overlap=2):
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), max_sentences - overlap):
        chunk = " ".join(sentences[i:i + max_sentences])
        chunks.append(chunk)
    return chunks

# === FETCH ALL SPACES ===
def fetch_all_spaces():
    spaces = []
    start = 0
    limit = 50
    while True:
        url = f"{CONFLUENCE_BASE_URL}/rest/api/space?limit={limit}&start={start}"
        response = requests.get(url, headers=headers, auth=auth)
        response.raise_for_status()
        data = response.json()
        spaces.extend(data.get("results", []))
        if len(spaces) >= data.get("size", 0):
            break
        start += limit
    return spaces

# === FETCH PAGES FOR ONE SPACE ===
def fetch_pages_for_space(space_key):
    pages = []
    start = 0
    limit = 50
    while True:
        url = (
            f"{CONFLUENCE_BASE_URL}/rest/api/content"
            f"?type=page&spaceKey={space_key}&expand=body.storage"
            f"&limit={limit}&start={start}"
        )
        response = requests.get(url, headers=headers, auth=auth)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        pages.extend(results)
        if len(results) < limit or len(pages) >= MAX_PAGES_PER_SPACE:
            break
        start += limit
        time.sleep(0.5)
    return pages

# === PROCESS ALL SPACES AND PAGES ===
def process_all_spaces():
    all_chunks = []
    spaces = fetch_all_spaces()
    print(f"Found {len(spaces)} spaces.")

    for space in tqdm(spaces, desc="Spaces"):
        space_key = space["key"]
        space_name = space["name"]
        print(f"\nProcessing space: {space_name} ({space_key})")

        try:
            pages = fetch_pages_for_space(space_key)
        except Exception as e:
            print(f"⚠️  Failed to fetch pages from space {space_key}: {e}")
            continue

        for page in pages:
            title = page["title"]
            page_id = page["id"]
            url = f"{CONFLUENCE_BASE_URL}/pages/viewpage.action?pageId={page_id}"
            content_html = page.get("body", {}).get("storage", {}).get("value", "")
            if not content_html:
                continue
            plain_text = clean_html(content_html)
            chunks = chunk_text(plain_text)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "space_key": space_key,
                    "space_name": space_name,
                    "page_id": page_id,
                    "title": title,
                    "chunk_index": i,
                    "text": chunk,
                    "url": url
                })

    return all_chunks

# === SAVE TO JSON ===
def save_chunks_to_json(chunks, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved {len(chunks)} chunks to {filename}")

# === MAIN ===
if __name__ == "__main__":
    chunks = process_all_spaces()
    save_chunks_to_json(chunks, CHUNK_OUTPUT_FILE)
