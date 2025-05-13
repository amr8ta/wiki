import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
import time

# === CONFIG ===
CONFLUENCE_BASE_URL = "https://yourcompany.atlassian.net/wiki"
API_USERNAME = "your.email@company.com"
API_TOKEN = "your_api_token"
SPACE_KEY = "SPACEKEY"
MAX_PAGES = 1000  # Adjust as needed

# === AUTH ===
auth = (API_USERNAME, API_TOKEN)
headers = {"Accept": "application/json"}

# === HELPER: Extract plain text from Confluence storage format (HTML) ===
def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")

# === HELPER: Chunk into small segments (~500 tokens or 1000 words total with overlap) ===
def chunk_text(text, max_sentences=5, overlap=2):
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), max_sentences - overlap):
        chunk = " ".join(sentences[i:i + max_sentences])
        chunks.append(chunk)
    return chunks

# === STEP 1: Fetch all pages from a Confluence space ===
def fetch_all_pages():
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
        time.sleep(0.5)  # Avoid rate-limiting

    return pages

# === STEP 2–4: Clean, Chunk, Return ===
def extract_chunks_from_confluence():
    pages = fetch_all_pages()
    all_chunks = []

    for page in pages:
        title = page["title"]
        page_id = page["id"]
        content_html = page["body"]["storage"]["value"]
        plain_text = clean_html(content_html)
        chunks = chunk_text(plain_text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "page_id": page_id,
                "title": title,
                "chunk_index": i,
                "text": chunk,
                "url": f"{CONFLUENCE_BASE_URL}/pages/viewpage.action?pageId={page_id}"
            })

    return all_chunks

# === USAGE ===
if __name__ == "__main__":
    chunks = extract_chunks_from_confluence()
    print(f"Extracted {len(chunks)} chunks from Confluence.")
    for c in chunks[:3]:  # Show first few chunks as example
        print("\n---")
        print(f"[{c['title']}] ({c['url']}) - Chunk {c['chunk_index']}")
        print(c['text'])
