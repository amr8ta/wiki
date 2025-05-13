from flask import Flask, render_template, request
from indexer import build_index, get_embedding
import numpy as np

app = Flask(__name__)
index, chunks, metadata = build_index()

@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    if request.method == "POST":
        query = request.form["query"]
        query_vec = get_embedding(query).astype("float32").reshape(1, -1)
        distances, indices = index.search(query_vec, 5)

        for idx in indices[0]:
            results.append({
                "title": metadata[idx]["title"],
                "url": metadata[idx]["url"],
                "text": chunks[idx]
            })

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
