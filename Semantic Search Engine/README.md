# Semantic Search Engine using Pinecone and SentenceTransformers

## Project Overview

This is a **mini project** demonstrating a **semantic search engine** built with:

* **Pinecone**: Vector database for storing and querying embeddings.
* **SentenceTransformers**: Pretrained models for generating sentence embeddings.

The project allows you to **search documents based on meaning**, not just exact keywords.

---

## Features

* Batch embeddings generation for efficiency
* Semantic search with top-K results
* Handles metadata safely
* Fully CPU-compatible
* Console-based interactive query interface

---

## Requirements

Python 3.9+ and the following packages:

```
sentence-transformers==2.2.2
pinecone==2.2.0
numpy>=1.24.0
ipython>=8.0.0
ipywidgets>=8.0.0
tqdm>=4.65.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup Instructions

1. **Pinecone API Key**

   * Sign up at [Pinecone](https://www.pinecone.io/).
   * Create a new API key.
   * Optionally, set environment variable:

   ```bash
   export PINECONE_API_KEY='YOUR_API_KEY'
   ```

2. **Index Creation**

   * Uses **ServerlessSpec** on free tier (`us-east-1` AWS region).

3. **Load Embeddings Model**

   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer("all-MiniLM-L6-v2")
   ```

4. **Insert Documents**

   * Batch encode documents and upsert into Pinecone index.

5. **Run Semantic Search**

   * Launch console app and type queries:

   ```bash
   python semantic_search.py
   ```

   * Type `exit` to quit.

---

## Example Usage

```
Query: python web framework
Results:
- Flask is a lightweight Python web framework.
- Vector databases store embeddings for similarity search.

Query: reduce model size
Results:
- Quantization reduces model size and speeds up inference.
```

---

## Notes & Tips

* Metadata keys are **case-sensitive**. Always use lowercase `text`.
* Batch embedding generation significantly **reduces upsert time**.
* On Windows, you may see HuggingFace symlink warnings; you can disable them:

```python
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
```

* For Jupyter users, install ipywidgets to avoid render errors:

```bash
pip install ipywidgets
```

---

## Future Improvements

* Web-based frontend (Flask/Streamlit)
* Persistent storage of index metadata
* Support for larger document collections with batching
* Integration with a chatbot interface

---

## Project Status

* Fully functional semantic search engine
* CPU-friendly
* Ready for sharing on GitHub / LinkedIn

---

## Author

* Vishal Yadav
* Email: [[your-email@example.com](mailto:your-email@example.com)]
* LinkedIn: [https://www.linkedin.com/in/yadavji-vishal](https://www.linkedin.com/in/yadavji-vishal)
