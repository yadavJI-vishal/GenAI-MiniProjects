# 🤖 RAG Chatbot API

A powerful **Retrieval-Augmented Generation (RAG)** chatbot API built with FastAPI that allows users to upload PDF documents and ask questions about them using AI. Powered by Mistral-7B and LlamaIndex.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.5-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- 📄 **PDF Upload & Processing** - Upload any PDF document
- 🤖 **AI-Powered Q&A** - Ask questions and get intelligent answers
- 🚀 **FastAPI Backend** - High-performance REST API
- 🌐 **Ngrok Integration** - Public URL for easy sharing
- 💾 **Vector Search** - Efficient semantic search using embeddings
- ⚡ **4-bit Quantization** - Optimized model loading with bitsandbytes
- 🔒 **Session Management** - Isolated user sessions

---

## 🛠️ Tech Stack

- **Framework:** FastAPI
- **LLM:** Mistral-7B-Instruct-v0.3 (4-bit quantized)
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **Vector Store:** LlamaIndex
- **PDF Processing:** PyPDF2
- **Deployment:** Ngrok (for public URLs)

---

## 📋 Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended for faster inference)
- 16GB+ RAM recommended
- Ngrok account (free) - [Get your token here](https://dashboard.ngrok.com/get-started/your-authtoken)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Ngrok Token

Open `app.py` and replace the ngrok token:

```python
NGROK_TOKEN = "your_ngrok_token_here"  # Line ~280
```

Get your free token from: https://dashboard.ngrok.com/get-started/your-authtoken

### 3. Update PDF Path

In `app.py`, update the PDF path to your document (Line ~50):

```python
PDF_PATH = "path/to/your/document.pdf"
```

### 4. Run the Application

```bash
python app.py
```

The server will start and display:
```
🚀 FastAPI server is running!
📡 Public URL: https://xxxxx.ngrok-free.dev
📚 API Docs: https://xxxxx.ngrok-free.dev/docs
```

---

## 📖 API Documentation

Once the server is running, visit the interactive API documentation:

- **Swagger UI:** `http://your-url/docs`
- **ReDoc:** `http://your-url/redoc`

### Available Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### 2. Query the RAG System
```http
POST /query
```

**Request Body:**
```json
{
  "question": "What are the main topics in this document?",
  "top_k": 3
}
```

**Response:**
```json
{
  "answer": "The main topics discussed are...",
  "processing_time": 5.43
}
```

#### 3. Reinitialize with New PDF
```http
POST /reinitialize?pdf_path=/path/to/new.pdf
```

---

## 💻 Usage Examples

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic of this document?",
    "top_k": 3
  }'
```

### Using Python Requests

```python
import requests

# Base URL (use your ngrok URL)
base_url = "https://xxxxx.ngrok-free.dev"

# Query the chatbot
response = requests.post(
    f"{base_url}/query",
    json={
        "question": "Summarize the key points",
        "top_k": 3
    }
)

print(response.json())
```
---

## 🏗️ Project Structure

```
rag-chatbot-api/
│
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
│
├── data/                 # (Optional) Store your PDFs here
│   └── your-document.pdf
│
└── venv/                 # Virtual environment (created after setup)
```

---

## ⚙️ Configuration Options

### Model Settings (in `app.py`)

```python
# Chunk settings for document processing
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Number of relevant chunks to retrieve
TOP_K = 3

# LLM Configuration
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
max_new_tokens = 512
temperature = 0.1
```

### Quantization (for GPU memory optimization)

The model uses 4-bit quantization by default. To adjust:

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
```

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
Solution: Reduce max_new_tokens or use a smaller model
```

**2. Ngrok Connection Errors**
```bash
# Kill existing ngrok processes
pkill ngrok

# Restart the application
python app.py
```

**3. PDF Extraction Issues**
```
Solution: Ensure PDF is text-based (not scanned images)
Consider using OCR for scanned documents
```

**4. Port Already in Use**
```python
# Change port in app.py
run_server(ngrok_token=NGROK_TOKEN, port=8001)
```

---

## 🔧 Advanced Usage

### Running Without Ngrok (Local Only)

Comment out ngrok lines in `app.py`:

```python
# public_url = ngrok.connect(port)
print(f"Server running at: http://localhost:{port}")
```

### Using Different LLM Models

Replace the model_id in `app.py`:

```python
# Example: Using a different Mistral variant
model_id = "mistralai/Mistral-7B-v0.1"

# Example: Using Llama 2
model_id = "meta-llama/Llama-2-7b-chat-hf"
```

---

## 📊 Performance Benchmarks

| Metric | Value |
|--------|-------|
| Average Query Time | 3-8 seconds |
| PDF Processing Time | 10-30 seconds |
| Memory Usage (4-bit) | ~6-8 GB |
| Concurrent Users | Up to 5 (depends on hardware) |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [LlamaIndex](https://www.llamaindex.ai/) - RAG framework
- [Hugging Face](https://huggingface.co/) - Model hosting
- [Mistral AI](https://mistral.ai/) - LLM provider
- [Ngrok](https://ngrok.com/) - Tunneling service

---

**⭐ If you find this project helpful, please give it a star!**
