# 📘 Large Document Q&A using LlamaIndex

A beginner-friendly **Large Document Question Answering (Q&A)** system built using **LlamaIndex** that demonstrates core concepts such as **Nodes**, **Chunking**, **Embeddings**, **Vector Indexing**, and **Query Engines** — without using any paid APIs.

This project focuses on **retrieval logic**, making it ideal for **learning and interviews**.

---

## 🚀 Features

- 📄 **Load large text documents** from a directory
- ✂️ **Chunk documents** into manageable pieces (Nodes)
- 🔢 **Convert text into numerical vectors** using embeddings
- 🧠 **Store vectors** in a vector index
- 🔍 **Retrieve the most relevant chunks** for a user query
- ❌ **No OpenAI / Paid API required** (fully offline)

---

## 🧠 Core Concepts Explained

### 1️⃣ Documents
Raw text files loaded from the `data/` directory.

### 2️⃣ Chunking
Large documents are split into smaller pieces to improve retrieval accuracy.

```python
SentenceSplitter(chunk_size=512, chunk_overlap=50)
```

### 3️⃣ Nodes
A **Node** is a container that holds:
- Text chunk
- Metadata
- Embedding vector (numerical representation)

Think of it as:
```python
Node = { text + metadata + vector }
```

### 4️⃣ Embeddings
Embeddings convert text into numbers (vectors) so that similarity can be calculated mathematically.

In this project we use:
```python
MockEmbedding(embed_dim=384)
```

✅ No internet  
✅ No API keys  
✅ Ideal for learning and testing

### 5️⃣ Vector Store Index
Stores all node embeddings and enables fast similarity search.

```python
VectorStoreIndex(nodes)
```

### 6️⃣ Query Engine
Handles:
- Converting user query → vector
- Finding similar nodes
- Returning relevant chunks

```python
index.as_query_engine(similarity_top_k=3)
```

---

## 📂 Project Structure

```
large-doc-qa-llamaindex/
│
├── data/
│   └── sample.txt          # Large document
│
├── app.py                  # Main application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🛠 Installation

### 1️⃣ Create Virtual Environment

```bash
conda create -n genai python=3.10
conda activate genai
```

### 2️⃣ Install Dependencies

```bash
pip install llama-index
```

Or using requirements file:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

1. **Add your `.txt` files** inside the `data/` folder

2. **Run the application:**

```bash
python app.py
```

3. **Ask questions** in the terminal:

```
Ask: What is LlamaIndex?
```

4. **Type `exit` to stop.**

---

## 📌 Sample Output

```
Retrieved Nodes:

--- Node 1 ---
LlamaIndex is a data framework designed to connect LLMs with external data...

--- Node 2 ---
It provides tools for data ingestion, indexing, and querying...

--- Node 3 ---
The framework supports various data sources including documents, APIs...
```

---

## 🔐 No API Keys Required

- ❌ OpenAI not used
- ❌ Hugging Face login not required
- ✅ **Fully offline project**

This makes it perfect for:
- 🎓 Learning RAG concepts
- 💼 Interview preparation
- 🧪 Testing retrieval logic
- 🔒 Privacy-focused applications

---

## 🎯 Learning Outcomes

After completing this project, you will understand:

✅ Difference between **documents, nodes, and vectors**  
✅ How **vector similarity search** works  
✅ How **query engines** retrieve relevant context  
✅ The role of **chunking** in information retrieval  
✅ How **embeddings** enable semantic search  

---

## 🔧 Configuration

### Adjust Chunk Size

Modify in `app.py`:

```python
text_splitter = SentenceSplitter(
    chunk_size=512,      # Increase for longer chunks (256-1024)
    chunk_overlap=50     # Adjust overlap (20-100)
)
```

### Change Top-K Results

Modify the number of retrieved chunks:

```python
query_engine = index.as_query_engine(
    similarity_top_k=3   # Change to retrieve more/fewer chunks (1-10)
)
```

### Use Different Embeddings

Replace `MockEmbedding` with real embeddings:

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

---

## 📦 requirements.txt

```
llama-index>=0.9.0
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'llama_index'"

**Solution:**
```bash
pip install llama-index
```

### Issue: "Empty data folder"

**Solution:**
- Ensure you have `.txt` files in the `data/` directory
- Check file permissions

### Issue: "No relevant chunks found"

**Solution:**
- Reduce `similarity_top_k` value
- Ensure your document contains relevant information
- Try rephrasing your query

---

## 🔮 Future Improvements

- [ ] Add **PDF document support**
- [ ] Use **HuggingFace** or **OpenAI** embeddings
- [ ] Integrate **FAISS** vector store
- [ ] Add **local LLM (Ollama)** for answer generation
- [ ] Build a **Streamlit UI**
- [ ] Add **multi-document support**
- [ ] Implement **query history**
- [ ] Add **evaluation metrics** (BLEU, ROUGE)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📚 Learning Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Understanding RAG Systems](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Vector Embeddings Explained](https://www.deeplearning.ai/short-courses/google-cloud-vertex-ai/)
- [Chunking Strategies](https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Vishal Yadav**  
Aspiring Data Scientist / GenAI Engineer  
📍 Pune, India

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/vishal-yadav-294138203/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/yadavJI-vishal)
[![Email](https://img.shields.io/badge/Email-Contact-red)](mailto:vy5068@gmail.com)

---

## ⭐ If you found this useful

Give this repository a ⭐ and share it on LinkedIn!

**Tags:** `#LlamaIndex` `#RAG` `#GenAI` `#VectorSearch` `#NLP` `#MachineLearning` `#Python`

---

