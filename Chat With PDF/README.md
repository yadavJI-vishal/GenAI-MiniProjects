# 📄 Chat with PDF - LangChain Project

A powerful AI-powered application that allows you to have interactive conversations with your PDF documents using LangChain, Pinecone, and Hugging Face models.

## 🌟 Features

- 📚 **PDF Document Loading** - Extract and process text from PDF files
- 🔍 **Intelligent Text Chunking** - Split documents into manageable chunks for better context
- 🧠 **Semantic Search** - Find relevant information using vector embeddings
- 💬 **Interactive Chat** - Ask questions and get accurate answers from your documents
- 🗄️ **Vector Database** - Store embeddings in Pinecone for fast retrieval
- 🤖 **AI-Powered Responses** - Generate natural language answers using Flan-T5

## 🛠️ Technologies Used

- **LangChain** (v1.2.0+) - Framework for building LLM applications
- **Pinecone** (v7.0.0+) - Vector database for semantic search
- **Hugging Face Transformers** (v4.57.0+) - NLP models
- **Sentence Transformers** (v5.2.0+) - Text embeddings
- **pypdf** (v6.5.0+) - PDF parsing
- **Python 3.10+** - Programming language

## 📋 Prerequisites

- Python 3.10 or higher
- Pinecone API account ([Sign up here](https://www.pinecone.io/))
- Git (for version control)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/chat-with-pdf.git
cd chat-with-pdf
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install langchain>=1.2.0 langchain-community>=0.4.1 langchain-huggingface
pip install langchain-text-splitters>=1.1.0
pip install pinecone>=7.0.0
pip install pypdf>=6.5.0
pip install sentence-transformers>=5.2.0
pip install transformers>=4.57.0 torch
pip install python-dotenv
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
PINECONE_API_KEY=your_pinecone_api_key_here
```

**Important:** Never commit your `.env` file to Git! It's already included in `.gitignore`.

## 📖 Usage

### Running the Notebook

1. **Start Jupyter Notebook:**

```bash
jupyter notebook
```

2. **Open `chat_pdf.ipynb`**

3. **Run cells sequentially** (Cell 1 through Cell 13)

4. **Add your PDF:**
   - Place your PDF file in the project folder
   - Update `PDF_PATH` in Cell 5 to point to your file

5. **Start chatting:**
   - Run Cell 13 to start the interactive chat
   - Type your questions
   - Type `exit` or `quit` to stop

### Example Interaction

```
You: What is the main topic of this document?

🤔 Thinking...

Bot: This document discusses machine learning techniques for natural language processing, 
focusing on transformer architectures and their applications in text understanding.

----------------------------------------------------------

You: Can you explain the transformer architecture?

🤔 Thinking...

Bot: The transformer architecture uses self-attention mechanisms to process input sequences 
in parallel, allowing it to capture long-range dependencies more effectively than 
recurrent neural networks.
```

## 📁 Project Structure

```
chat-with-pdf/
│
├── chat_pdf.ipynb          # Main Jupyter notebook
├── .env                    # Environment variables (not committed)
├── .env.example           # Template for environment variables
├── .gitignore             # Git ignore rules
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── sample.pdf            # Your PDF file (example)
```

## 🔧 Configuration

### Adjusting Chunk Size

In Cell 6, modify the text splitter parameters:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Adjust chunk size (500-2000)
    chunk_overlap=200,      # Adjust overlap (100-300)
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
```

### Changing Retrieval Settings

In Cell 8, adjust the number of relevant chunks retrieved:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Change number of chunks (1-10)
)
```

### Using Different Models

Replace the model in Cell 9:

```python
# Alternative models:
# - "google/flan-t5-base"  (larger, more accurate)
# - "google/flan-t5-large" (even larger, requires more resources)

hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",  # Change model here
    max_length=200,
    temperature=0.7,
    do_sample=True,
    top_p=0.9
)
```

## 🎯 How It Works

1. **Document Loading** - PDF is loaded and text is extracted
2. **Text Splitting** - Document is split into smaller, overlapping chunks
3. **Embedding Generation** - Each chunk is converted to a vector embedding
4. **Vector Storage** - Embeddings are stored in Pinecone vector database
5. **Query Processing** - User question is converted to embedding
6. **Semantic Search** - Most relevant chunks are retrieved using similarity search
7. **Answer Generation** - LLM generates answer based on relevant context

## 🔒 Security Notes

- ✅ Never commit `.env` files to Git
- ✅ Always use `.gitignore` to protect sensitive files
- ✅ Regenerate API keys if accidentally exposed
- ✅ Use environment variables for all secrets

## 🐛 Troubleshooting

### Issue: "PINECONE_API_KEY not found"

**Solution:** 
- Ensure `.env` file exists in project root
- Check that `python-dotenv` is installed
- Restart VS Code/Jupyter after creating `.env`

### Issue: "ModuleNotFoundError: No module named 'langchain.text_splitter'"

**Solution:**
```bash
pip install langchain-text-splitters>=1.1.0
```

### Issue: "Model loading too slow"

**Solution:**
- Use `flan-t5-small` for faster performance
- Consider using GPU if available
- Reduce `max_length` parameter

### Issue: "Out of memory"

**Solution:**
- Reduce `chunk_size` in text splitter
- Use smaller model (`flan-t5-small`)
- Process smaller PDF files

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/) - Framework for LLM applications
- [Pinecone](https://www.pinecone.io/) - Vector database
- [Hugging Face](https://huggingface.co/) - Pre-trained models
- [Sentence Transformers](https://www.sbert.net/) - Embedding models

## 📧 Contact

Vishal Yadav - vy5068@gmail.com 

## 🎓 Learning Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Hugging Face Course](https://huggingface.co/course/chapter1/1)

---
