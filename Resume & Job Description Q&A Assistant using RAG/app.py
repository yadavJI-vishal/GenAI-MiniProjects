#!pip install llama-index llama-index-llms-huggingface llama-index-embeddings-huggingface transformers accelerate bitsandbytes pypdf2
# pip install -U bitsandbytes

import os
import time
import torch
from PyPDF2 import PdfReader
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig

# ============================================================
# 1. Configuration & Paths
# ============================================================
# Update this path to your actual PDF file location
PDF_PATH = "/content/drive/MyDrive/Gen AI Mini Projects/RAG Chatbot/data/Vishal-Yadav.pdf"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 3

# ============================================================
# 2. Extract PDF Text
# ============================================================
def extract_text_from_pdf(pdf_path):
    print("📄 Extracting PDF...")
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

resume_text = extract_text_from_pdf(PDF_PATH)
documents = [Document(text=resume_text)]

# ============================================================
# 3. Setup Embedding Model (Lightweight & Free)
# ============================================================
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 4. Setup LLM (Mistral-7B with 4-bit Quantization)
# ============================================================
print("🤖 Loading Mistral-7B (4-bit)...")

# This config allows the model to run on free Colab GPUs
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Note: You may need to login to Hugging Face or use a token if the model is gated
# from huggingface_hub import login; login()

llm = HuggingFaceLLM(
    model_name=model_id,
    tokenizer_name=model_id,
    context_window=4096,
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.1, "do_sample": False},
    model_kwargs={"quantization_config": quantization_config},
    device_map="auto",
)

Settings.llm = llm

# ============================================================
# 5. Create Index and Query Engine
# ============================================================
print("💾 Creating Vector Index...")
index = VectorStoreIndex.from_documents(
    documents,
    transformations=[Settings.embed_model]
)

# Custom Prompt for Mistral (Uses [INST] tags for better instruction following)
qa_prompt_tmpl = (
    "<s>[INST] Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: [/INST]"
)
qa_prompt = PromptTemplate(qa_prompt_tmpl)

# Define Query Engine ONCE to avoid overriding settings
query_engine = index.as_query_engine(
    similarity_top_k=TOP_K,
    text_qa_template=qa_prompt,
    response_mode="compact" # Efficient for single-document RAG
)

# ============================================================
# 6. Testing & Interactive Mode
# ============================================================
def run_test_queries():
    queries = [
        "List all the skills mentioned in this resume.",
        "What is the email address in this resume?"
    ]

    for q in queries:
        print(f"\n📝 Query: {q}")
        start = time.time()
        response = query_engine.query(q)
        print(f"✅ Response ({time.time()-start:.2f}s):")
        print(f"--------------------------------------------------")
        print(response)
        print(f"--------------------------------------------------")

run_test_queries()

print("\n💬 Entering Interactive Mode (type 'exit' to quit)")
while True:
    user_query = input("💬 You: ")
    if user_query.lower() == "exit": break
    if not user_query: continue

    response = query_engine.query(user_query)
    print(f"\n🤖 Bot:\n{response}\n")









