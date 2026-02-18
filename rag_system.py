# rag_system.py
# Retrieval-Augmented Generation (RAG) for Warehouse Robotics

import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ==========================
# CONFIG
# ==========================
DOCS_PATH = "/content/warehouse_rag_documents/"
TOP_K = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# Load & Chunk Documents
# ==========================
def load_documents(path):
    docs = []
    names = []

    for file in sorted(Path(path).glob("*.txt")):
        text = file.read_text(encoding="utf8")
        chunks = chunk_text(text, chunk_size=400)

        for chunk in chunks:
            docs.append(chunk)
            names.append(file.name)

    return docs, names


def chunk_text(text, chunk_size=400):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


print("Loading documents...")
documents, doc_names = load_documents(DOCS_PATH)
print(f"Loaded {len(documents)} chunks.")

# ==========================
# Embedding Model
# ==========================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

print("FAISS index ready.")

# ==========================
# Load LLM
# ==========================
LLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)
model.eval()

print("RAG system ready.")

# ==========================
# Query Function
# ==========================
def ask_question(question):

    # Embed query
    query_embedding = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, TOP_K)

    retrieved_chunks = [documents[i] for i in indices[0]]
    retrieved_sources = [doc_names[i] for i in indices[0]]

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a warehouse robotics assistant.

Use ONLY the context below to answer the question.
If the answer is not found, respond:
Information not found in documentation.

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True,
                       max_length=2048).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0,
            do_sample=False,
            repetition_penalty=1.1
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer, retrieved_sources


# ==========================
# Interactive Demo
# ==========================
if __name__ == "__main__":

    print("\nExample Queries:")
    print("1. How should the robot handle fragile items?")
    print("2. What is the maximum weight capacity for the gripper arm?")
    print("3. What safety checks are required before moving hazardous materials?\n")

    while True:
        user_query = input("Ask a question (type 'exit' to quit): ")

        if user_query.lower() == "exit":
            break

        response, sources = ask_question(user_query)

        print("\nRetrieved Documents:", sources)
        print("\nAnswer:\n")
        print(response)
        print("\n----------------------------------------\n")
