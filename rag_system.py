import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

embedder = SentenceTransformer("all-MiniLM-L6-v2")

docs = []
for f in Path("warehouse_rag_documents").glob("*.txt"):
    docs.append(f.read_text())

embeddings = embedder.encode(docs, convert_to_numpy=True)
dim = embeddings.shape[1]

index = faiss.IndexFlatL2(dim)
index.add(embeddings)

tokenizer = AutoTokenizer.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
)
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
).to(DEVICE)

def ask(question):
    q_emb = embedder.encode([question], convert_to_numpy=True)
    _, I = index.search(q_emb, 3)

    context = "\n".join([docs[i] for i in I[0]])

    prompt = f"""
    You are a warehouse safety assistant.
    
    Use ONLY the provided CONTEXT to answer the QUESTION.
    
    If the answer is not found in the context, respond exactly:
    Information not found in documentation.
    
    Always answer directly like a conversation, do not invoke the prompt again in the answer.
    CONTEXT:
    {context}
    
    Use the detection to understand more and better about the questioN.
    QUESTION:
    {question}
    
    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    print(ask("How should fragile items be handled?"))
