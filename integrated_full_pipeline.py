# integration_full_fixed.py
# YOLO -> CNN -> FAISS RAG -> TinyLLaMA

import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CONFIG (RELATIVE PATHS)
# =========================
YOLO_MODEL_PATH = "models/best-2.pt"
CNN_MODEL_PATH  = "models/best_cnn_model.pth"
DOCS_PATH       = "warehouse_rag_documents"
CLASS_MAPPING_FILE = "models/class_mapping.txt"
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# =========================
# Load Class Names
# =========================
def load_class_names():
    if Path(CLASS_MAPPING_FILE).exists():
        names = [l.strip() for l in open(CLASS_MAPPING_FILE) if l.strip()]
        print("Loaded class mapping:", names)
        return names

    fallback = ["fragile", "hazardous", "standard"]
    print("Using fallback class order:", fallback)
    return fallback

CLASS_NAMES = load_class_names()
NUM_CLASSES = len(CLASS_NAMES)

# =========================
# Load YOLO
# =========================
print("Loading YOLO...")
yolo_model = YOLO(YOLO_MODEL_PATH)

# =========================
# Load CNN
# =========================
print("Loading CNN...")
cnn_model = models.efficientnet_b0(weights=None)
in_f = cnn_model.classifier[1].in_features
cnn_model.classifier[1] = torch.nn.Linear(in_f, NUM_CLASSES)

state = torch.load(CNN_MODEL_PATH, map_location=DEVICE)
cnn_model.load_state_dict(state)
cnn_model.to(DEVICE)
cnn_model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# =========================
# Load RAG
# =========================
print("Building FAISS index...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

doc_texts = []
doc_names = []

for f in sorted(Path(DOCS_PATH).glob("*.txt")):
    doc_texts.append(f.read_text(encoding="utf8"))
    doc_names.append(f.name)

embeddings = embedder.encode(doc_texts, convert_to_numpy=True)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

TOP_K = min(3, len(doc_texts))

# =========================
# Load TinyLLaMA
# =========================
LLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
llama_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)
llama_model.eval()

# =========================
# RAG Query
# =========================
def ask_rag(question, detection_info):

    query = f"{detection_info['class']} object. {question}"
    q_emb = embedder.encode([query], convert_to_numpy=True)
    _, I = index.search(q_emb, TOP_K)

    context = "\n\n".join([doc_texts[i] for i in I[0]])

    prompt = f"""
You are a warehouse safety assistant.

Use ONLY the provided context to answer.
If not found, say: Information not found in documentation.

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
        outputs = llama_model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0,
            do_sample=False,
            repetition_penalty=1.1
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response, [doc_names[i] for i in I[0]]

# =========================
# Select Best Detection
# =========================
def select_best_box(boxes):
    confs = boxes.conf.cpu().numpy()
    idx = int(np.argmax(confs))
    return boxes.xyxy[idx].cpu().numpy(), float(confs[idx])

# =========================
# Main Pipeline
# =========================
def run_pipeline(image_path):

    print("\nProcessing:", image_path)

    results = yolo_model(image_path)
    r0 = results[0]

    if len(r0.boxes) == 0:
        print("No object detected.")
        return

    xyxy, det_conf = select_best_box(r0.boxes)
    x1, y1, x2, y2 = map(int, xyxy)

    img = cv2.imread(image_path)
    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        crop = img.copy()

    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = transform(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = cnn_model(tensor)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_prob = float(probs[pred_idx])

    label = CLASS_NAMES[pred_idx]

    print("Detected:", label, f"(confidence {pred_prob:.4f})")

    detection_info = {
        "class": label,
        "confidence": pred_prob
    }

    while True:
        user_q = input("\nAsk question (type exit to quit): ").strip()
        if user_q.lower() == "exit":
            print("Session ended.")
            break

        answer, docs = ask_rag(user_q, detection_info)

        print("\nRetrieved Docs:", docs)
        print("\nAnswer:\n", answer)

# =========================
# Entry
# =========================
if __name__ == "__main__":
    img_path = input("Enter image path: ").strip()
    if not Path(img_path).exists():
        print("Image not found.")
        sys.exit(1)

    run_pipeline(img_path)
