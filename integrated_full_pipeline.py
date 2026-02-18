# integration_full_fixed.py
# Single-file integration: YOLO detector -> CNN classifier -> FAISS RAG -> LLaMA generation
# Edit CONFIG paths and run.

import os
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
# CONFIG - edit these paths
# =========================
YOLO_MODEL_PATH = "/content/best-2.pt"             # path to your YOLO .pt
CNN_MODEL_PATH  = "/content/best_cnn_model.pth"    # path to your CNN weights (.pth)
DOCS_PATH       = "/content/warehouse_rag_documents"  # folder with .txt docs
CLASS_MAPPING_FILE = "/content/class_mapping.txt"  # optional file with one class per line
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# =========================
# Helper: load class order
# =========================
def load_class_names():
    # Try multiple fallbacks, prefer explicit mapping file, then checkpoint, then default
    # 1) mapping text file (one class per line)
    if Path(CLASS_MAPPING_FILE).exists():
        names = [l.strip() for l in open(CLASS_MAPPING_FILE) if l.strip()]
        print("Loaded class mapping from", CLASS_MAPPING_FILE, "->", names)
        return names

    # 2) try to read from a saved checkpoint if it contains 'classes' key
    ckpt_candidates = ["final_model_and_classes.pth", "best_cnn_model_with_meta.pth"]
    for c in ckpt_candidates:
        if Path(c).exists():
            try:
                data = torch.load(c, map_location="cpu")
                if isinstance(data, dict) and "classes" in data:
                    print("Loaded class mapping from checkpoint", c)
                    return list(data["classes"])
            except Exception:
                pass

    # 3) fallback to the order which matched your standalone inference previously
    fallback = ["fragile", "hazardous", "standard"]
    print("Class mapping file not found; using fallback order:", fallback)
    return fallback

CLASS_NAMES = load_class_names()
NUM_CLASSES = len(CLASS_NAMES)

# =========================
# Load YOLO detector
# =========================
print("Loading YOLO model...")
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("YOLO loaded.")
except Exception as e:
    print("Failed to load YOLO model:", e)
    raise

# =========================
# Load CNN (EfficientNet-B0) - must match training arch
# =========================
print("Loading CNN model...")
cnn_model = models.efficientnet_b0(weights=None)
in_f = cnn_model.classifier[1].in_features
cnn_model.classifier[1] = torch.nn.Linear(in_f, NUM_CLASSES)
cnn_state = torch.load(CNN_MODEL_PATH, map_location=DEVICE)
# If state dict wrapped, handle both cases
if isinstance(cnn_state, dict) and "model_state" in cnn_state:
    state_dict = cnn_state["model_state"]
elif isinstance(cnn_state, dict) and "state_dict" in cnn_state:
    state_dict = cnn_state["state_dict"]
else:
    state_dict = cnn_state
cnn_model.load_state_dict(state_dict)
cnn_model.to(DEVICE)
cnn_model.eval()
print("CNN loaded. Classes:", CLASS_NAMES)

# transform must match training (use the proven one)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# =========================
# Load RAG (Embeddings + FAISS)
# =========================
print("Loading documents and building FAISS index...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

doc_texts = []
doc_names = []
docs_path = Path(DOCS_PATH)
if not docs_path.exists():
    raise FileNotFoundError("DOCS_PATH not found: " + str(DOCS_PATH))

for f in sorted(docs_path.glob("*.txt")):
    txt = f.read_text(encoding="utf8")
    doc_names.append(f.name)
    doc_texts.append(txt)
print(f"Loaded {len(doc_texts)} documents from {DOCS_PATH}")

doc_embeddings = embedder.encode(doc_texts, convert_to_numpy=True)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)
TOP_K = min(3, len(doc_texts))
print("FAISS index ready (dim {}).".format(dimension))

# =========================
# Load LLaMA (deterministic generation)
# =========================
LLAMA_MODEL =  "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # previously working
print("Loading LLaMA model (this may take a while)...")
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
llama_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)
llama_model.eval()
print("LLaMA loaded.")

# =========================
# RAG query function (strict prompt + deterministic generation)
# =========================
def ask_rag(question, detection_info, top_k=TOP_K):

    # 1️⃣ Embed query
    combined_query = f"{detection_info['class']} object. {question}"
    q_emb = embedder.encode([combined_query], convert_to_numpy=True)
    distances, indices = index.search(np.array(q_emb), top_k)

    retrieved_idxs = indices[0].tolist()
    retrieved_docs = [doc_texts[i] for i in retrieved_idxs]
    retrieved_names = [doc_names[i] for i in retrieved_idxs]

    context = "\n\n---\n\n".join(retrieved_docs)

    # 2️⃣ Inject detection metadata explicitly
    detection_block = f"""
DETECTION INFORMATION:
- Detected object class: {detection_info['class']}
- Classification confidence: {detection_info['confidence']:.4f}
"""


    prompt = f"""
You are a warehouse robotics assistant.

Use ONLY the context below to answer the question.
If the answer is not found, respond:
Information not found in documentation.


DETECTION:
Class: {detection_info['class']}
Confidence: {detection_info['confidence']:.2f}

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
            temperature=0,          # deterministic
            do_sample=False,          # no randomness
            repetition_penalty=1.1,   # prevents loops
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "FINAL ANSWER:" in text:
        ans = text.split("FINAL ANSWER:")[-1].strip()
    else:
        ans = text[-1000:].strip()

    return ans, retrieved_names


# =========================
# Utility: get best detection box (highest confidence)
# =========================
def select_best_box(boxes):
    """
    boxes is ultralytics results[0].boxes object
    We'll pick highest confidence; return xyxy numpy
    """
    try:
        confs = boxes.conf.cpu().numpy()  # shape (N,)
        idx = int(np.argmax(confs))
        xyxy = boxes.xyxy[idx].cpu().numpy()
        return xyxy, float(confs[idx])
    except Exception:
        # fallback to first box
        try:
            return boxes.xyxy[0].cpu().numpy(), float(boxes.conf[0].cpu().item())
        except Exception:
            return None, None

# =========================
# Main pipeline
# =========================
def run_pipeline(image_path, pad_ratio=0.15, confidence_threshold=0.5, fallback_threshold=0.6):
    print("\n=== Running pipeline for:", image_path, "===\n")
    # 1) Detect
    results = yolo_model(image_path)  # Ultralytics model call
    r0 = results[0]
    if not hasattr(r0, "boxes") or len(r0.boxes) == 0:
        print("No object detected.")
        return

    # pick best detection (highest confidence)
    xyxy, det_conf = select_best_box(r0.boxes)
    if xyxy is None:
        print("No valid box found.")
        return
    x1, y1, x2, y2 = map(int, xyxy.tolist())
    print(f"Selected detection xyxy={x1,y1,x2,y2}  conf={det_conf:.3f}")

    # read image
    img = cv2.imread(str(image_path))
    if img is None:
        print("Failed to read image.")
        return
    h, w = img.shape[:2]

    # 2) pad bbox proportionally (clamp)
    bx = x2 - x1
    by = y2 - y1
    pad_x = int(pad_ratio * bx)
    pad_y = int(pad_ratio * by)
    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(w, x2 + pad_x)
    y2p = min(h, y2 + pad_y)

    crop = img[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        print("Invalid crop after padding; using full image.")
        crop = img.copy()

    # 3) CNN classify padded crop
    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = cnn_model(input_tensor)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_prob = float(probs[pred_idx])

    print("CNN (crop) prediction:", CLASS_NAMES[pred_idx], f"prob={pred_prob:.4f}")
    print("All class probabilities:", {CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))})

    # 4) fallback: if confidence low, try full image and use whichever has higher predicted prob for same class
    if pred_prob < fallback_threshold:
        pil_full = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        input_full = transform(pil_full).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out_full = cnn_model(input_full)
            probs_full = torch.softmax(out_full, dim=1).cpu().numpy()[0]
            pred_idx_full = int(np.argmax(probs_full))
            pred_prob_full = float(probs_full[pred_idx_full])

        print("CNN (full image) prediction:", CLASS_NAMES[pred_idx_full], f"prob={pred_prob_full:.4f}")
        print("All class probs (full):", {CLASS_NAMES[i]: float(probs_full[i]) for i in range(len(probs_full))})

        # decide which to use: prefer same predicted class with higher probability, else take higher absolute
        if pred_idx_full == pred_idx and pred_prob_full > pred_prob:
            pred_idx, pred_prob = pred_idx_full, pred_prob_full
            print("Using full-image result (same class, higher prob).")
        elif pred_prob_full > pred_prob and pred_prob_full >= confidence_threshold:
            pred_idx, pred_prob = pred_idx_full, pred_prob_full
            print("Using full-image result (higher prob).")
        else:
            print("Keeping crop result (even if low confidence).")

    label = CLASS_NAMES[pred_idx]

    # 5) Auto-generate precise question
    print("\nDetected Object Class:", label)
    print("You can now ask a question about this object.")
    print("Type 'exit' to quit.\n")

    while True:
        user_q = input("Your Question: ").strip()

        if user_q.lower() == "exit":
            break

        # Optional: guide retrieval using detected label
        enriched_question = f"""
    The detected object is classified as '{label}'.
    User question: {user_q}
    """

        detection_info = {
    "class": label,
    "confidence": pred_prob
}

        answer, retrieved_names = ask_rag(user_q, detection_info)


        print("\nTop retrieved docs:", retrieved_names)
        print("\n===== RESPONSE =====\n")
        print(answer)
        print("\n====================\n")

    # 6) RAG + LLaMA
    answer, retrieved_names = ask_rag(question)
    print("\nTop retrieved docs:", retrieved_names)
    print("\n===== FINAL RESPONSE =====\n")
    print(answer)
    print("\n==========================\n")

# =========================
# Interactive demo
# =========================
if __name__ == "__main__":
    img_path = input("Enter image path: ").strip()
    if not Path(img_path).exists():
        print("Image not found:", img_path)
        sys.exit(1)
    run_pipeline(img_path)
