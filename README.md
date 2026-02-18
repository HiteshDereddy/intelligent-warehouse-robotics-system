# Intelligent Warehouse Object Recognition & Safety Query System

An end-to-end intelligent warehouse assistant that detects objects, classifies them into safety categories, retrieves relevant documentation, and generates grounded safety responses using a Retrieval-Augmented Generation (RAG) pipeline.

---

# Problem Statement

Develop an intelligent system that:

1. Detects warehouse objects from images
2. Classifies them into:
   - Fragile
   - Hazardous
   - Standard
3. Retrieves relevant warehouse safety documentation
4. Answers natural language safety-related questions grounded in documentation

---

# System Architecture

```
Image
   ↓
YOLO Detection
   ↓
CNN Classification (EfficientNet-B0)
   ↓
Detection Metadata Injection
   ↓
FAISS Retrieval
   ↓
TinyLlama (LLM)
   ↓
Grounded Safety Response
```

---

# 📁 Project Structure

```
warehouse-intelligent-system/
│
├── openCV_detector.py
│   
│
├── train_cnn.py
│   
│
├── rag_system.py
│  
│
├── integration_full_fixed.py
│ 
│
├── warehouse_rag_documents/
│   └── 10–15 warehouse documentation files (.txt)
│
├── models/
│   ├── best-2.pt
│   └── best_cnn_model.pth
│
│
├── results/
│   ├── confusion_matrix.png
│   ├── metrics.txt
│   └── demo_screenshots/
│
├── requirements.txt
└── README.md
```

---

# 🖥️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/HiteshDereddy/intelligent-warehouse-robotics-system
cd warehouse-intelligent-system
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# How to Run Each Component

---

## Part 1 – Classical Computer Vision (OpenCV)

```bash
python openCV_detector.py
```

Outputs:
- Bounding boxes
- Pixel dimensions
- Center coordinates
- Saved detection images in `results/`

---

## Part 2 – CNN Classification

### Train Model

```bash
python train_cnn.py
```

Outputs:
- Saved model (`CNNModel.pth`)
- Confusion matrix (`results/confusion_matrix.png`)
- Metrics file (`results/metrics.txt`)

### Inference

```python
predict_image("sample.jpg")
```

Displays:
- Input image
- Predicted class
- Confidence score

---

## Part 3 – RAG System

```bash
python rag_system.py
```

Example queries:
- How should the robot handle fragile items?
- What is the maximum weight capacity of the gripper?
- What safety checks are required before moving hazardous materials?

---

## Part 4 – Integrated System

```bash
python integration_full_fixed.py
```

Workflow:
1. Input image
2. YOLO detection
3. CNN classification
4. User safety query
5. FAISS retrieval
6. TinyLlama generates grounded response

---

# Model Performance (CNN)

| Class      | Precision | Recall | F1-score |
|------------|-----------|--------|----------|
| Fragile    | 1.00      | 1.00   | 1.00     |
| Hazardous  | 1.00      | 1.00   | 1.00     |
| Standard   | 1.00      | 1.00   | 1.00     |

**Overall Accuracy: 99%**

Confusion matrix available in:
```
results/confusion_matrix.png
```

---

# RAG System Details

Embedding Model:
- `all-MiniLM-L6-v2`

Vector Store:
- FAISS (L2 similarity)

LLM:
- TinyLlama-1.1B
- Deterministic decoding (temperature = 0)

Pipeline:
1. Chunk warehouse documents
2. Embed documents
3. Retrieve top-k relevant chunks
4. Inject detection metadata
5. Generate context-grounded response

---

# Key Design Decisions

- Classical CV included to demonstrate foundational OpenCV knowledge
- EfficientNet-B0 selected for lightweight transfer learning
- Detection class injected into retrieval query for contextual grounding
- Deterministic decoding used to reduce hallucination
- Modular architecture for clarity and maintainability

---

# Challenges Faced & Solutions

### 1. Class Imbalance
Some safety categories had fewer samples.
→ Addressed through proper train/validation/test splits and monitoring F1-score.

### 2. Bounding Box Cropping Errors
Incorrect crops reduced CNN confidence.
→ Implemented bounding box padding and fallback full-image classification.

### 3. Hallucination in LLM
LLM sometimes generated information not in documents.
→ Used strict prompt instructions and deterministic decoding (temperature = 0).

### 4. Retrieval Irrelevance
Generic queries retrieved unrelated documents.
→ Injected detected class label into embedding query for contextual narrowing.

---

# Results Folder Contents

The `results/` directory contains:

- Confusion matrix
- Performance metrics
- Detection screenshots
- RAG response examples
- Integration demo screenshots

---

# Future Improvements

- Upgrade to larger instruction-tuned LLM (e.g., Llama 3 series)
- Hybrid retrieval (keyword + vector search)
- Real-time camera stream integration
- Web-based user interface
- Multi-object scene reasoning

---

# Technologies Used

- Python
- PyTorch
- OpenCV
- Ultralytics YOLO
- FAISS
- SentenceTransformers
- HuggingFace Transformers

---

# Demo

A 5-minute demonstration video is included with submission, showcasing:

- Object detection
- CNN classification
- RAG-based query answering
- Full integrated workflow

---

# Status

Complete end-to-end working prototype with modular architecture and interactive CLI interface.
