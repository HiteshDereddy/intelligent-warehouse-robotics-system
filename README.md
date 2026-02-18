# Intelligent Warehouse Object Recognition & Safety Query System

End-to-end system that detects warehouse objects, classifies them into safety categories, and answers handling-related questions using a Retrieval-Augmented Generation (RAG) pipeline.

---

## Problem Statement

Build an intelligent warehouse assistant that:

1. Detects objects from images
2. Classifies them as:
   - Fragile
   - Hazardous
   - Standard
3. Retrieves relevant warehouse documentation
4. Answers natural language safety queries grounded in documentation

---

## System Architecture

Image  
→ YOLO Detection  
→ CNN Classification (EfficientNet-B0)  
→ Detection Metadata  
→ FAISS Retrieval  
→ TinyLlama (LLM)  
→ Grounded Safety Response  

---

## Project Structure

# Intelligent Warehouse Object Recognition & Safety Query System

End-to-end system that detects warehouse objects, classifies them into safety categories, and answers handling-related questions using a Retrieval-Augmented Generation (RAG) pipeline.

---

## Problem Statement

Build an intelligent warehouse assistant that:

1. Detects objects from images
2. Classifies them as:
   - Fragile
   - Hazardous
   - Standard
3. Retrieves relevant warehouse documentation
4. Answers natural language safety queries grounded in documentation

---

## System Architecture

Image  
→ YOLO Detection  
→ CNN Classification (EfficientNet-B0)  
→ Detection Metadata  
→ FAISS Retrieval  
→ TinyLlama (LLM)  
→ Grounded Safety Response  

---

## Project Structure

# Intelligent Warehouse Object Recognition & Safety Query System

End-to-end system that detects warehouse objects, classifies them into safety categories, and answers handling-related questions using a Retrieval-Augmented Generation (RAG) pipeline.

---

## Problem Statement

Build an intelligent warehouse assistant that:

1. Detects objects from images
2. Classifies them as:
   - Fragile
   - Hazardous
   - Standard
3. Retrieves relevant warehouse documentation
4. Answers natural language safety queries grounded in documentation

---

## System Architecture

Image  
→ YOLO Detection  
→ CNN Classification (EfficientNet-B0)  
→ Detection Metadata  
→ FAISS Retrieval  
→ TinyLlama (LLM)  
→ Grounded Safety Response  

---

## Project Structure

part1_classical_cv/
part2_cnn_classifier/
part3_rag_system/
part4_integration/
results/
requirements.txt
README.md


---

## Part 1 – Classical Computer Vision

Technique Used:
- Template Matching / Feature Detection (ORB)

Outputs:
- Bounding box coordinates
- Object dimensions (pixels)
- Center coordinates

Purpose:
- Lightweight object recognition without training
- Demonstrates foundational CV knowledge

Limitations:
- Sensitive to scale and lighting variations

---

## Part 2 – CNN Object Classification

Model:
- EfficientNet-B0
- Input: 224×224
- Classes: Fragile, Hazardous, Standard

### Test Performance

| Class      | Precision | Recall | F1 |
|------------|-----------|--------|----|
| Fragile    | 1.00      | 1.00   | 1.00 |
| Hazardous  | 1.00      | 1.00   | 1.00 |
| Standard   | 1.00      | 1.00   | 1.00 |

Overall Accuracy: **99%**

---

## Part 3 – Retrieval-Augmented Generation (RAG)

Embedding Model:
- all-MiniLM-L6-v2

Vector Store:
- FAISS (L2 similarity)

LLM:
- TinyLlama-1.1B
- Deterministic decoding (temperature = 0)

Pipeline:
1. Embed warehouse documents
2. Enrich query with detected class
3. Retrieve top-k relevant docs
4. Generate grounded answer

---

## Part 4 – Integrated System

Workflow:
1. User inputs image
2. YOLO detects object
3. CNN classifies object
4. User asks safety question
5. System retrieves documentation
6. LLM generates context-grounded response

Run integration:


---

## Installation

Install dependencies:
pip install -r requirements.txt


---

## Key Design Decisions

- Classical CV included to satisfy foundational requirement
- EfficientNet chosen for lightweight accuracy
- Detection class injected into retrieval query
- Deterministic decoding used to reduce hallucination
- Modular structure for clarity and scalability

---

## Limitations

- Sensitive to lighting and occlusion
- Template matching not rotation-invariant
- LLM reasoning limited by 1.1B parameter size
- RAG limited by document coverage

---

## Future Improvements

- Upgrade to Llama-3.2-3B-Instruct
- Hybrid keyword + vector retrieval
- Web-based UI
- Real-time camera feed integration

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- Ultralytics YOLO
- FAISS
- SentenceTransformers
- HuggingFace Transformers

---

## Status

Complete end-to-end working prototype with interactive CLI interface.

