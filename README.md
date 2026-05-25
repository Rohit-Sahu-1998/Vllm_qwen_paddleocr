# Hybrid PaddleOCR + vLLM Document Extraction Pipeline

## Overview

This project is a hybrid document extraction pipeline that combines:

- PaddleOCR for OCR text extraction, bounding boxes, and OCR confidence
- Qwen2.5-VL (via vLLM) for semantic document understanding and structured JSON extraction
- FastAPI backend for asynchronous document processing
- Locust for load testing and concurrency testing

The system supports:
- PDF documents
- TIFF/TIF images
- Multi-page documents
- Concurrent document processing
- OCR + LLM matching
- Metrics logging
- GPU monitoring

---

# Architecture

Client / Locust
        ↓
FastAPI Backend
        ↓
PDF/TIFF → Image Conversion
        ↓
PaddleOCR Extraction
(text + bbox + confidence)
        ↓
Qwen2.5-VL via vLLM
(semantic extraction)
        ↓
OCR ↔ Qwen Matching
        ↓
Final Structured JSON
        ↓
Metrics + Logs + GPU Stats

---

# Pipeline Flow

## Step 1 — Document Upload

Documents are uploaded through:

POST /submit

Supported formats:
- .pdf
- .tif
- .tiff

Each upload creates a unique job_id.

---

## Step 2 — Document Conversion

PDF/TIFF pages are converted into images using:
- pdf2image
- PIL

---

## Step 3 — PaddleOCR Extraction

PaddleOCR extracts:
- OCR text
- bounding boxes
- OCR confidence

---

## Step 4 — Qwen Extraction (vLLM)

Qwen2.5-VL performs semantic extraction.

vLLM handles:
- request scheduling
- GPU batching
- concurrent inference
- KV cache management

---

## Step 5 — OCR + Qwen Matching

The backend matches:
- Qwen extracted values
- OCR text lines

This enables:
- field verification
- OCR confidence attachment
- bounding box highlighting
- visual validation

---

## Step 6 — Final JSON Generation

Final JSON includes:
- extracted fields
- OCR text
- OCR confidence
- bounding boxes
- matching score
- warnings
- metrics

---

# Queue Delay Explanation

## What is queue_delay_sec

queue_delay_sec = processing_started_at - submitted_at

It represents:
Time spent waiting before processing begins.

---

## Why Queue Delay Increased

Initially the pipeline was:

PDF → Qwen Extraction

After adding PaddleOCR:

PDF
↓
PaddleOCR
↓
Qwen Extraction
↓
OCR Matching

Each document now takes significantly longer to complete.

Under high concurrent load:

incoming requests > document completion rate

documents begin waiting in queue.

This waiting time appears as queue_delay_sec.

---

# Parallelism Strategy

## PaddleOCR

PaddleOCR can experience tensor memory issues under unsafe thread concurrency.

To improve stability:
- OCR execution is controlled carefully
- GPU OCR can be enabled
- OCR batching can be tuned

## Qwen/vLLM

vLLM handles:
- concurrent inference
- internal batching
- GPU scheduling

---

# Metrics Captured

The system logs:
- queue_delay_sec
- document_conversion_time_sec
- paddle_ocr_time_sec
- vllm_inference_time_sec
- matching_time_sec
- end_to_end_time_sec
- tokens_per_sec

---

# API Endpoints

## Submit Document

POST /submit

## Check Status

GET /status/{job_id}

## Health Check

GET /

---

# Running the System

## Start vLLM

```bash
vllm serve /path/to/model \
  --host 0.0.0.0 \
  --port 8002 \
  --trust-remote-code \
  --gpu-memory-utilization 0.90
```

## Start FastAPI

```bash
uvicorn app_hybrid_vllm:app --host 0.0.0.0 --port 8000
```

## Start Locust

```bash
locust -f locust_hybrid_vllm.py --host http://127.0.0.1:8000
```

---

# Output Files

Outputs are saved inside:

outputs_hybrid_vllm/

Generated files:
- extracted JSON
- metrics logs
- GPU logs
- failed job logs

---

# Tech Stack

- FastAPI
- PaddleOCR
- PaddlePaddle
- vLLM
- Qwen2.5-VL
- Locust
- PDF2Image
- PIL
- Python

---

# Future Improvements

Potential optimizations:
- Separate OCR GPU and Qwen GPU
- OCR process pool
- Redis/Celery queue
- Multi-GPU scaling
- Dynamic batching optimization
- OCR microservice architecture

---

# Author

Rohit Sahu
