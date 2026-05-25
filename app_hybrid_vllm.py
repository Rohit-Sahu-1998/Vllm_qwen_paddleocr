import os
import uuid
import time
import json
import re
import base64
import asyncio
import csv
import subprocess
import math
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
from difflib import SequenceMatcher
#from threading import Lock
import numpy as np
from PIL import Image, ImageSequence
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pdf2image import convert_from_bytes
from openai import OpenAI
from paddleocr import PaddleOCR


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "outputs_hybrid_vllm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS_LOG_FILE = os.path.join(OUTPUT_DIR, "metrics_log.jsonl")
GPU_METRICS_CSV_FILE = os.path.join(OUTPUT_DIR, "gpu_metrics_log.csv")

VLLM_BASE_URL = "http://127.0.0.1:8002/v1"
VLLM_API_KEY = "EMPTY"
VLLM_MODEL_NAME = "/home/rohit.sahu/Qwen_model/cpt_codes/quantized_model/Quantized_model_qwen_4bit"

CPU_THREAD_WORKERS = 6
PAGE_CONCURRENCY_LIMIT = 2
JOB_THREAD_WORKERS = 2
PDF_DPI = 160
MAX_IMAGE_SIZE = 1600
MAX_TOKENS = 1200

SUPPORTED_EXTENSIONS = (".pdf", ".tif", ".tiff")

# Options: cheque / comp_check / patient
DOCUMENT_TYPE = "cheque"


# ============================================================
# PROMPTS
# ============================================================

CHEQUE_PROMPT = """
You are extracting data from a cheque image.

Return ONLY a valid JSON object.
No explanation.
No markdown.
No backticks.

Required JSON structure:
{
  "check_number": null,
  "check_amount": null,
  "pay_to": null,
  "provider_name": null
}

Rules:
- check_number: read from MICR bottom line first, else from printed number.
- check_number must be digits only.
- Strip leading zeros.
- Maximum 9 digits. If longer than 9 digits, keep only the last 9.
- check_amount: dollar amount exactly as printed.
- pay_to: name written immediately after "Pay to the Order of".
- provider_name: company/provider who issued the cheque, not the bank.
- Any field not found must be null.
"""

COMP_CHECK_PROMPT = """
Look at every part of this image very carefully.
Include handwritten text, stamps, annotations, or printed text anywhere on the page.

Does this image contain the words "Comp Benefits" or "Comp Benefit" anywhere?

Examples that count:
- ATTN: Comp Benefits
- RE: Comp Benefits
- Comp Benefits Plan
- Workers Comp Benefits

Reply with ONLY one word:
YES or NO
"""

PATIENT_PROMPT = """
You are extracting patient/claim records from this document image.

Return ONLY valid JSON.
No markdown.
No explanation.
No backticks.

Rules:
- Extract only values visible on the page.
- If a value is missing, use null.
- Never copy instruction text into values.
- Extract every visible patient row/record.
- If only patient name and amount are visible, still create a patient object.
- Missing fields must be null.

Return this JSON:
{
  "patients": [
    {
      "patient_name": null,
      "first_name": null,
      "last_name": null,
      "dos": null,
      "member_id": null,
      "claim_number": null,
      "reason_for_refund": null,
      "claim_payment_amount": null,
      "payee_name": null
    }
  ]
}
""".strip()

PROMPT_MAP = {
    "cheque": CHEQUE_PROMPT,
    "comp_check": COMP_CHECK_PROMPT,
    "patient": PATIENT_PROMPT,
}

PROMPT = PROMPT_MAP[DOCUMENT_TYPE]


# ============================================================
# APP STATE
# ============================================================

app = FastAPI(title="Hybrid PaddleOCR + vLLM Document Extraction API")

jobs: Dict[str, Dict[str, Any]] = {}

cpu_executor = ThreadPoolExecutor(max_workers=CPU_THREAD_WORKERS)
job_executor = ThreadPoolExecutor(max_workers=JOB_THREAD_WORKERS)

#ocr_lock = Lock()


vllm_client = OpenAI(
    api_key=VLLM_API_KEY,
    base_url=VLLM_BASE_URL,
)


# ============================================================
# RESPONSE MODELS
# ============================================================

class SubmitResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Dict[str, Any] | None = None
    error: str | None = None
    metrics: Dict[str, Any] | None = None
    json_file_path: str | None = None


# ============================================================
# BASIC HELPERS
# ============================================================

def now() -> float:
    return time.time()


def current_timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds")


def round_or_none(value):
    return None if value is None else round(value, 3)


def get_file_extension(filename: str) -> str:
    return os.path.splitext(filename.lower())[1]


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_code_like_text(text: Any) -> str:
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9]", "", text)
    return text


def bbox_center(bbox: List[List[float]]) -> Tuple[float, float]:
    xs = [pt[0] for pt in bbox]
    ys = [pt[1] for pt in bbox]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


# ============================================================
# PLACEHOLDER CLEANING
# ============================================================

PLACEHOLDER_PHRASES = [
    "full name exactly as written",
    "first name only",
    "last name only",
    "date of service",
    "admit date",
    "disch date",
    "value from subscriber",
    "member id",
    "patient id",
    "policy id",
    "exactly 15-digit",
    "never use group",
    "full text from reason",
    "reason for refund",
    "description text after patient name",
    "dollar amount from amount",
    "your share",
    "correct amnt",
    "refund",
    "total charges",
    "provider/payee name",
    "null if not found",
    "if clearly separable",
    "include $ if printed",
]


def clean_placeholder_value(value):
    if value is None:
        return None

    if not isinstance(value, str):
        value = str(value)

    value = value.strip()

    if value in ("", "null", "None", "NONE", "NULL"):
        return None

    lower_value = value.lower()

    for phrase in PLACEHOLDER_PHRASES:
        if phrase in lower_value:
            return None

    return value


# ============================================================
# SCHEMA HELPERS
# ============================================================

def get_empty_result_schema() -> Dict[str, Any]:
    if DOCUMENT_TYPE == "cheque":
        return {
            "check_number": None,
            "check_amount": None,
            "pay_to": None,
            "provider_name": None,
        }

    if DOCUMENT_TYPE == "comp_check":
        return {
            "comp_benefits_detected": False,
        }

    if DOCUMENT_TYPE == "patient":
        return {
            "comp_benefits_detected": False,
            "has_patient_data": False,
            "patients": [],
        }

    return {}


def normalize_cheque_output(parsed: Dict[str, Any]) -> Dict[str, Any]:
    schema = get_empty_result_schema()

    if not isinstance(parsed, dict):
        return schema

    for key in schema:
        schema[key] = clean_placeholder_value(parsed.get(key))

    if schema["check_number"]:
        digits = re.sub(r"\D", "", schema["check_number"])
        digits = digits.lstrip("0")
        if len(digits) > 9:
            digits = digits[-9:]
        schema["check_number"] = digits if digits else None

    return schema


def normalize_patient_output(parsed: Dict[str, Any]) -> Dict[str, Any]:
    schema = get_empty_result_schema()

    if not isinstance(parsed, dict):
        return schema

    comp_detected = parsed.get("comp_benefits_detected", False)
    has_patient_data = parsed.get("has_patient_data", False)
    patients = parsed.get("patients", [])

    schema["comp_benefits_detected"] = bool(comp_detected)
    schema["has_patient_data"] = bool(has_patient_data)

    if schema["comp_benefits_detected"]:
        schema["has_patient_data"] = False
        schema["patients"] = []
        return schema

    if not isinstance(patients, list):
        patients = []

    normalized_patients = []

    patient_schema_keys = [
        "patient_name",
        "first_name",
        "last_name",
        "dos",
        "member_id",
        "claim_number",
        "reason_for_refund",
        "claim_payment_amount",
        "payee_name",
    ]

    for patient in patients:
        if not isinstance(patient, dict):
            continue

        normalized = {}

        for key in patient_schema_keys:
            normalized[key] = clean_placeholder_value(patient.get(key))

        claim_number = normalized.get("claim_number")
        if claim_number:
            digits = re.sub(r"\D", "", claim_number)
            normalized["claim_number"] = digits if len(digits) == 15 else None

        if all(value is None for value in normalized.values()):
            continue

        normalized_patients.append(normalized)

    schema["patients"] = normalized_patients
    schema["has_patient_data"] = len(normalized_patients) > 0

    return schema


def normalize_comp_check_output(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip().upper()
    return {
        "comp_benefits_detected": text.startswith("YES")
    }


def normalize_output(parsed: Dict[str, Any], raw_text: str = "") -> Dict[str, Any]:
    if DOCUMENT_TYPE == "cheque":
        return normalize_cheque_output(parsed)

    if DOCUMENT_TYPE == "patient":
        return normalize_patient_output(parsed)

    if DOCUMENT_TYPE == "comp_check":
        return normalize_comp_check_output(raw_text)

    return parsed if isinstance(parsed, dict) else {}


def is_empty_extraction(data: Dict[str, Any]) -> bool:
    if DOCUMENT_TYPE == "comp_check":
        return False

    if DOCUMENT_TYPE == "cheque":
        return all(value in ("", None, [], {}) for value in data.values())

    if DOCUMENT_TYPE == "patient":
        return (
            data.get("comp_benefits_detected") is False
            and data.get("has_patient_data") is False
            and data.get("patients") == []
        )

    return not bool(data)


# ============================================================
# GPU METRICS HELPERS
# ============================================================

def get_gpu_stats() -> Dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()

        first_gpu = output.splitlines()[0]
        gpu_util, mem_used, mem_total, temp, power = first_gpu.split(",")

        return {
            "gpu_util_percent": int(gpu_util.strip()),
            "vram_used_mb": int(mem_used.strip()),
            "vram_total_mb": int(mem_total.strip()),
            "gpu_temp_c": int(temp.strip()),
            "gpu_power_w": float(power.strip()),
        }

    except Exception as e:
        return {
            "gpu_util_percent": None,
            "vram_used_mb": None,
            "vram_total_mb": None,
            "gpu_temp_c": None,
            "gpu_power_w": None,
            "gpu_error": str(e),
        }


def init_gpu_csv_log():
    if not os.path.exists(GPU_METRICS_CSV_FILE):
        with open(GPU_METRICS_CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "job_id",
                    "filename",
                    "file_type",
                    "document_type",
                    "status",
                    "stage",
                    "gpu_util_percent",
                    "vram_used_mb",
                    "vram_total_mb",
                    "gpu_temp_c",
                    "gpu_power_w",
                ],
            )
            writer.writeheader()


def log_gpu_metrics_csv(
    job_id: str,
    filename: str,
    file_type: str,
    status: str,
    stage: str,
    gpu_stats: Dict[str, Any],
):
    init_gpu_csv_log()

    row = {
        "timestamp": current_timestamp(),
        "job_id": job_id,
        "filename": filename,
        "file_type": file_type,
        "document_type": DOCUMENT_TYPE,
        "status": status,
        "stage": stage,
        "gpu_util_percent": gpu_stats.get("gpu_util_percent"),
        "vram_used_mb": gpu_stats.get("vram_used_mb"),
        "vram_total_mb": gpu_stats.get("vram_total_mb"),
        "gpu_temp_c": gpu_stats.get("gpu_temp_c"),
        "gpu_power_w": gpu_stats.get("gpu_power_w"),
    }

    with open(GPU_METRICS_CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)


# ============================================================
# JSON REPAIR / PARSER
# ============================================================

def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def repair_json_text(text: str) -> str:
    text = strip_code_fences(text or "")

    first = text.find("{")
    last = text.rfind("}")

    if first == -1:
        raise ValueError("No JSON object found")

    if last == -1 or last <= first:
        text = text[first:] + "}"
    else:
        text = text[first:last + 1]

    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(
        r'(?<=[{,\s])([A-Za-z_][A-Za-z0-9_]*)\s*:',
        r'"\1":',
        text,
    )

    text = text.replace("\n", " ").replace("\t", " ")

    return text.strip()


def safe_extract_json(text: str) -> tuple[bool, Dict[str, Any] | None, str | None]:
    if DOCUMENT_TYPE == "comp_check":
        return True, normalize_comp_check_output(text), None

    try:
        cleaned = repair_json_text(text)
        parsed = json.loads(cleaned)
        return True, parsed, None
    except Exception as e:
        return False, None, str(e)


# ============================================================
# DOCUMENT CONVERSION
# ============================================================

def resize_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return image


def convert_pdf_to_images(file_bytes: bytes) -> List[Image.Image]:
    images = convert_from_bytes(file_bytes, dpi=PDF_DPI)
    return [resize_image(img) for img in images]


def convert_tiff_to_images(file_bytes: bytes) -> List[Image.Image]:
    images = []

    with Image.open(BytesIO(file_bytes)) as img:
        for page in ImageSequence.Iterator(img):
            page = page.convert("RGB")
            page.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
            images.append(page.copy())

    if not images:
        raise ValueError("No image pages found in TIFF file.")

    return images


def convert_document_to_images(file_bytes: bytes, filename: str) -> List[Image.Image]:
    ext = get_file_extension(filename)

    if ext == ".pdf":
        return convert_pdf_to_images(file_bytes)

    if ext in (".tif", ".tiff"):
        return convert_tiff_to_images(file_bytes)

    raise ValueError(f"Unsupported file type: {ext}")


async def convert_document_to_images_async(file_bytes: bytes, filename: str) -> List[Image.Image]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        cpu_executor,
        convert_document_to_images,
        file_bytes,
        filename,
    )


def pil_image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ============================================================
# PADDLE OCR
# ============================================================

paddle_ocr = None


def get_paddle_ocr():
    global paddle_ocr

    if paddle_ocr is None:
        print("[OCR] Initializing PaddleOCR...")

        paddle_ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=True,
            rec_batch_num=4,
            det_db_thresh=0.2,
            det_db_box_thresh=0.3,
            det_db_unclip_ratio=2.0,

            show_log=False,
        )

        print("[OCR] PaddleOCR initialized")

    return paddle_ocr


def run_paddle_ocr_blocking(images: List[Image.Image]) -> Dict[str, List[Dict[str, Any]]]:
    ocr = get_paddle_ocr()
    ocr_result: Dict[str, List[Dict[str, Any]]] = {}

    for page_idx, image in enumerate(images, start=1):
        page_key = f"page_{page_idx}"

        image_np = np.array(image.convert("RGB")).copy()

        if image_np is None or image_np.size == 0:
            print(f"[OCR ERROR] Empty image for {page_key}")
            ocr_result[page_key] = []
            continue

        print(f"[OCR] {page_key} image shape: {image_np.shape}")

        try:
            result = ocr.ocr(image_np, cls=True)
        except Exception as e:
            print(f"[OCR ERROR] {page_key}: {e}")
            ocr_result[page_key] = []
            continue

        page_lines: List[Dict[str, Any]] = []

        if result and result[0]:
            for line in result[0]:
                bbox = line[0]
                text = line[1][0]
                conf = float(line[1][1])

                page_lines.append({
                    "text": text,
                    "confidence": round(conf, 4),
                    "bbox": bbox,
                    "center": bbox_center(bbox),
                })

        ocr_result[page_key] = page_lines

        print(f"[OCR] {page_key}: detected {len(page_lines)} lines")

    return ocr_result


async def run_paddle_ocr_async(images: List[Image.Image]) -> Dict[str, List[Dict[str, Any]]]:
    loop = asyncio.get_running_loop()

    return await loop.run_in_executor(
        cpu_executor,
        run_paddle_ocr_blocking,
        images,
    )

# ============================================================
# OCR MATCHING
# ============================================================

FIELD_LABEL_ALIASES = {
    "check_number": ["check number", "cheque number", "check no", "cheque no"],
    "check_amount": ["amount", "check amount", "cheque amount"],
    "pay_to": ["pay to the order of", "pay to", "payee"],
    "provider_name": ["provider", "provider name", "payer"],

    "patient_name": ["patient name", "name", "member name"],
    "first_name": ["first name"],
    "last_name": ["last name"],
    "dos": ["dos", "date of service", "service date"],
    "member_id": ["member id", "subscriber", "subscriber#", "policy id"],
    "claim_number": ["claim number", "claim no", "claim #", "icn"],
    "reason_for_refund": ["reason", "reason for refund", "description"],
    "claim_payment_amount": ["amount", "payment amount", "claim payment", "refund"],
    "payee_name": ["payee", "payee name", "provider"],

    "comp_benefits_detected": ["comp benefits", "comp benefit"],
    "has_patient_data": ["patient", "claim"],
}


class OCRFieldMatcher:
    def _find_field_anchors(
        self,
        field_name: str,
        ocr_lines: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        aliases = FIELD_LABEL_ALIASES.get(field_name, [])
        anchors = []

        for line in ocr_lines:
            text_norm = normalize_text(line["text"])
            for alias in aliases:
                if normalize_text(alias) in text_norm:
                    anchors.append(line)
                    break

        return anchors

    def _text_match_score(self, qwen_value: Any, ocr_text: str) -> float:
        q_raw = normalize_text(qwen_value)
        o_raw = normalize_text(ocr_text)

        q_code = normalize_code_like_text(qwen_value)
        o_code = normalize_code_like_text(ocr_text)

        if not q_raw or not o_raw:
            return 0.0

        if q_raw == o_raw:
            return 1.0

        if q_code and o_code and q_code == o_code:
            return 0.98

        if q_raw in o_raw or o_raw in q_raw:
            return 0.92

        if q_code and o_code and (q_code in o_code or o_code in q_code):
            return 0.90

        ratio_raw = SequenceMatcher(None, q_raw, o_raw).ratio()
        ratio_code = SequenceMatcher(None, q_code, o_code).ratio() if q_code and o_code else 0.0

        return max(ratio_raw, ratio_code)

    def _anchor_proximity_score(
        self,
        candidate_line: Dict[str, Any],
        anchors: List[Dict[str, Any]],
    ) -> float:
        if not anchors:
            return 0.0

        candidate_center = candidate_line["center"]
        distances = [
            euclidean_distance(candidate_center, anchor["center"])
            for anchor in anchors
        ]

        min_dist = min(distances)
        return math.exp(-min_dist / 250.0)

    def _calculate_proportional_bbox(
        self,
        full_bbox: List[List[float]],
        full_text: str,
        match_text: str,
    ) -> Optional[List[List[float]]]:
        full_norm = normalize_text(full_text)
        match_norm = normalize_text(match_text)

        if not full_norm or not match_norm:
            return full_bbox

        total_chars = len(full_norm)
        start_pos = -1
        end_pos = -1

        idx = full_norm.find(match_norm)
        if idx != -1:
            start_pos = idx
            end_pos = idx + len(match_norm)

        if start_pos == -1:
            full_code = normalize_code_like_text(full_text)
            match_code = normalize_code_like_text(match_text)

            if full_code and match_code:
                code_idx = full_code.find(match_code)
                if code_idx != -1:
                    code_to_norm = [
                        i for i, ch in enumerate(full_norm) if ch.isalnum()
                    ]

                    if code_idx < len(code_to_norm):
                        start_pos = code_to_norm[code_idx]
                        code_end = code_idx + len(match_code) - 1
                        end_pos = (
                            code_to_norm[code_end] + 1
                            if code_end < len(code_to_norm)
                            else total_chars
                        )

        if start_pos == -1:
            match_tokens = [t for t in match_norm.split() if len(t) > 1]

            if match_tokens:
                first_tok = match_tokens[0]
                last_tok = match_tokens[-1]

                fp = full_norm.find(first_tok)
                if fp != -1:
                    lp = full_norm.rfind(last_tok, fp)
                    start_pos = fp
                    end_pos = (
                        lp + len(last_tok)
                        if lp != -1
                        else min(fp + len(match_norm), total_chars)
                    )

        if start_pos == -1 or end_pos == -1:
            return full_bbox

        x_coords = [pt[0] for pt in full_bbox]
        y_coords = [pt[1] for pt in full_bbox]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        start_ratio = start_pos / total_chars
        end_ratio = end_pos / total_chars

        new_x_min = x_min + (x_max - x_min) * start_ratio
        new_x_max = x_min + (x_max - x_min) * end_ratio

        return [
            [new_x_min, y_min],
            [new_x_max, y_min],
            [new_x_max, y_max],
            [new_x_min, y_max],
        ]

    def match_value_to_ocr(
        self,
        field_name: str,
        qwen_value: Any,
        ocr_lines: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if qwen_value is None:
            return {
                "ocr_match_text": None,
                "ocr_confidence": None,
                "ocr_bbox": None,
                "ocr_match_score": 0.0,
            }

        qwen_value_str = str(qwen_value).strip()

        if not qwen_value_str:
            return {
                "ocr_match_text": None,
                "ocr_confidence": None,
                "ocr_bbox": None,
                "ocr_match_score": 0.0,
            }

        anchors = self._find_field_anchors(field_name, ocr_lines)

        best_line = None
        best_score = 0.0

        for line in ocr_lines:
            text_score = self._text_match_score(qwen_value_str, line["text"])
            proximity_score = self._anchor_proximity_score(line, anchors)

            total_score = (0.85 * text_score) + (0.15 * proximity_score)

            if total_score > best_score:
                best_score = total_score
                best_line = line

        if best_line is None or best_score < 0.55:
            return {
                "ocr_match_text": None,
                "ocr_confidence": None,
                "ocr_bbox": None,
                "ocr_match_score": round(best_score, 4),
            }

        refined_bbox = self._calculate_proportional_bbox(
            best_line["bbox"],
            best_line["text"],
            qwen_value_str,
        )

        return {
            "ocr_match_text": best_line["text"],
            "ocr_confidence": round(float(best_line["confidence"]), 4),
            "ocr_bbox": refined_bbox,
            "ocr_match_score": round(best_score, 4),
        }


ocr_matcher = OCRFieldMatcher()


def enrich_single_field(
    field_name: str,
    value: Any,
    ocr_lines: List[Dict[str, Any]],
) -> Dict[str, Any]:
    ocr_info = ocr_matcher.match_value_to_ocr(
        field_name=field_name,
        qwen_value=value,
        ocr_lines=ocr_lines,
    )

    review_required = False

    if value not in (None, "", [], {}) and ocr_info["ocr_match_text"] is None:
        review_required = True

    return {
        "value": value,
        "llm_confidence": None,
        "logprob": None,
        "ocr_match_text": ocr_info["ocr_match_text"],
        "ocr_confidence": ocr_info["ocr_confidence"],
        "ocr_bbox": ocr_info["ocr_bbox"],
        "ocr_match_score": ocr_info["ocr_match_score"],
        "review_required": review_required,
    }


def enrich_page_with_ocr(
    page_data: Dict[str, Any],
    ocr_lines: List[Dict[str, Any]],
) -> Dict[str, Any]:
    enriched: Dict[str, Any] = {}

    if DOCUMENT_TYPE == "patient":
        for key, value in page_data.items():
            if key == "patients":
                patient_list = []

                for patient in value:
                    enriched_patient = {}

                    for patient_field, patient_value in patient.items():
                        enriched_patient[patient_field] = enrich_single_field(
                            patient_field,
                            patient_value,
                            ocr_lines,
                        )

                    patient_list.append(enriched_patient)

                enriched["patients"] = patient_list
            else:
                enriched[key] = enrich_single_field(key, value, ocr_lines)

        return enriched

    for field_name, value in page_data.items():
        enriched[field_name] = enrich_single_field(
            field_name,
            value,
            ocr_lines,
        )

    return enriched


def match_qwen_with_ocr(
    qwen_extracted_data: Dict[str, Any],
    paddle_ocr_output: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    hybrid_data = {}

    for page_key, page_data in qwen_extracted_data.items():
        ocr_lines = paddle_ocr_output.get(page_key, [])
        hybrid_data[page_key] = enrich_page_with_ocr(page_data, ocr_lines)

    return hybrid_data


# ============================================================
# vLLM INFERENCE
# ============================================================

def extract_usage(response) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)

    if usage is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }

    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def call_vllm_qwen(image: Image.Image, prompt: str):
    image_url = pil_image_to_data_url(image)

    response = vllm_client.chat.completions.create(
        model=VLLM_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        max_tokens=MAX_TOKENS,
        temperature=0,
    )

    content = response.choices[0].message.content
    usage = extract_usage(response)

    return content, usage


async def call_vllm_qwen_async(image: Image.Image, prompt: str):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        call_vllm_qwen,
        image,
        prompt,
    )


def summarize_token_usage(token_usage: Dict[str, Any]) -> Dict[str, Any]:
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    has_any_value = False

    for _, usage in token_usage.items():
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        page_total_tokens = usage.get("total_tokens")

        if prompt_tokens is not None:
            total_prompt_tokens += prompt_tokens
            has_any_value = True

        if completion_tokens is not None:
            total_completion_tokens += completion_tokens
            has_any_value = True

        if page_total_tokens is not None:
            total_tokens += page_total_tokens
            has_any_value = True

    if not has_any_value:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }

    return {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
    }


async def process_single_page(
    idx: int,
    image: Image.Image,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    page_key = f"page_{idx}"

    page_start = now()
    print(f"[PAGE] {page_key} started at {current_timestamp()}")

    async with semaphore:
        raw_text, usage = await call_vllm_qwen_async(image, PROMPT)

    raw_outputs = {page_key: raw_text}
    token_usage = {page_key: usage}
    warnings = []
    had_warning = False

    print(f"\n===== RAW vLLM OUTPUT: {page_key} =====")
    print(raw_text)
    print("=====================================\n")

    print(f"\n===== TOKEN USAGE: {page_key} =====")
    print(json.dumps(usage, indent=2))
    print("===================================\n")

    ok, parsed, parse_error = safe_extract_json(raw_text)

    if not ok:
        print(f"[WARN] First JSON parse failed for {page_key}: {parse_error}")
        print(f"[INFO] Retrying {page_key} once...")

        async with semaphore:
            retry_raw_text, retry_usage = await call_vllm_qwen_async(image, PROMPT)

        raw_outputs[f"{page_key}_retry"] = retry_raw_text
        token_usage[f"{page_key}_retry"] = retry_usage

        print(f"\n===== RAW vLLM RETRY OUTPUT: {page_key} =====")
        print(retry_raw_text)
        print("===========================================\n")

        print(f"\n===== RETRY TOKEN USAGE: {page_key} =====")
        print(json.dumps(retry_usage, indent=2))
        print("=========================================\n")

        ok, parsed, retry_error = safe_extract_json(retry_raw_text)

        if not ok:
            had_warning = True
            warnings.append({
                "page": page_key,
                "type": "json_parse_failed",
                "first_error": parse_error,
                "retry_error": retry_error,
            })
            normalized = get_empty_result_schema()
        else:
            normalized = normalize_output(parsed, retry_raw_text)
    else:
        normalized = normalize_output(parsed, raw_text)

    if is_empty_extraction(normalized):
        had_warning = True
        warnings.append({
            "page": page_key,
            "type": "empty_extraction",
            "message": "Model returned valid response but all fields are empty.",
        })

    page_end = now()

    print(
        f"[PAGE] {page_key} finished at {current_timestamp()} "
        f"in {round(page_end - page_start, 3)} sec"
    )

    return {
        "page_key": page_key,
        "normalized": normalized,
        "raw_outputs": raw_outputs,
        "token_usage": token_usage,
        "warnings": warnings,
        "had_warning": had_warning,
        "page_time_sec": round(page_end - page_start, 3),
    }


async def run_vllm_inference_on_images(images: List[Image.Image]) -> Dict[str, Any]:
    extracted_data = {}
    raw_outputs = {}
    token_usage = {}
    warnings = []
    had_warning = False
    page_timings = {}

    print(
        f"[INFO] Starting async page inference. "
        f"pages={len(images)}, page_concurrency_limit={PAGE_CONCURRENCY_LIMIT}"
    )

    semaphore = asyncio.Semaphore(PAGE_CONCURRENCY_LIMIT)

    tasks = [
        process_single_page(idx, image, semaphore)
        for idx, image in enumerate(images, start=1)
    ]

    page_results = await asyncio.gather(*tasks)

    page_results = sorted(
        page_results,
        key=lambda x: int(x["page_key"].split("_")[1])
    )

    for page_result in page_results:
        page_key = page_result["page_key"]

        extracted_data[page_key] = page_result["normalized"]
        raw_outputs.update(page_result["raw_outputs"])
        token_usage.update(page_result["token_usage"])
        warnings.extend(page_result["warnings"])
        page_timings[page_key] = page_result["page_time_sec"]

        if page_result["had_warning"]:
            had_warning = True

    token_summary = summarize_token_usage(token_usage)

    return {
        "pages_processed": len(images),
        "document_type": DOCUMENT_TYPE,
        "extracted_data": extracted_data,
        "raw_outputs": raw_outputs,
        "token_usage": token_usage,
        "token_summary": token_summary,
        "warnings": warnings,
        "had_warning": had_warning,
        "page_timings": page_timings,
        "page_concurrency_limit": PAGE_CONCURRENCY_LIMIT,
    }


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_json_output(job_id: str, filename: str, result: Dict[str, Any]) -> str:
    base_name = os.path.splitext(os.path.basename(filename))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_{DOCUMENT_TYPE}_{job_id}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return out_path


def save_failed_raw(job_id: str, filename: str, error_text: str) -> str:
    base_name = os.path.splitext(os.path.basename(filename))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_{DOCUMENT_TYPE}_{job_id}_failed.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(error_text)

    return out_path


# ============================================================
# METRICS
# ============================================================

def count_total_ocr_lines(paddle_ocr_output: Dict[str, List[Dict[str, Any]]]) -> int:
    return sum(len(lines) for lines in paddle_ocr_output.values())


def finalize_metrics(
    job: Dict[str, Any],
    token_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metrics = {
        "queue_delay_sec": round_or_none(
            job["processing_started_at"] - job["submitted_at"]
            if job.get("processing_started_at") and job.get("submitted_at") else None
        ),
        "document_conversion_time_sec": round_or_none(
            job["document_conversion_finished_at"] - job["document_conversion_started_at"]
            if job.get("document_conversion_finished_at") and job.get("document_conversion_started_at") else None
        ),
        "paddle_ocr_time_sec": round_or_none(
            job["paddle_ocr_finished_at"] - job["paddle_ocr_started_at"]
            if job.get("paddle_ocr_finished_at") and job.get("paddle_ocr_started_at") else None
        ),
        "vllm_inference_time_sec": round_or_none(
            job["inference_finished_at"] - job["inference_started_at"]
            if job.get("inference_finished_at") and job.get("inference_started_at") else None
        ),
        "matching_time_sec": round_or_none(
            job["matching_finished_at"] - job["matching_started_at"]
            if job.get("matching_finished_at") and job.get("matching_started_at") else None
        ),
        "end_to_end_time_sec": round_or_none(
            job["completed_at"] - job["submitted_at"]
            if job.get("completed_at") and job.get("submitted_at") else None
        ),
        "tokens_per_sec": None,
    }

    if token_summary:
        completion_tokens = token_summary.get("completion_tokens")
        inference_time = metrics.get("vllm_inference_time_sec")

        if completion_tokens is not None and inference_time and inference_time > 0:
            metrics["tokens_per_sec"] = round(completion_tokens / inference_time, 3)

    return metrics


def log_metrics(
    job_id: str,
    filename: str,
    file_type: str,
    status: str,
    metrics: Dict[str, Any],
    token_summary: Dict[str, Any] | None = None,
    result: Dict[str, Any] | None = None,
    error: str | None = None,
):
    token_summary = token_summary or {}

    entry = {
        "timestamp": current_timestamp(),
        "job_id": job_id,
        "filename": filename,
        "file_type": file_type,
        "document_type": DOCUMENT_TYPE,
        "status": status,

        "queue_delay_sec": metrics.get("queue_delay_sec") if metrics else None,
        "document_conversion_time_sec": metrics.get("document_conversion_time_sec") if metrics else None,
        "paddle_ocr_time_sec": metrics.get("paddle_ocr_time_sec") if metrics else None,
        "vllm_inference_time_sec": metrics.get("vllm_inference_time_sec") if metrics else None,
        "matching_time_sec": metrics.get("matching_time_sec") if metrics else None,
        "end_to_end_time_sec": metrics.get("end_to_end_time_sec") if metrics else None,

        "prompt_tokens": token_summary.get("prompt_tokens"),
        "completion_tokens": token_summary.get("completion_tokens"),
        "total_tokens": token_summary.get("total_tokens"),

        "tokens_per_sec": metrics.get("tokens_per_sec") if metrics else None,

        "page_concurrency_limit": metrics.get("page_concurrency_limit") if metrics else None,
        "page_timings": metrics.get("page_timings") if metrics else None,

        "ocr_pages_processed": metrics.get("ocr_pages_processed") if metrics else None,
        "ocr_total_lines": metrics.get("ocr_total_lines") if metrics else None,

        "gpu_before_inference": metrics.get("gpu_before_inference") if metrics else None,
        "gpu_after_inference": metrics.get("gpu_after_inference") if metrics else None,

        "review_required": result.get("review_required") if result else None,
        "warnings": result.get("warnings") if result else None,
        "error": error,
    }

    with open(METRICS_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ============================================================
# BACKGROUND JOB PROCESSING
# ============================================================

async def process_job(job_id: str):
    job = jobs[job_id]

    try:
        print(f"[JOB] Started job {job_id}")

        job["status"] = "processing"
        job["processing_started_at"] = now()

        job["document_conversion_started_at"] = now()
        images = await convert_document_to_images_async(
            job["file_bytes"],
            job["filename"],
        )
        job["document_conversion_finished_at"] = now()

        print(f"[JOB] {job_id} converted document into {len(images)} page/image(s)")

        job["paddle_ocr_started_at"] = now()
        paddle_ocr_output = await run_paddle_ocr_async(images)
        job["paddle_ocr_finished_at"] = now()

        print(f"[JOB] {job_id} PaddleOCR completed")

        gpu_before = get_gpu_stats()

        log_gpu_metrics_csv(
            job_id=job_id,
            filename=job["filename"],
            file_type=job["file_type"],
            status="processing",
            stage="before_vllm_inference",
            gpu_stats=gpu_before,
        )

        job["inference_started_at"] = now()
        vllm_result = await run_vllm_inference_on_images(images)
        job["inference_finished_at"] = now()

        gpu_after = get_gpu_stats()

        log_gpu_metrics_csv(
            job_id=job_id,
            filename=job["filename"],
            file_type=job["file_type"],
            status="processing",
            stage="after_vllm_inference",
            gpu_stats=gpu_after,
        )

        job["matching_started_at"] = now()
        hybrid_extracted_data = match_qwen_with_ocr(
            qwen_extracted_data=vllm_result["extracted_data"],
            paddle_ocr_output=paddle_ocr_output,
        )
        job["matching_finished_at"] = now()

        token_summary = vllm_result.get("token_summary", {})

        has_ocr_review = False
        for page_data in hybrid_extracted_data.values():
            page_text = json.dumps(page_data)
            if '"review_required": true' in page_text.lower():
                has_ocr_review = True
                break

        clean_result = {
            "document_type": DOCUMENT_TYPE,
            "pages_processed": vllm_result["pages_processed"],
            "extracted_data": hybrid_extracted_data,
            "paddle_ocr_output": paddle_ocr_output,
            "raw_vllm_outputs": vllm_result.get("raw_outputs", {}),
            "token_summary": token_summary,
            "warnings": vllm_result.get("warnings", []),
            "review_required": bool(vllm_result.get("had_warning", False) or has_ocr_review),
        }

        job["status"] = (
            "completed_with_warning"
            if clean_result["review_required"]
            else "completed"
        )

        job["result"] = clean_result
        job["completed_at"] = now()
        job["metrics"] = finalize_metrics(job, token_summary=token_summary)

        job["metrics"]["gpu_before_inference"] = gpu_before
        job["metrics"]["gpu_after_inference"] = gpu_after
        job["metrics"]["page_timings"] = vllm_result.get("page_timings", {})
        job["metrics"]["page_concurrency_limit"] = vllm_result.get("page_concurrency_limit")
        job["metrics"]["ocr_pages_processed"] = len(paddle_ocr_output)
        job["metrics"]["ocr_total_lines"] = count_total_ocr_lines(paddle_ocr_output)

        job["json_file_path"] = save_json_output(
            job_id=job_id,
            filename=job["filename"],
            result=clean_result,
        )

        log_metrics(
            job_id=job_id,
            filename=job["filename"],
            file_type=job["file_type"],
            status=job["status"],
            metrics=job["metrics"],
            token_summary=token_summary,
            result=clean_result,
            error=None,
        )

        job["file_bytes"] = None

        print("\n====================================")
        print(f"[JOB] FINISHED: {job_id}")
        print(f"Status: {job['status']}")
        print(f"Saved JSON: {job['json_file_path']}")
        print(f"Metrics Log: {METRICS_LOG_FILE}")
        print(f"GPU CSV Log: {GPU_METRICS_CSV_FILE}")
        print(f"Page concurrency limit: {PAGE_CONCURRENCY_LIMIT}")
        print("Token Summary:")
        print(json.dumps(token_summary, indent=2))
        print("Page Timings:")
        print(json.dumps(vllm_result.get("page_timings", {}), indent=2))
        print("OCR Total Lines:")
        print(job["metrics"]["ocr_total_lines"])
        print("GPU Before:")
        print(json.dumps(gpu_before, indent=2))
        print("GPU After:")
        print(json.dumps(gpu_after, indent=2))
        print("Metrics:")
        print(json.dumps(job["metrics"], indent=2))
        print("====================================\n")

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = now()
        job["metrics"] = finalize_metrics(job, token_summary=None)
        job["file_bytes"] = None

        gpu_error_stats = get_gpu_stats()
        job["metrics"]["gpu_error_stage"] = gpu_error_stats

        failed_path = save_failed_raw(job_id, job["filename"], str(e))
        job["json_file_path"] = failed_path

        log_gpu_metrics_csv(
            job_id=job_id,
            filename=job["filename"],
            file_type=job.get("file_type", "unknown"),
            status="failed",
            stage="error",
            gpu_stats=gpu_error_stats,
        )

        log_metrics(
            job_id=job_id,
            filename=job["filename"],
            file_type=job.get("file_type", "unknown"),
            status=job["status"],
            metrics=job["metrics"],
            token_summary=None,
            result=None,
            error=str(e),
        )

        print(f"\n[JOB] FAILED: {job_id}")
        print(f"Error saved to: {failed_path}")
        print(f"Metrics Log: {METRICS_LOG_FILE}")
        print(f"GPU CSV Log: {GPU_METRICS_CSV_FILE}")
        print(str(e)[:3000])


# ============================================================
# STARTUP / SHUTDOWN
# ============================================================

@app.on_event("startup")
async def startup_event():
    init_gpu_csv_log()

    print("[INFO] Hybrid FastAPI started")
    print("[INFO] Pipeline: PDF/TIFF -> PaddleOCR -> vLLM Qwen -> OCR Matching")
    print("[INFO] vLLM internal queue/scheduler will handle inference requests")
    print(f"[INFO] Document type: {DOCUMENT_TYPE}")
    print(f"[INFO] Page concurrency limit: {PAGE_CONCURRENCY_LIMIT}")
    print(f"[INFO] Using vLLM server: {VLLM_BASE_URL}")
    print(f"[INFO] vLLM model name: {VLLM_MODEL_NAME}")
    print(f"[INFO] Metrics log file: {METRICS_LOG_FILE}")
    print(f"[INFO] GPU metrics CSV file: {GPU_METRICS_CSV_FILE}")


@app.on_event("shutdown")
async def shutdown_event():
    cpu_executor.shutdown(wait=False)
    job_executor.shutdown(wait=False)

def process_job_sync(job_id: str):
    asyncio.run(process_job(job_id))
# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def root():
    return {
        "message": "Hybrid PaddleOCR + vLLM document extraction backend is running",
        "pipeline": "document_conversion -> paddleocr -> vllm_qwen -> ocr_matching",
        "document_type": DOCUMENT_TYPE,
        "vllm_url": VLLM_BASE_URL,
        "vllm_model": VLLM_MODEL_NAME,
        "metrics_log_file": METRICS_LOG_FILE,
        "gpu_metrics_csv_file": GPU_METRICS_CSV_FILE,
        "page_concurrency_limit": PAGE_CONCURRENCY_LIMIT,
        "supported_extensions": list(SUPPORTED_EXTENSIONS),
        "possible_document_types": list(PROMPT_MAP.keys()),
        "possible_statuses": [
            "queued",
            "processing",
            "completed",
            "completed_with_warning",
            "failed",
        ],
    }


@app.post("/submit", response_model=SubmitResponse)
async def submit_file(file: UploadFile = File(...)):
    filename = file.filename or ""
    ext = get_file_extension(filename)

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TIF, and TIFF files are supported.",
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "queued",
        "result": None,
        "error": None,
        "metrics": None,
        "json_file_path": None,
        "file_bytes": file_bytes,
        "filename": filename,
        "file_type": ext.replace(".", ""),
        "document_type": DOCUMENT_TYPE,
        "submitted_at": now(),
        "processing_started_at": None,
        "document_conversion_started_at": None,
        "document_conversion_finished_at": None,
        "paddle_ocr_started_at": None,
        "paddle_ocr_finished_at": None,
        "inference_started_at": None,
        "inference_finished_at": None,
        "matching_started_at": None,
        "matching_finished_at": None,
        "completed_at": None,
    }

    loop = asyncio.get_running_loop()
    loop.run_in_executor(job_executor, process_job_sync, job_id)

    return SubmitResponse(
        job_id=job_id,
        status="queued",
        message=(
            f"File accepted. Document type: {DOCUMENT_TYPE}. "
            f"Page concurrency limit: {PAGE_CONCURRENCY_LIMIT}. "
            "Hybrid PaddleOCR + vLLM extraction started."
        ),
    )


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job ID not found.")

    job = jobs[job_id]

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        result=job["result"],
        error=job["error"],
        metrics=job["metrics"],
        json_file_path=job["json_file_path"],
    )
