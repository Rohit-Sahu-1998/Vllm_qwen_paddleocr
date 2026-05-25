import os
import time
import random
from locust import HttpUser, task, between


DOC_DIR = "/home/rohit.sahu/Qwen_model/cpt_codes/vllm_based_locust/cash_docs"

DOC_PATHS = [
    os.path.join(DOC_DIR, f)
    for f in os.listdir(DOC_DIR)
    if f.lower().endswith((".pdf", ".tif", ".tiff"))
]

print(f"[INFO] Total documents found: {len(DOC_PATHS)}")


def get_mime_type(file_path: str) -> str:
    file_path_lower = file_path.lower()

    if file_path_lower.endswith(".pdf"):
        return "application/pdf"

    if file_path_lower.endswith((".tif", ".tiff")):
        return "image/tiff"

    return "application/octet-stream"


class HybridExtractionUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def submit_document(self):
        if not DOC_PATHS:
            print("[ERROR] No PDF/TIF/TIFF files found in folder")
            return

        doc_path = random.choice(DOC_PATHS)
        filename = os.path.basename(doc_path)
        mime_type = get_mime_type(doc_path)

        try:
            with open(doc_path, "rb") as f:
                files = {
                    "file": (
                        filename,
                        f,
                        mime_type,
                    )
                }

                response = self.client.post(
                    "/submit",
                    files=files,
                    timeout=80,
                    name="/submit",
                )

            if response.status_code != 200:
                print("[ERROR] Submit failed:", response.status_code, response.text[:500])
                return

            data = response.json()
            job_id = data.get("job_id")

            if not job_id:
                print("[ERROR] No job_id returned:", data)
                return

            print(f"[SUBMITTED] {job_id} | file={filename}")

            for _ in range(300):
                status_response = self.client.get(
                    f"/status/{job_id}",
                    timeout=30,
                    name="/status/{job_id}",
                )

                if status_response.status_code != 200:
                    print("[ERROR] Status failed:", status_response.status_code)
                    return

                status_data = status_response.json()
                status = status_data.get("status")

                if status in ("completed", "completed_with_warning"):
                    metrics = status_data.get("metrics") or {}

                    print(
                        f"[DONE] {job_id} | {status} | "
                        f"ocr={metrics.get('paddle_ocr_time_sec')}s | "
                        f"vllm={metrics.get('vllm_inference_time_sec')}s | "
                        f"match={metrics.get('matching_time_sec')}s | "
                        f"total={metrics.get('end_to_end_time_sec')}s"
                    )
                    return

                if status == "failed":
                    print("[FAILED JOB]", status_data.get("error"))
                    return

                time.sleep(1)

            print("[TIMEOUT]", job_id)

        except Exception as e:
            print("[EXCEPTION]", repr(e))
