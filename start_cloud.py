from __future__ import annotations

import os
from pathlib import Path

from cloud_app import DEFAULT_HOST, DEFAULT_PORT, run_server


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = BASE_DIR / "pill_data" / "pill_yolo"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "pill_detector.pt"
DEFAULT_POSITIVE_LABELS = "Pill Back,Pill Front"


def configure_vision_runtime() -> None:
    os.environ.setdefault("PILL_DATASET_DIR", str(DEFAULT_DATASET_DIR))
    os.environ.setdefault("VISION_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    os.environ.setdefault("VISION_POSITIVE_LABELS", DEFAULT_POSITIVE_LABELS)


def main() -> None:
    configure_vision_runtime()
    host = os.getenv("HOST", DEFAULT_HOST)
    port = int(os.getenv("PORT", str(DEFAULT_PORT)))
    print("Dataset vision:", os.getenv("PILL_DATASET_DIR"))
    print("Modele vision:", os.getenv("VISION_MODEL_PATH"))
    print("Classes vision positives:", os.getenv("VISION_POSITIVE_LABELS"))
    run_server(host=host, port=port)


if __name__ == "__main__":
    main()
