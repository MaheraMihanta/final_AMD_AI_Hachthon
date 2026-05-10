from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "pill_detector.pt"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class VisionDetection:
    label: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class VisionResult:
    enabled: bool
    ok: bool
    message: str
    model_path: str | None = None
    image_path: str | None = None
    detections: tuple[VisionDetection, ...] = ()


def configured_model_path(model_path: str | Path | None = None) -> Path:
    if model_path is not None:
        return Path(model_path)
    return Path(os.getenv("VISION_MODEL_PATH", str(DEFAULT_MODEL_PATH)))


def configured_confidence(confidence: float | None = None) -> float:
    if confidence is not None:
        return confidence
    raw_value = os.getenv("VISION_CONF_THRESHOLD", "0.25")
    try:
        return float(raw_value)
    except ValueError:
        return 0.25


@lru_cache(maxsize=2)
def load_yolo_model(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


def process_image(
    image_path: str | Path,
    model_path: str | Path | None = None,
    confidence: float | None = None,
) -> VisionResult:
    """Run the trained pill detector on one image."""
    image = Path(image_path)
    model = configured_model_path(model_path)
    conf_threshold = configured_confidence(confidence)

    if not image.exists():
        return VisionResult(
            enabled=False,
            ok=False,
            message=f"Image introuvable: {image}",
            model_path=str(model),
            image_path=str(image),
        )

    if image.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
        return VisionResult(
            enabled=True,
            ok=False,
            message=(
                f"Format non compatible avec le modele vision: {image.suffix}. "
                "Associez une photo JPG/PNG du dataset a l'article pour tester la vision."
            ),
            model_path=str(model),
            image_path=str(image),
        )

    if not model.exists():
        return VisionResult(
            enabled=False,
            ok=False,
            message=(
                f"Modele vision introuvable: {model}. Lancez "
                "python scripts/train_pill_detector.py sur le Cloud AMD."
            ),
            model_path=str(model),
            image_path=str(image),
        )

    try:
        yolo_model = load_yolo_model(str(model))
    except ImportError:
        return VisionResult(
            enabled=False,
            ok=False,
            message=(
                "Ultralytics n'est pas installe. Lancez: "
                "python -m pip install -r requirements-vision.txt"
            ),
            model_path=str(model),
            image_path=str(image),
        )
    except Exception as exc:
        return VisionResult(
            enabled=False,
            ok=False,
            message=f"Impossible de charger le modele vision: {exc}",
            model_path=str(model),
            image_path=str(image),
        )

    try:
        results = yolo_model.predict(
            source=str(image),
            conf=conf_threshold,
            verbose=False,
        )
    except Exception as exc:
        return VisionResult(
            enabled=True,
            ok=False,
            message=f"Erreur pendant l'inference vision: {exc}",
            model_path=str(model),
            image_path=str(image),
        )

    detections = parse_yolo_detections(results)
    if detections:
        return VisionResult(
            enabled=True,
            ok=True,
            message=f"Vision OK: {len(detections)} detection(s) au-dessus de {conf_threshold:.2f}.",
            model_path=str(model),
            image_path=str(image),
            detections=detections,
        )

    return VisionResult(
        enabled=True,
        ok=False,
        message=f"Vision KO: aucune pilule detectee au-dessus de {conf_threshold:.2f}.",
        model_path=str(model),
        image_path=str(image),
    )


def parse_yolo_detections(results) -> tuple[VisionDetection, ...]:
    detections: list[VisionDetection] = []
    for result in results:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        for box in boxes:
            class_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else 0
            confidence = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0
            xyxy_values = box.xyxy[0].tolist()
            label = str(names.get(class_id, f"class_{class_id}"))
            detections.append(
                VisionDetection(
                    label=label,
                    confidence=confidence,
                    box_xyxy=tuple(float(value) for value in xyxy_values[:4]),
                )
            )

    return tuple(detections)