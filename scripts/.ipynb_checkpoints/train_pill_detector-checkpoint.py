from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pill_dataset import GENERATED_DATASET_YAML, write_dataset_yaml


DEFAULT_OUTPUT_MODEL = Path("models") / "pill_detector.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrainer un detecteur YOLO sur le dataset pill_data/pill."
    )
    parser.add_argument(
        "--dataset",
        help=(
            "Chemin du dataset YOLO. Par defaut: PILL_DATASET_DIR, puis "
            "/shared-docker/amd_ai_hackathon/pill_data/pill, puis "
            "./pill_data/pill_yolo, ./pill_fixed, ./pill_data/pill."
        ),
    )
    parser.add_argument("--base-model", default="yolo11n.pt", help="Poids de depart YOLO.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Ex: 0, cpu. Laissez vide pour auto.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--class-names",
        default=None,
        help="Noms separes par virgule. Defaut: classes.txt/data.yaml du dataset, sinon inference.",
    )
    parser.add_argument("--data-yaml", default=str(GENERATED_DATASET_YAML))
    parser.add_argument("--project", default="runs/pill_detector")
    parser.add_argument("--name", default="train")
    parser.add_argument("--output-model", default=str(DEFAULT_OUTPUT_MODEL))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verifie le dataset et genere le YAML sans lancer l'entrainement.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        data_yaml = write_dataset_yaml(
            output_path=args.data_yaml,
            dataset_dir=args.dataset,
            class_names=args.class_names,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Dataset YAML genere: {data_yaml}")

    if args.dry_run:
        print("Dry-run termine: entrainement non lance.")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics n'est pas installe. Lancez: "
            "python -m pip install -r requirements-vision.txt"
        ) from exc

    model = YOLO(args.base_model)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
    }
    if args.device:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best_model = save_dir / "weights" / "best.pt"
    if not best_model.exists():
        raise SystemExit(f"Modele entraine introuvable: {best_model}")

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model, output_model)
    print(f"Modele exporte pour l'application: {output_model}")
    print("Utilisation runtime: VISION_MODEL_PATH=" + str(output_model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
