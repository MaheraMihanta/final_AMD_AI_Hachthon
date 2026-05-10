from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pill_dataset import discover_dataset_images, resolve_dataset_image
from vision import process_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tester le module vision sur une image.")
    parser.add_argument(
        "--image",
        help=(
            "Image a tester. Accepte un chemin absolu ou un chemin relatif au "
            "dataset, par exemple test/images/example.jpg."
        ),
    )
    parser.add_argument("--dataset", help="Chemin du dataset si different du defaut.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = None

    if args.image:
        image_path = resolve_dataset_image(args.image, args.dataset)
        if image_path is None:
            image_path = args.image
    else:
        samples = discover_dataset_images(args.dataset, limit=1)
        if samples:
            image_path = samples[0].path

    if image_path is None:
        raise SystemExit("Aucune image dataset trouvee. Verifiez PILL_DATASET_DIR.")

    result = process_image(image_path)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
