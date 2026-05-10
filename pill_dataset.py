from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CLOUD_DATASET_DIR = Path("/shared-docker/amd_ai_hackathon/pill_data/pill")
DEFAULT_REBUILT_DATASET_DIR = BASE_DIR / "pill_data" / "pill_yolo"
DEFAULT_LOCAL_DATASET_DIR = BASE_DIR / "pill_data" / "pill"
DEFAULT_FIXED_DATASET_DIR = BASE_DIR / "pill_fixed"
GENERATED_DATASET_YAML = BASE_DIR / "data" / "generated_pill_dataset.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
YOLO_SPLITS = ("train", "valid", "val", "test")


@dataclass(frozen=True)
class DatasetImage:
    path: Path
    relative_path: str
    split: str


@dataclass(frozen=True)
class DatasetConfig:
    dataset_dir: Path
    train: str
    val: str
    test: str | None
    names: tuple[str, ...]


def dataset_candidates(explicit_path: str | Path | None = None) -> list[Path]:
    if explicit_path:
        return [Path(explicit_path)]

    candidates: list[Path] = []

    env_path = os.getenv("PILL_DATASET_DIR")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        (
            DEFAULT_CLOUD_DATASET_DIR,
            DEFAULT_REBUILT_DATASET_DIR,
            DEFAULT_FIXED_DATASET_DIR,
            DEFAULT_LOCAL_DATASET_DIR,
        )
    )
    return candidates


def find_dataset_dir(explicit_path: str | Path | None = None) -> Path | None:
    for candidate in dataset_candidates(explicit_path):
        if looks_like_yolo_dataset(candidate):
            return candidate.resolve()
    return None


def require_dataset_dir(explicit_path: str | Path | None = None) -> Path:
    dataset_dir = find_dataset_dir(explicit_path)
    if dataset_dir is None:
        checked = ", ".join(str(path) for path in dataset_candidates(explicit_path))
        raise FileNotFoundError(
            "Dataset pilules introuvable. Chemins verifies: " + checked
        )
    return dataset_dir


def looks_like_yolo_dataset(path: str | Path) -> bool:
    root = Path(path)
    return (root / "train").exists() and any(
        (root / split / "images").is_dir()
        for split in YOLO_SPLITS
    )


def split_image_dir(dataset_dir: Path, split: str) -> Path:
    return dataset_dir / split / "images"


def split_label_dir(dataset_dir: Path, split: str) -> Path:
    return dataset_dir / split / "labels"


def split_has_images(dataset_dir: Path, split: str) -> bool:
    image_dir = split_image_dir(dataset_dir, split)
    return image_dir.is_dir() and any(iter_image_files(image_dir))


def split_has_labels(dataset_dir: Path, split: str) -> bool:
    label_dir = split_label_dir(dataset_dir, split)
    return label_dir.is_dir() and any(label_dir.glob("*.txt"))


def iter_image_files(directory: Path):
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS:
            yield path


def choose_validation_split(dataset_dir: Path) -> str:
    for split in ("valid", "val", "test", "train"):
        if split_has_images(dataset_dir, split):
            return split
    raise FileNotFoundError("Aucun split image utilisable pour la validation.")


def choose_test_split(dataset_dir: Path) -> str | None:
    for split in ("test", "valid", "val"):
        if split_has_images(dataset_dir, split):
            return split
    return None


def infer_class_count(dataset_dir: Path) -> int:
    max_class_id = -1
    for split in YOLO_SPLITS:
        label_dir = split_label_dir(dataset_dir, split)
        if not label_dir.is_dir():
            continue
        for label_file in label_dir.glob("*.txt"):
            for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    continue
                max_class_id = max(max_class_id, class_id)

    return max(1, max_class_id + 1)


def normalize_class_names(class_names: str | list[str] | tuple[str, ...] | None, count: int) -> tuple[str, ...]:
    if class_names is None:
        return ("pill",) if count == 1 else tuple(f"class_{index}" for index in range(count))

    if isinstance(class_names, str):
        names = tuple(name.strip() for name in class_names.split(",") if name.strip())
    else:
        names = tuple(str(name).strip() for name in class_names if str(name).strip())

    if not names:
        return normalize_class_names(None, count)

    if len(names) < count:
        names = names + tuple(f"class_{index}" for index in range(len(names), count))

    return names[:count]


def strip_yaml_scalar(value: str) -> str:
    value = value.strip()
    if not value:
        return value
    if value[0] in {"'", '"'} and value[-1:] == value[0]:
        value = value[1:-1]
    return value.replace('\\"', '"').replace("\\'", "'")


def read_dataset_class_names(dataset_dir: Path) -> tuple[str, ...] | None:
    classes_file = dataset_dir / "classes.txt"
    if classes_file.is_file():
        names = tuple(
            line.strip()
            for line in classes_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        )
        if names:
            return names

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        return None

    names_by_id: dict[int, str] = {}
    in_names_block = False
    for line in data_yaml.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "names:":
            in_names_block = True
            continue
        if in_names_block:
            if not line[:1].isspace():
                break
            if ":" not in stripped:
                continue
            raw_index, raw_name = stripped.split(":", 1)
            try:
                index = int(raw_index)
            except ValueError:
                continue
            names_by_id[index] = strip_yaml_scalar(raw_name)

    if not names_by_id:
        return None
    return tuple(names_by_id[index] for index in sorted(names_by_id))


def build_dataset_config(
    dataset_dir: str | Path | None = None,
    class_names: str | list[str] | tuple[str, ...] | None = None,
) -> DatasetConfig:
    root = require_dataset_dir(dataset_dir)
    if not split_has_images(root, "train"):
        raise FileNotFoundError(f"Split train/images introuvable dans {root}")

    val_split = choose_validation_split(root)
    test_split = choose_test_split(root)
    class_count = infer_class_count(root)
    if class_names is None:
        class_names = read_dataset_class_names(root)
    names = normalize_class_names(class_names, class_count)

    return DatasetConfig(
        dataset_dir=root,
        train="train/images",
        val=f"{val_split}/images",
        test=f"{test_split}/images" if test_split else None,
        names=names,
    )


def write_dataset_yaml(
    output_path: str | Path = GENERATED_DATASET_YAML,
    dataset_dir: str | Path | None = None,
    class_names: str | list[str] | tuple[str, ...] | None = None,
) -> Path:
    config = build_dataset_config(dataset_dir=dataset_dir, class_names=class_names)
    output = Path(output_path)
    if not output.is_absolute():
        output = BASE_DIR / output
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"path: {config.dataset_dir.as_posix()}",
        f"train: {config.train}",
        f"val: {config.val}",
    ]
    if config.test is not None:
        lines.append(f"test: {config.test}")
    lines.append("names:")
    for index, name in enumerate(config.names):
        lines.append(f"  {index}: {name}")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def discover_dataset_images(
    dataset_dir: str | Path | None = None,
    limit: int = 24,
    preferred_splits: tuple[str, ...] = ("test", "valid", "val", "train"),
) -> list[DatasetImage]:
    root = find_dataset_dir(dataset_dir)
    if root is None:
        return []

    images: list[DatasetImage] = []
    for split in preferred_splits:
        image_dir = split_image_dir(root, split)
        if not image_dir.is_dir():
            continue
        for image_path in iter_image_files(image_dir):
            images.append(
                DatasetImage(
                    path=image_path,
                    relative_path=image_path.relative_to(root).as_posix(),
                    split=split,
                )
            )
            if len(images) >= limit:
                return images

    return images


def resolve_dataset_image(
    relative_or_absolute_path: str | Path,
    dataset_dir: str | Path | None = None,
) -> Path | None:
    root = find_dataset_dir(dataset_dir)
    if root is None:
        return None

    raw_path = Path(relative_or_absolute_path)
    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None

    if resolved.is_file() and resolved.suffix.casefold() in IMAGE_EXTENSIONS:
        return resolved
    return None
