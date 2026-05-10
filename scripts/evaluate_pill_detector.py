from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_CSV = (
    PROJECT_ROOT / "runs" / "detect" / "runs" / "pill_detector" / "train" / "results.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "pill_detector_performance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generer les plots et le resume des performances du detecteur de pilules."
    )
    parser.add_argument("--results-csv", default=str(DEFAULT_RESULTS_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def normalized_row(row: dict[str, str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for raw_key, raw_value in row.items():
        key = raw_key.strip()
        value = raw_value.strip()
        if not key or not value:
            continue
        try:
            parsed[key] = float(value)
        except ValueError:
            continue
    return parsed


def load_rows(results_csv: Path) -> list[dict[str, float]]:
    with results_csv.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return [normalized_row(row) for row in reader]


def get_series(rows: list[dict[str, float]], key: str) -> list[float]:
    return [row[key] for row in rows if key in row and math.isfinite(row[key])]


def best_row(rows: list[dict[str, float]], metric: str) -> dict[str, float]:
    return max(rows, key=lambda row: row.get(metric, float("-inf")))


def final_row(rows: list[dict[str, float]]) -> dict[str, float]:
    return rows[-1]


def metric_delta(rows: list[dict[str, float]], metric: str) -> float | None:
    values = get_series(rows, metric)
    if len(values) < 2:
        return None
    return values[-1] - values[0]


def plot_curves(rows: list[dict[str, float]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = get_series(rows, "epoch")
    plot_paths: list[Path] = []

    metric_groups = [
        (
            "detection_metrics.png",
            "Detection metrics",
            [
                ("metrics/precision(B)", "Precision"),
                ("metrics/recall(B)", "Recall"),
                ("metrics/mAP50(B)", "mAP50"),
                ("metrics/mAP50-95(B)", "mAP50-95"),
            ],
            "Score",
        ),
        (
            "training_losses.png",
            "Training losses",
            [
                ("train/box_loss", "Train box"),
                ("train/cls_loss", "Train cls"),
                ("train/dfl_loss", "Train dfl"),
            ],
            "Loss",
        ),
        (
            "validation_losses.png",
            "Validation losses",
            [
                ("val/box_loss", "Val box"),
                ("val/cls_loss", "Val cls"),
                ("val/dfl_loss", "Val dfl"),
            ],
            "Loss",
        ),
        (
            "learning_rates.png",
            "Learning rates",
            [
                ("lr/pg0", "pg0"),
                ("lr/pg1", "pg1"),
                ("lr/pg2", "pg2"),
            ],
            "LR",
        ),
    ]

    for filename, title, series_defs, ylabel in metric_groups:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for key, label in series_defs:
            values = get_series(rows, key)
            if values:
                ax.plot(epochs[: len(values)], values, linewidth=2, label=label)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        plot_paths.append(output_path)

    return plot_paths


def build_summary(rows: list[dict[str, float]]) -> dict[str, object]:
    best_map50 = best_row(rows, "metrics/mAP50(B)")
    best_map5095 = best_row(rows, "metrics/mAP50-95(B)")
    final = final_row(rows)

    return {
        "epochs": int(final.get("epoch", len(rows))),
        "training_time_seconds": round(final.get("time", 0.0), 3),
        "best_map50": {
            "epoch": int(best_map50.get("epoch", 0)),
            "value": round(best_map50.get("metrics/mAP50(B)", 0.0), 5),
        },
        "best_map50_95": {
            "epoch": int(best_map5095.get("epoch", 0)),
            "value": round(best_map5095.get("metrics/mAP50-95(B)", 0.0), 5),
        },
        "final_metrics": {
            "precision": round(final.get("metrics/precision(B)", 0.0), 5),
            "recall": round(final.get("metrics/recall(B)", 0.0), 5),
            "map50": round(final.get("metrics/mAP50(B)", 0.0), 5),
            "map50_95": round(final.get("metrics/mAP50-95(B)", 0.0), 5),
            "train_box_loss": round(final.get("train/box_loss", 0.0), 5),
            "val_box_loss": round(final.get("val/box_loss", 0.0), 5),
        },
        "improvement": {
            "precision_delta": round(metric_delta(rows, "metrics/precision(B)") or 0.0, 5),
            "recall_delta": round(metric_delta(rows, "metrics/recall(B)") or 0.0, 5),
            "map50_delta": round(metric_delta(rows, "metrics/mAP50(B)") or 0.0, 5),
            "map50_95_delta": round(metric_delta(rows, "metrics/mAP50-95(B)") or 0.0, 5),
        },
    }


def write_markdown(summary: dict[str, object], plot_paths: list[Path], output_dir: Path) -> Path:
    final_metrics = summary["final_metrics"]
    best_map50 = summary["best_map50"]
    best_map5095 = summary["best_map50_95"]
    improvement = summary["improvement"]

    lines = [
        "# Rapport performance - detecteur de pilules",
        "",
        f"- Epochs: {summary['epochs']}",
        f"- Temps d'entrainement: {summary['training_time_seconds']} s",
        f"- Meilleur mAP50: {best_map50['value']} a l'epoch {best_map50['epoch']}",
        f"- Meilleur mAP50-95: {best_map5095['value']} a l'epoch {best_map5095['epoch']}",
        "",
        "## Metriques finales",
        "",
        f"- Precision: {final_metrics['precision']}",
        f"- Recall: {final_metrics['recall']}",
        f"- mAP50: {final_metrics['map50']}",
        f"- mAP50-95: {final_metrics['map50_95']}",
        f"- Train box loss: {final_metrics['train_box_loss']}",
        f"- Val box loss: {final_metrics['val_box_loss']}",
        "",
        "## Progression epoch 1 -> finale",
        "",
        f"- Precision: {improvement['precision_delta']:+.5f}",
        f"- Recall: {improvement['recall_delta']:+.5f}",
        f"- mAP50: {improvement['map50_delta']:+.5f}",
        f"- mAP50-95: {improvement['map50_95_delta']:+.5f}",
        "",
        "## Plots generes",
        "",
    ]
    for path in plot_paths:
        lines.append(f"- `{path.name}`")

    report_path = output_dir / "summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    results_csv = Path(args.results_csv)
    output_dir = Path(args.output_dir)
    if not results_csv.is_absolute():
        results_csv = PROJECT_ROOT / results_csv
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    if not results_csv.is_file():
        raise SystemExit(f"Fichier results.csv introuvable: {results_csv}")

    rows = load_rows(results_csv)
    if not rows:
        raise SystemExit(f"Aucune ligne de metriques dans: {results_csv}")

    plot_paths = plot_curves(rows, output_dir)
    summary = build_summary(rows)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path = write_markdown(summary, plot_paths, output_dir)

    print(f"Rapport JSON: {summary_path}")
    print(f"Rapport Markdown: {markdown_path}")
    for path in plot_paths:
        print(f"Plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
