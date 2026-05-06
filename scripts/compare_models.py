"""
Aggregate experiment reports across models into a single comparison table.

Expected input reports:
results/experiments/<model_name>/<experiment_name>/reports/metrics_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build cross-model comparison report")
    parser.add_argument("--results_root", default="results/experiments", help="Experiments root folder")
    parser.add_argument("--output_dir", default="results/comparison", help="Comparison output folder")
    parser.add_argument("--sort_by", default="dice", choices=["dice", "iou", "f1", "inference_ms_per_slice"])
    return parser.parse_args()


def discover_reports(root: Path):
    reports = []
    for path in root.glob("*/**/reports/metrics_summary.json"):
        try:
            reports.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return reports


def main():
    args = parse_args()
    root = Path(args.results_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_reports(root)
    if not discovered:
        raise SystemExit(f"No reports found under {root}")

    rows = []
    for path, rep in discovered:
        rows.append(
            {
                "model_name": rep.get("model_name", path.parts[-4]),
                "model_display_name": rep.get("model_display_name", rep.get("model_name", "unknown")),
                "experiment": path.parts[-3],
                "dice": rep.get("dice", 0.0),
                "iou": rep.get("iou", 0.0),
                "precision": rep.get("precision", 0.0),
                "recall": rep.get("recall", 0.0),
                "f1": rep.get("f1", 0.0),
                "accuracy": rep.get("accuracy", 0.0),
                "total_parameters": rep.get("total_parameters", 0),
                "trainable_parameters": rep.get("trainable_parameters", 0),
                "inference_ms_per_slice": rep.get("inference_ms_per_slice", 0.0),
                "report_path": str(path),
            }
        )

    reverse = args.sort_by != "inference_ms_per_slice"
    rows.sort(key=lambda x: x.get(args.sort_by, 0.0), reverse=reverse)

    json_path = out_dir / "comparison_report.json"
    csv_path = out_dir / "comparison_report.csv"
    md_path = out_dir / "comparison_report.md"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Model Comparison Report",
        "",
        f"Sorted by `{args.sort_by}`",
        "",
        "| Model | Experiment | Dice | IoU | Precision | Recall | F1 | Params(M) | Infer(ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model_display_name']} | {r['experiment']} | "
            f"{r['dice']:.4f} | {r['iou']:.4f} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['f1']:.4f} | {r['total_parameters']/1e6:.2f} | "
            f"{r['inference_ms_per_slice']:.2f} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
