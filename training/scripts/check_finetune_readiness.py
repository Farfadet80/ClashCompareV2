"""Décide si un nouveau fine-tune est justifié par des données validées."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"

# Seuils TH10–18 avant de considérer un FT utile (hors gate Jimmy)
MIN_WALL_TRAIN = 40
MIN_WALL_EVAL = 5  # val+test boxes
MIN_VERY_WEAK_TRAIN = 50
PRIORITY_WEAK = ("wall", "tornado-trap", "giga-bomb")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=(
            TRAINING
            / "runs"
            / "evaluations"
            / "v5-baseline-val-800-tta-20260905-data-plan"
            / "metrics.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TRAINING / "reports" / "finetune-readiness.json",
    )
    args = parser.parse_args()

    split_report = json.loads(
        (TRAINING / "reports" / "split-integrity.json").read_text(encoding="utf-8")
    )
    coverage = json.loads(
        (TRAINING / "reports" / "class-coverage.json").read_text(encoding="utf-8")
    )
    baseline = (
        json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
        if args.baseline_metrics.exists()
        else None
    )
    provenance_dir = TRAINING / "dataset" / "detector" / "provenance"
    human_annotations = []
    if provenance_dir.exists():
        for path in sorted(provenance_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("exhaustive") is True:
                human_annotations.append(
                    {
                        "file": path.relative_to(ROOT).as_posix(),
                        "box_count": int(payload.get("box_count", 0)),
                        "village_group": payload.get("village_group"),
                        "split": payload.get("split", "train"),
                    }
                )

    rows = {row["id"]: row for row in coverage.get("rows") or []}
    reasons = []
    data_gaps = []
    if not baseline:
        reasons.append("baseline VAL 800+TTA absente")
    if not human_annotations:
        reasons.append("aucune nouvelle session humaine exhaustive importée")
    if split_report.get("cross_split_same_village_group_pairs"):
        reasons.append("village_group présent dans plusieurs splits")

    wall = rows.get("wall") or {}
    wall_eval = int(wall.get("val") or 0) + int(wall.get("test") or 0)
    if int(wall.get("train") or 0) < MIN_WALL_TRAIN:
        data_gaps.append(f"wall train < {MIN_WALL_TRAIN} ({wall.get('train', 0)})")
    if wall_eval < MIN_WALL_EVAL:
        data_gaps.append(f"wall val+test < {MIN_WALL_EVAL} ({wall_eval})")

    for name in PRIORITY_WEAK:
        if name == "wall":
            continue
        row = rows.get(name) or {}
        if int(row.get("train") or 0) < MIN_VERY_WEAK_TRAIN:
            data_gaps.append(
                f"{name} train < {MIN_VERY_WEAK_TRAIN} ({row.get('train', 0)})"
            )

    mechanical_ready = not reasons
    useful_ready = mechanical_ready and not data_gaps

    payload = {
        "ready": useful_ready,
        "mechanical_ready": mechanical_ready,
        "decision": (
            "fine-tune données OK — attendre OK Jimmy"
            if useful_ready
            else "conserver V5 — données insuffisantes pour FT utile"
        ),
        "jimmy_gate": {
            "train_authorized": False,
            "reason": (
                "OK Jimmy explicite toujours requis (préférence projet). "
                "Ne pas lancer train_detector sans validation verbale."
            ),
        },
        "reasons": reasons,
        "data_gaps": data_gaps,
        "thresholds": {
            "min_wall_train": MIN_WALL_TRAIN,
            "min_wall_eval_boxes": MIN_WALL_EVAL,
            "min_very_weak_train": MIN_VERY_WEAK_TRAIN,
            "priority_weak": list(PRIORITY_WEAK),
        },
        "baseline": None
        if not baseline
        else {
            "map50": baseline.get("map50"),
            "map50_95": baseline.get("map50_95"),
            "imgsz": baseline.get("imgsz"),
            "augment_tta": baseline.get("augment_tta"),
        },
        "clean_train_images": split_report.get("train_images_clean"),
        "human_annotations": human_annotations,
        "coverage_blockers": {
            "unlearnable": coverage.get("unlearnable", []),
            "very_weak": coverage.get("very_weak", []),
            "no_eval_split": coverage.get("no_eval_split", []),
        },
        "policy": (
            "Ne pas entraîner sur les seuls pseudo-labels, sprites mur seuls, "
            "ou villages non exhaustifs. Gate Jimmy obligatoire même si ready=true."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Rapport: {args.output}")


if __name__ == "__main__":
    main()
