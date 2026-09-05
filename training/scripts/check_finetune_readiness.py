"""Décide si un nouveau fine-tune est justifié par des données validées."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"


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
                    }
                )

    reasons = []
    if not baseline:
        reasons.append("baseline VAL 800+TTA absente")
    if not human_annotations:
        reasons.append("aucune nouvelle session humaine exhaustive importée")
    if split_report.get("cross_split_same_village_group_pairs"):
        reasons.append("village_group présent dans plusieurs splits")

    payload = {
        "ready": not reasons,
        "decision": "fine-tune autorisé" if not reasons else "conserver V5",
        "reasons": reasons,
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
            "Ne pas entraîner sur les seuls pseudo-labels ou sur le même dataset "
            "sans nouvelles annotations exhaustives validées."
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
