"""Mesure la précision/couverture d'un classifieur de niveaux selon son seuil."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("weights", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = ROOT / "training" / "classification" / args.dataset / args.split
    weights = args.weights if args.weights.is_absolute() else ROOT / args.weights
    samples = sorted(
        (path, int(path.parent.name.removeprefix("level-")))
        for path in data.glob("level-*/*")
        if path.is_file()
    )
    if not samples:
        raise SystemExit(f"Aucun exemple dans {data}")

    model = YOLO(str(weights))
    rows = []
    confusion: dict[int, Counter[int]] = defaultdict(Counter)
    predictions = model.predict(
        source=[str(path) for path, _ in samples],
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        verbose=False,
        stream=True,
    )
    for (path, expected), result in zip(samples, predictions, strict=True):
        predicted = int(model.names[int(result.probs.top1)].removeprefix("level-"))
        confidence = float(result.probs.top1conf)
        rows.append({
            "path": str(path.relative_to(ROOT)),
            "expected": expected,
            "predicted": predicted,
            "confidence": confidence,
            "correct": predicted == expected,
        })
        confusion[expected][predicted] += 1

    thresholds = []
    for threshold in [value / 100 for value in range(50, 100)]:
        accepted = [row for row in rows if row["confidence"] >= threshold]
        correct = sum(row["correct"] for row in accepted)
        thresholds.append({
            "threshold": threshold,
            "accepted": len(accepted),
            "coverage": len(accepted) / len(rows),
            "accuracy": correct / len(accepted) if accepted else None,
        })

    report = {
        "dataset": args.dataset,
        "weights": str(weights.relative_to(ROOT)),
        "split": args.split,
        "samples": len(rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "per_level": {
            str(level): dict(sorted(guesses.items()))
            for level, guesses in sorted(confusion.items())
        },
        "thresholds": thresholds,
        "errors": [row for row in rows if not row["correct"]],
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"Rapport: {output}")
    print(rendered)


if __name__ == "__main__":
    main()
