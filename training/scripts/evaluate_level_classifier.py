"""Évalue un classifieur de niveaux sur les crops réservés au test."""

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
    parser.add_argument("building")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    weights = ROOT / "training" / "runs" / "levels" / args.building / "weights" / "best.pt"
    manifest = ROOT / "training" / "dataset" / "level-splits.jsonl"
    if not weights.exists():
        raise SystemExit(f"Poids absents: {weights}")

    model = YOLO(str(weights))
    known_levels = {
        int(name.removeprefix("level-")): class_id
        for class_id, name in model.names.items()
        if name.startswith("level-")
    }
    samples: list[tuple[Path, int]] = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["building"] != args.building or row["split"] != args.split:
                continue
            level = int(row["level"])
            path = ROOT / row["path"]
            if level in known_levels and path.exists():
                samples.append((path, level))

    if not samples:
        raise SystemExit(f"Aucun exemple {args.split!r} compatible pour {args.building}")

    correct = Counter()
    total = Counter()
    confusion: dict[int, Counter[int]] = defaultdict(Counter)
    predictions = model.predict(
        source=[str(path) for path, _ in samples],
        imgsz=224,
        batch=args.batch,
        device=0,
        verbose=False,
        stream=True,
    )
    for (_, expected), result in zip(samples, predictions, strict=True):
        predicted = int(model.names[int(result.probs.top1)].removeprefix("level-"))
        total[expected] += 1
        correct[expected] += predicted == expected
        confusion[expected][predicted] += 1

    hits = sum(correct.values())
    count = sum(total.values())
    print(f"{args.building}: {hits}/{count} corrects ({hits / count:.1%}) sur split={args.split}")
    for level in sorted(total):
        details = ", ".join(f"niv.{guess}={n}" for guess, n in sorted(confusion[level].items()))
        print(f"  niveau {level}: {correct[level]}/{total[level]} corrects | {details}")


if __name__ == "__main__":
    main()
