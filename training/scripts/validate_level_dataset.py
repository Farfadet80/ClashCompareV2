"""Mesure la couverture des images de niveaux avant classification."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum", type=int, default=10, help="Minimum d'images par niveau")
    parser.add_argument("--min-train", type=int, default=50)
    parser.add_argument("--min-val", type=int, default=10)
    parser.add_argument("--min-test", type=int, default=10)
    args = parser.parse_args()

    catalog = json.loads((ROOT / "data" / "buildings.json").read_text(encoding="utf-8"))
    dataset = ROOT / "training" / "dataset" / "levels"
    manifest = ROOT / "training" / "dataset" / "level-splits.jsonl"
    split_paths: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            path = ROOT / row["path"]
            if path.exists():
                split_paths[(row["building"], int(row["level"]), row["split"])].add(row["path"])

    missing: list[tuple[str, int, int]] = []
    trainable: dict[str, list[int]] = defaultdict(list)
    independently_testable: dict[str, list[int]] = defaultdict(list)
    total_images = total_levels = ready_levels = 0
    for building in catalog["buildings"]:
        for level in building.get("levels", []):
            total_levels += 1
            folder = dataset / building["id"] / f"level-{level}"
            count = sum(1 for path in folder.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
            total_images += count
            if count >= args.minimum:
                ready_levels += 1
            else:
                missing.append((building["id"], level, count))
            train_count = len(split_paths[(building["id"], level, "train")])
            val_count = len(split_paths[(building["id"], level, "val")])
            test_count = len(split_paths[(building["id"], level, "test")])
            if train_count >= args.min_train and val_count >= args.min_val:
                trainable[building["id"]].append(level)
                if test_count >= args.min_test:
                    independently_testable[building["id"]].append(level)

    print(f"Images de niveaux: {total_images}")
    print(f"Niveaux prêts: {ready_levels}/{total_levels} (minimum {args.minimum} images par niveau)")
    trainable_levels = sum(len(levels) for levels in trainable.values())
    testable_levels = sum(len(levels) for levels in independently_testable.values())
    trainable_buildings = {building: levels for building, levels in trainable.items() if len(levels) >= 2}
    testable_buildings = {
        building: levels for building, levels in independently_testable.items() if len(levels) >= 2
    }
    print(
        f"Niveaux entraînables: {trainable_levels}/{total_levels} "
        f"(>= {args.min_train} train et >= {args.min_val} val)"
    )
    print(
        f"Niveaux avec test indépendant: {testable_levels}/{total_levels} "
        f"(+ >= {args.min_test} test)"
    )
    print(f"Bâtiments entraînables sur au moins 2 niveaux: {len(trainable_buildings)}")
    for building, levels in sorted(trainable_buildings.items()):
        print(f"- {building}: {levels}")
    print(f"Bâtiments évaluables sur test indépendant: {len(testable_buildings)}")
    for building, levels in sorted(testable_buildings.items()):
        print(f"- {building}: {levels}")
    if missing:
        print("Premiers niveaux incomplets:")
        for building, level, count in missing[:50]:
            print(f"- {building}/level-{level}: {count}/{args.minimum}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
