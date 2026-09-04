"""Mesure la couverture des images de niveaux avant classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum", type=int, default=10, help="Minimum d'images par niveau")
    args = parser.parse_args()

    catalog = json.loads((ROOT / "data" / "buildings.json").read_text(encoding="utf-8"))
    dataset = ROOT / "training" / "dataset" / "levels"
    missing: list[tuple[str, int, int]] = []
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

    print(f"Images de niveaux: {total_images}")
    print(f"Niveaux prêts: {ready_levels}/{total_levels} (minimum {args.minimum} images par niveau)")
    if missing:
        print("Premiers niveaux incomplets:")
        for building, level, count in missing[:50]:
            print(f"- {building}/level-{level}: {count}/{args.minimum}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

