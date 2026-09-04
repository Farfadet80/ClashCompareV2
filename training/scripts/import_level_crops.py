"""Extrait des crops bâtiment/niveau depuis un export YOLO spécialisé."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def names_list(value: list[str] | dict[int | str, str]) -> list[str]:
    if isinstance(value, list):
        return value
    return [value[key] for key in sorted(value, key=lambda item: int(item))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("source_key")
    parser.add_argument("--padding", type=float, default=0.08)
    args = parser.parse_args()

    registry = json.loads((TRAINING / "sources" / "level-datasets.json").read_text(encoding="utf-8"))
    if args.source_key not in registry:
        raise SystemExit(f"Source inconnue: {args.source_key}")
    info = registry[args.source_key]
    mapping = info["mapping"]

    yaml_files = list(args.source.glob("*.yaml")) + list(args.source.glob("*.yml"))
    if not yaml_files:
        raise SystemExit("data.yaml absent")
    config = yaml.safe_load(yaml_files[0].read_text(encoding="utf-8"))
    names = names_list(config["names"])
    manifest = TRAINING / "dataset" / "level-splits.jsonl"
    existing = set(manifest.read_text(encoding="utf-8").splitlines()) if manifest.exists() else set()
    added: list[str] = []
    crop_count = 0

    for source_split, target_split in (("train", "train"), ("valid", "val"), ("val", "val"), ("test", "test")):
        image_dir = args.source / source_split / "images"
        label_dir = args.source / source_split / "labels"
        if not image_dir.exists():
            image_dir = args.source / "images" / source_split
            label_dir = args.source / "labels" / source_split
        if not image_dir.exists():
            continue

        for image_path in image_dir.rglob("*"):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = (label_dir / image_path.relative_to(image_dir)).with_suffix(".txt")
            if not label_path.exists():
                continue
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                width, height = image.size
                for object_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    class_id = int(parts[0])
                    if not 0 <= class_id < len(names) or names[class_id] not in mapping:
                        continue
                    building, level = mapping[names[class_id]]
                    x, y, box_width, box_height = map(float, parts[1:])
                    pad_x, pad_y = box_width * args.padding, box_height * args.padding
                    left = max(0, int((x - box_width / 2 - pad_x) * width))
                    top = max(0, int((y - box_height / 2 - pad_y) * height))
                    right = min(width, int((x + box_width / 2 + pad_x) * width))
                    bottom = min(height, int((y + box_height / 2 + pad_y) * height))
                    # Tiny boxes occasionally occur in public annotations and create
                    # unreadable/corrupt classifier samples (for example 1x2 px).
                    # They carry no useful visual information for a building level.
                    if right - left < 10 or bottom - top < 10:
                        continue
                    digest = hashlib.sha1(str(image_path.relative_to(image_dir)).encode("utf-8")).hexdigest()[:12]
                    filename = f"{args.source_key}-{target_split}-{digest}-{object_index}.jpg"
                    output = TRAINING / "dataset" / "levels" / building / f"level-{level}" / filename
                    output.parent.mkdir(parents=True, exist_ok=True)
                    image.crop((left, top, right, bottom)).save(output, quality=95)
                    record = json.dumps({
                        "path": str(output.relative_to(ROOT)).replace("\\", "/"),
                        "building": building,
                        "level": level,
                        "split": target_split,
                        "source": args.source_key,
                    }, ensure_ascii=False, sort_keys=True)
                    if record not in existing:
                        existing.add(record)
                        added.append(record)
                    crop_count += 1

    if added:
        with manifest.open("a", encoding="utf-8") as handle:
            for record in added:
                handle.write(record + "\n")
    attribution = TRAINING / "dataset" / "SOURCES.md"
    entry = f"- {info['author']} — {info['url']} — {info['license']} (niveaux)\n"
    current = attribution.read_text(encoding="utf-8") if attribution.exists() else "# Sources des données\n\n"
    if entry not in current:
        attribution.write_text(current + entry, encoding="utf-8")
    print(f"Extraction terminée: {crop_count} crops, {len(added)} entrées ajoutées au manifeste")


if __name__ == "__main__":
    main()
