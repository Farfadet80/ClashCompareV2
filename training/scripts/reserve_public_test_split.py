"""Restaure les splits test publics fusionnés par les anciens imports.

Le mode par défaut est un aperçu. Utiliser --apply pour déplacer les copies déjà
importées de val vers test. Les originaux restent dans training/imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

from import_public_dataset import IMAGE_SUFFIXES, find_yaml, names_list, to_box

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"


def split_dirs(source: Path) -> tuple[Path, Path] | None:
    image_dir = source / "test" / "images"
    label_dir = source / "test" / "labels"
    if not image_dir.exists():
        image_dir = source / "images" / "test"
        label_dir = source / "labels" / "test"
    return (image_dir, label_dir) if image_dir.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    registry = json.loads((TRAINING / "sources" / "public-datasets.json").read_text(encoding="utf-8"))
    target_config = yaml.safe_load((TRAINING / "dataset.yaml").read_text(encoding="utf-8"))
    target_names = names_list(target_config["names"])
    target_ids = {name: index for index, name in enumerate(target_names)}
    target_root = TRAINING / "dataset" / "detector"
    records: list[dict] = []

    for source_key, source_info in registry.items():
        source = TRAINING / "imports" / source_key
        if not source.exists():
            continue
        dirs = split_dirs(source)
        if not dirs:
            continue
        image_dir, label_dir = dirs
        source_config = yaml.safe_load(find_yaml(source).read_text(encoding="utf-8"))
        source_names = names_list(source_config["names"])
        mapping: dict[str, str] = source_info["mapping"]

        for image in image_dir.rglob("*"):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = image.relative_to(image_dir)
            digest = hashlib.sha1(str(relative).encode("utf-8")).hexdigest()[:10]
            stem = f"{source_key}-{digest}-{image.stem}"
            old_image = target_root / "images" / "val" / f"{stem}{image.suffix.lower()}"
            old_label = target_root / "labels" / "val" / f"{stem}.txt"
            new_image = target_root / "images" / "test" / old_image.name
            new_label = target_root / "labels" / "test" / old_label.name

            converted: list[str] = []
            label = (label_dir / relative).with_suffix(".txt")
            if label.exists():
                for line in label.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    source_id = int(parts[0])
                    if not 0 <= source_id < len(source_names):
                        continue
                    target_name = mapping.get(source_names[source_id])
                    if target_name not in target_ids:
                        continue
                    box = to_box(parts)
                    if box and all(0 <= value <= 1 for value in box) and box[2] > 0 and box[3] > 0:
                        converted.append(f"{target_ids[target_name]} " + " ".join(f"{value:.6f}" for value in box))

            record = {"source": source_key, "image": new_image.name, "boxes": len(converted), "was_in_val": old_image.exists()}
            records.append(record)
            if not args.apply:
                continue
            new_image.parent.mkdir(parents=True, exist_ok=True)
            new_label.parent.mkdir(parents=True, exist_ok=True)
            if old_image.exists():
                shutil.move(str(old_image), str(new_image))
            elif not new_image.exists():
                shutil.copy2(image, new_image)
            if old_label.exists():
                shutil.move(str(old_label), str(new_label))
            else:
                new_label.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")

    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "images": len(records),
        "previously_in_val": sum(record["was_in_val"] for record in records),
        "boxes": sum(record["boxes"] for record in records),
        "note": "Ce split devient réservé pour V4+. Il n'est pas indépendant de V3, qui l'a déjà utilisé en validation.",
        "records": records,
    }
    if args.apply:
        manifest = target_root / "test-split-manifest.json"
        manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Manifest: {manifest}")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
