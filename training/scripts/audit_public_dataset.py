"""Audit en lecture seule d'un export YOLO public avant tout import.

L'audit vérifie la licence déclarée, les labels, les classes mappées et les
doublons exacts. Il ne peut pas prouver qu'une image est annotée exhaustivement :
ce point reste une validation humaine obligatoire.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_LICENSES = {"CC BY 4.0", "CC BY-SA 4.0", "CC0 1.0", "Public Domain"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names_list(value: list[str] | dict[int | str, str]) -> list[str]:
    if isinstance(value, list):
        return value
    return [value[key] for key in sorted(value, key=lambda item: int(item))]


def find_yaml(source: Path) -> Path:
    candidates = sorted((*source.glob("*.yaml"), *source.glob("*.yml")))
    if not candidates:
        raise SystemExit(f"Aucun data.yaml trouvé dans {source}")
    return candidates[0]


def split_dirs(source: Path, split: str) -> tuple[Path, Path] | None:
    aliases = (split, "valid" if split == "val" else split)
    for alias in aliases:
        image_dir = source / alias / "images"
        label_dir = source / alias / "labels"
        if image_dir.exists():
            return image_dir, label_dir
        image_dir = source / "images" / alias
        label_dir = source / "labels" / alias
        if image_dir.exists():
            return image_dir, label_dir
    return None


def existing_hashes() -> set[str]:
    root = TRAINING / "dataset" / "detector" / "images"
    return {
        sha256(path)
        for split in ("train", "val", "test")
        for path in (root / split).rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("source_key")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    registry = json.loads(
        (TRAINING / "sources" / "public-datasets.json").read_text(encoding="utf-8")
    )
    if args.source_key not in registry:
        raise SystemExit(f"Source absente du registre: {args.source_key}")
    source_info = registry[args.source_key]
    mapping = source_info.get("mapping") or {}
    source_yaml = find_yaml(args.source)
    names = names_list(yaml.safe_load(source_yaml.read_text(encoding="utf-8"))["names"])
    known_hashes = existing_hashes()
    seen_hashes: set[str] = set()
    class_boxes: Counter[str] = Counter()
    split_rows = {}
    total_missing_labels = total_invalid_lines = total_unmapped_boxes = 0
    total_duplicates_existing = total_duplicates_internal = 0

    for split in ("train", "val", "test"):
        dirs = split_dirs(args.source, split)
        if not dirs:
            continue
        image_dir, label_dir = dirs
        row = Counter()
        for image in sorted(image_dir.rglob("*")):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            row["images"] += 1
            digest = sha256(image)
            if digest in known_hashes:
                row["duplicates_existing"] += 1
                total_duplicates_existing += 1
            if digest in seen_hashes:
                row["duplicates_internal"] += 1
                total_duplicates_internal += 1
            seen_hashes.add(digest)
            label = (label_dir / image.relative_to(image_dir)).with_suffix(".txt")
            if not label.exists():
                row["missing_labels"] += 1
                total_missing_labels += 1
                continue
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    row["invalid_lines"] += 1
                    total_invalid_lines += 1
                    continue
                try:
                    class_id = int(parts[0])
                    values = [float(value) for value in parts[1:]]
                except ValueError:
                    row["invalid_lines"] += 1
                    total_invalid_lines += 1
                    continue
                if not 0 <= class_id < len(names) or not values:
                    row["invalid_lines"] += 1
                    total_invalid_lines += 1
                    continue
                source_name = names[class_id]
                row["boxes"] += 1
                class_boxes[source_name] += 1
                if source_name not in mapping:
                    row["unmapped_boxes"] += 1
                    total_unmapped_boxes += 1
        split_rows[split] = dict(row)

    blockers = []
    license_name = source_info.get("license")
    if license_name not in ALLOWED_LICENSES:
        blockers.append(f"licence non approuvée: {license_name!r}")
    if source_info.get("task", "object-detection") != "object-detection":
        blockers.append(f"tâche non compatible: {source_info.get('task')}")
    if source_info.get("import_blocked_reason"):
        blockers.append(source_info["import_blocked_reason"])
    if total_missing_labels:
        blockers.append(f"{total_missing_labels} image(s) sans label")
    if total_invalid_lines:
        blockers.append(f"{total_invalid_lines} ligne(s) de label invalide(s)")
    if total_unmapped_boxes:
        blockers.append(f"{total_unmapped_boxes} boîte(s) de classes non mappées")

    payload = {
        "source_key": args.source_key,
        "source": str(args.source.resolve()),
        "url": source_info.get("url"),
        "author": source_info.get("author"),
        "license": license_name,
        "task": source_info.get("task", "object-detection"),
        "splits": split_rows,
        "class_boxes": dict(sorted(class_boxes.items())),
        "duplicates": {
            "existing_dataset": total_duplicates_existing,
            "inside_source": total_duplicates_internal,
        },
        "blockers": blockers,
        "status": "blocked" if blockers else "manual_exhaustiveness_review_required",
        "human_review_required": (
            "Vérifier visuellement que toutes les classes cibles visibles sont annotées. "
            "Un audit structurel ne peut pas détecter les faux négatifs."
        ),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    else:
        print(text)
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
