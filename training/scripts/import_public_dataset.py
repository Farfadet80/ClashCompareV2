"""Fusionne un export YOLO public dans les 50 classes ClashCompare.

Le script accepte les boîtes YOLO et les polygones YOLO segmentation. Les classes
absentes du mapping sont ignorées. Il préfixe les noms pour éviter les collisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names_list(value: list[str] | dict[int | str, str]) -> list[str]:
    if isinstance(value, list):
        return value
    return [value[key] for key in sorted(value, key=lambda item: int(item))]


def to_box(parts: list[str]) -> list[float] | None:
    values = [float(value) for value in parts[1:]]
    if len(values) == 4:
        return values
    if len(values) >= 6 and len(values) % 2 == 0:
        xs, ys = values[0::2], values[1::2]
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        return [(left + right) / 2, (top + bottom) / 2, right - left, bottom - top]
    return None


def find_yaml(source: Path) -> Path:
    candidates = list(source.glob("*.yaml")) + list(source.glob("*.yml"))
    if not candidates:
        raise SystemExit(f"Aucun data.yaml trouvé dans {source}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Dossier décompressé exporté par Roboflow")
    parser.add_argument("source_key", help="Clé dans training/sources/public-datasets.json")
    parser.add_argument("--dry-run", action="store_true", help="Compte sans écrire dans le dataset")
    parser.add_argument("--skip-empty", action="store_true", help="Ignore les images sans classe mappée")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Importe uniquement le split train afin de conserver les jeux val/test fixes",
    )
    parser.add_argument(
        "--as-train",
        action="store_true",
        help="Importe tous les splits source dans train ClashCompare (val/test locaux inchangés)",
    )
    args = parser.parse_args()

    registry_path = TRAINING / "sources" / "public-datasets.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if args.source_key not in registry:
        raise SystemExit(f"Source inconnue: {args.source_key}")
    source_info = registry[args.source_key]
    mapping: dict[str, str] = source_info["mapping"]

    source_yaml = find_yaml(args.source)
    source_config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    source_names = names_list(source_config["names"])
    target_config = yaml.safe_load((TRAINING / "dataset.yaml").read_text(encoding="utf-8"))
    target_names = names_list(target_config["names"])
    target_ids = {name: index for index, name in enumerate(target_names)}
    target_root = TRAINING / "dataset" / "detector"

    # Val/test sont des splits protégés. Même avec --as-train, une image déjà
    # réservée ne doit jamais être recopiée dans train.
    protected_stems: set[str] = set()
    protected_hashes: set[str] = set()
    for protected_split in ("val", "test"):
        protected_dir = target_root / "images" / protected_split
        if not protected_dir.exists():
            continue
        for protected_image in protected_dir.rglob("*"):
            if protected_image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            protected_stems.add(protected_image.stem)
            protected_hashes.add(file_sha256(protected_image))

    imported_images = imported_boxes = protected_skipped = 0
    # Preserve the source test split. It must never be used for validation or
    # hyper-parameter/model selection in future training runs.
    if args.as_train:
        split_aliases = (
            ("train", "train"),
            ("valid", "train"),
            ("val", "train"),
            ("test", "train"),
        )
    elif args.train_only:
        split_aliases = (("train", "train"),)
    else:
        split_aliases = (
            ("train", "train"),
            ("valid", "val"),
            ("val", "val"),
            ("test", "test"),
        )
    for source_split, target_split in split_aliases:
        image_dir = args.source / source_split / "images"
        label_dir = args.source / source_split / "labels"
        if not image_dir.exists():
            image_dir = args.source / "images" / source_split
            label_dir = args.source / "labels" / source_split
        if not image_dir.exists():
            continue

        for image in image_dir.rglob("*"):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = image.relative_to(image_dir)
            label = (label_dir / relative).with_suffix(".txt")
            converted: list[str] = []
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

            if args.skip_empty and not converted:
                continue

            digest = hashlib.sha1(str(relative).encode("utf-8")).hexdigest()[:10]
            stem = f"{args.source_key}-{digest}-{image.stem}"
            target_image = target_root / "images" / target_split / f"{stem}{image.suffix.lower()}"
            target_label = target_root / "labels" / target_split / f"{stem}.txt"
            if target_split == "train" and (
                stem in protected_stems or file_sha256(image) in protected_hashes
            ):
                protected_skipped += 1
                continue
            if target_image.exists() or target_label.exists():
                continue
            if not args.dry_run:
                target_image.parent.mkdir(parents=True, exist_ok=True)
                target_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, target_image)
                target_label.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
            imported_images += 1
            imported_boxes += len(converted)

    if not args.dry_run:
        attribution = TRAINING / "dataset" / "SOURCES.md"
        entry = f"- {source_info['author']} — {source_info['url']} — {source_info['license']}\n"
        existing = attribution.read_text(encoding="utf-8") if attribution.exists() else "# Sources des données\n\n"
        if entry not in existing:
            attribution.write_text(existing + entry, encoding="utf-8")
    mode = "Simulation" if args.dry_run else "Import"
    print(f"{mode} terminé: {imported_images} images, {imported_boxes} boîtes conservées")
    if protected_skipped:
        print(
            f"Splits protégés: {protected_skipped} image(s) val/test "
            "non recopiée(s) dans train"
        )


if __name__ == "__main__":
    main()
