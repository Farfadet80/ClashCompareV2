"""Valide la structure et les annotations du dataset détecteur YOLO."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml


TRAINING = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def validate_split(root: Path, split: str, class_count: int) -> tuple[int, int, Counter[int], list[str]]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    images = [p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    classes: Counter[int] = Counter()
    errors: list[str] = []
    boxes = 0

    for image in images:
        relative = image.relative_to(image_dir).with_suffix(".txt")
        label = label_dir / relative
        if not label.exists():
            errors.append(f"Annotation manquante: {image}")
            continue
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{label}:{line_number}: 5 valeurs attendues")
                continue
            try:
                class_id = int(parts[0])
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                errors.append(f"{label}:{line_number}: valeur non numérique")
                continue
            if not 0 <= class_id < class_count:
                errors.append(f"{label}:{line_number}: classe {class_id} inconnue")
            if not all(0.0 <= value <= 1.0 for value in coords) or coords[2] <= 0 or coords[3] <= 0:
                errors.append(f"{label}:{line_number}: boîte hors limites ou vide")
            classes[class_id] += 1
            boxes += 1

    image_stems = {p.relative_to(image_dir).with_suffix("") for p in images}
    for label in label_dir.rglob("*.txt"):
        if label.relative_to(label_dir).with_suffix("") not in image_stems:
            errors.append(f"Image manquante pour: {label}")
    return len(images), boxes, classes, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=Path, default=TRAINING / "dataset.yaml")
    parser.add_argument("--min-train", type=int, default=100)
    parser.add_argument("--min-val", type=int, default=20)
    parser.add_argument("--min-test", type=int, default=0)
    parser.add_argument("--require-test", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.yaml.read_text(encoding="utf-8"))
    names = config["names"]
    class_count = len(names)
    root = (args.yaml.parent / config["path"]).resolve()
    all_errors: list[str] = []

    splits = [("train", args.min_train), ("val", args.min_val)]
    if "test" in config or args.require_test:
        splits.append(("test", max(args.min_test, 1 if args.require_test else 0)))

    for split, minimum in splits:
        images, boxes, classes, errors = validate_split(root, split, class_count)
        print(f"{split}: {images} images, {boxes} objets, {len(classes)} classes représentées")
        all_errors.extend(errors)
        if images < minimum:
            all_errors.append(f"{split}: {images} images seulement (minimum de sécurité: {minimum})")

    if all_errors:
        print("\nDataset non prêt:")
        for error in all_errors[:100]:
            print(f"- {error}")
        raise SystemExit(1)
    print("Dataset valide pour lancer un premier entraînement expérimental.")


if __name__ == "__main__":
    main()
