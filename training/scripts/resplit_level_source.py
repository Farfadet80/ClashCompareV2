"""Crée des splits indépendants pour les classes de niveaux sans validation/test.

Le script conserve les splits publiés lorsqu'une classe possède déjà des images
de validation et de test. Pour une classe uniquement présente dans train, il
répartit les images de façon déterministe et copie le résultat dans un nouvel
export YOLO, sans modifier la source téléchargée.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def names_list(value: list[str] | dict[int | str, str]) -> list[str]:
    if isinstance(value, list):
        return value
    return [value[key] for key in sorted(value, key=lambda item: int(item))]


def stable_order(seed: str, path: Path) -> str:
    return hashlib.sha256(f"{seed}:{path.stem}".encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", default="clashcompare-level-split-v1")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Le dossier de sortie existe déjà et n'est pas vide: {args.output}")
    config_path = next(iter(args.source.glob("*.yaml")), None) or next(iter(args.source.glob("*.yml")), None)
    if config_path is None:
        raise SystemExit("data.yaml absent")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    names = names_list(config["names"])

    samples: list[dict[str, object]] = []
    by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    for source_split in ("train", "valid", "test"):
        image_dir = args.source / source_split / "images"
        label_dir = args.source / source_split / "labels"
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = (label_dir / image_path.relative_to(image_dir)).with_suffix(".txt")
            if not label_path.exists():
                continue
            class_names = {
                names[int(line.split()[0])]
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if len(line.split()) == 5 and 0 <= int(line.split()[0]) < len(names)
            }
            if len(class_names) != 1:
                raise SystemExit(f"Une image doit contenir exactement une classe de niveau: {image_path} -> {class_names}")
            sample = {
                "image": image_path,
                "label": label_path,
                "class_name": next(iter(class_names)),
                "source_split": source_split,
                "target_split": source_split,
            }
            samples.append(sample)
            by_class[sample["class_name"]].append(sample)

    decisions: dict[str, dict[str, int | bool]] = {}
    for class_name, class_samples in sorted(by_class.items()):
        source_splits = {str(sample["source_split"]) for sample in class_samples}
        needs_split = not {"valid", "test"}.issubset(source_splits)
        if needs_split:
            ordered = sorted(class_samples, key=lambda sample: stable_order(args.seed, sample["image"]))
            total = len(ordered)
            if total < 6:
                raise SystemExit(f"Pas assez d'images pour séparer {class_name}: {total}")
            test_count = max(1, round(total * args.test_ratio))
            val_count = max(1, round(total * args.val_ratio))
            for sample in ordered[:test_count]:
                sample["target_split"] = "test"
            for sample in ordered[test_count:test_count + val_count]:
                sample["target_split"] = "valid"
            for sample in ordered[test_count + val_count:]:
                sample["target_split"] = "train"
        counts = defaultdict(int)
        for sample in class_samples:
            counts[str(sample["target_split"])] += 1
        decisions[class_name] = {
            "resplit": needs_split,
            "train": counts["train"],
            "valid": counts["valid"],
            "test": counts["test"],
        }

    for sample in samples:
        target_split = str(sample["target_split"])
        image_path = sample["image"]
        label_path = sample["label"]
        image_output = args.output / target_split / "images" / image_path.name
        label_output = args.output / target_split / "labels" / label_path.name
        image_output.parent.mkdir(parents=True, exist_ok=True)
        label_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, image_output)
        shutil.copy2(label_path, label_output)

    output_config = dict(config)
    output_config["train"] = "../train/images"
    output_config["val"] = "../valid/images"
    output_config["test"] = "../test/images"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "data.yaml").write_text(
        yaml.safe_dump(output_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    report = {"source": str(args.source), "seed": args.seed, "classes": decisions}
    (args.output / "split-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
