"""Audite les fuites entre splits YOLO et génère un train propre sans supprimer de données."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    split: str
    path: Path
    relative_stem: str
    sha256: str
    village_group: str | None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(config_path: Path) -> tuple[dict, Path]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_root = (config_path.parent / config["path"]).resolve()
    return config, dataset_root


def collect_records(config: dict, dataset_root: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for split in ("train", "val", "test"):
        split_value = config.get(split)
        if not split_value:
            continue
        image_dir = dataset_root / split_value
        if not image_dir.exists():
            raise FileNotFoundError(f"Split absent: {image_dir}")
        for path in sorted(image_dir.rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            provenance_path = dataset_root / "provenance" / f"{path.stem}.json"
            village_group = None
            if provenance_path.exists():
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                value = provenance.get("village_group")
                village_group = str(value) if value else None
            records.append(
                ImageRecord(
                    split=split,
                    path=path.resolve(),
                    relative_stem=path.relative_to(image_dir).with_suffix("").as_posix(),
                    sha256=file_sha256(path),
                    village_group=village_group,
                )
            )
    return records


def cross_split_groups(records: list[ImageRecord], key: str) -> list[list[ImageRecord]]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        value = getattr(record, key)
        if value is None:
            continue
        grouped[value].append(record)
    return [
        group
        for group in grouped.values()
        if len({record.split for record in group}) > 1
    ]


def overlap_pairs(groups: list[list[ImageRecord]]) -> dict[str, int]:
    pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for group in groups:
        for left in group:
            for right in group:
                if left.split >= right.split:
                    continue
                pair = f"{left.split}-{right.split}"
                pairs[pair].add((str(left.path), str(right.path)))
    return {pair: len(matches) for pair, matches in sorted(pairs.items())}


def relative_to_root(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def serialize_group(group: list[ImageRecord]) -> dict:
    return {
        "splits": sorted({record.split for record in group}),
        "files": [
            {
                "split": record.split,
                "path": relative_to_root(record.path),
                "sha256": record.sha256,
            }
            for record in sorted(group, key=lambda item: (item.split, str(item.path)))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=Path, default=TRAINING / "dataset.yaml")
    parser.add_argument(
        "--report",
        type=Path,
        default=TRAINING / "reports" / "split-integrity.json",
    )
    parser.add_argument(
        "--clean-train-list",
        type=Path,
        default=TRAINING / "dataset" / "detector" / "train-clean.txt",
        help="Liste Ultralytics générée ; aucune image source n'est supprimée.",
    )
    parser.add_argument(
        "--fail-on-overlap",
        action="store_true",
        help="Retourne un code d'erreur si train chevauche val/test.",
    )
    args = parser.parse_args()

    config_path = args.yaml.resolve()
    config, dataset_root = load_dataset(config_path)
    records = collect_records(config, dataset_root)
    stem_groups = cross_split_groups(records, "relative_stem")
    hash_groups = cross_split_groups(records, "sha256")
    village_groups = cross_split_groups(records, "village_group")

    protected_hashes = {
        record.sha256 for record in records if record.split in {"val", "test"}
    }
    protected_stems = {
        record.relative_stem for record in records if record.split in {"val", "test"}
    }
    protected_village_groups = {
        record.village_group
        for record in records
        if record.split in {"val", "test"} and record.village_group
    }
    train_records = [record for record in records if record.split == "train"]
    excluded = [
        record
        for record in train_records
        if (
            record.sha256 in protected_hashes
            or record.relative_stem in protected_stems
            or record.village_group in protected_village_groups
        )
    ]
    excluded_paths = {record.path for record in excluded}
    clean_train = [record for record in train_records if record.path not in excluded_paths]

    args.clean_train_list.parent.mkdir(parents=True, exist_ok=True)
    args.clean_train_list.write_text(
        "".join(f"{record.path.as_posix()}\n" for record in clean_train),
        encoding="utf-8",
    )

    split_counts: dict[str, int] = defaultdict(int)
    for record in records:
        split_counts[record.split] += 1
    report = {
        "dataset_yaml": relative_to_root(config_path),
        "dataset_root": relative_to_root(dataset_root),
        "split_image_counts": dict(sorted(split_counts.items())),
        "cross_split_same_stem_pairs": overlap_pairs(stem_groups),
        "cross_split_same_hash_pairs": overlap_pairs(hash_groups),
        "cross_split_same_village_group_pairs": overlap_pairs(village_groups),
        "train_images_before": len(train_records),
        "train_images_excluded": len(excluded),
        "train_images_clean": len(clean_train),
        "excluded_train_files": [relative_to_root(record.path) for record in excluded],
        "same_stem_groups": [serialize_group(group) for group in stem_groups],
        "same_hash_groups": [serialize_group(group) for group in hash_groups],
        "same_village_group_groups": [
            {
                "village_group": group[0].village_group,
                **serialize_group(group),
            }
            for group in village_groups
        ],
        "clean_train_list": relative_to_root(args.clean_train_list.resolve()),
        "policy": (
            "Exclude a train image when its relative stem, exact SHA-256 or declared "
            "village_group occurs in val/test. Source files are never moved or deleted."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Splits: "
        + ", ".join(f"{name}={count}" for name, count in sorted(split_counts.items()))
    )
    print(f"Chevauchements même stem: {overlap_pairs(stem_groups)}")
    print(f"Chevauchements SHA-256 exact: {overlap_pairs(hash_groups)}")
    print(f"Chevauchements village_group: {overlap_pairs(village_groups)}")
    print(
        f"Train propre: {len(clean_train)}/{len(train_records)} images "
        f"({len(excluded)} exclues, aucune supprimée)"
    )
    print(f"Liste: {args.clean_train_list}")
    print(f"Rapport: {args.report}")

    if args.fail_on_overlap and excluded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
