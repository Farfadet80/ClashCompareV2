"""Stage des candidats CC BY prioritaires vers inbox/public-reference.

Copie images + provenance + brouillons de session annotateur (exhaustive=false,
boîtes status=pending). N'importe PAS dans dataset/detector. N'invente aucun dataId.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
MANIFEST = TRAINING / "sources" / "visual-collection-manifest.json"
REGISTRY = TRAINING / "sources" / "public-datasets.json"
CLASSES = TRAINING / "classes.json"
INBOX = TRAINING / "inbox" / "public-reference"
REFERENCE = TRAINING / "reference-only"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

SOURCE_KEYS = [
    "coc-all-traps",
    "glenn-th18-gigabomb",
    "evans-home-village-building-detector",
    "base-finder-th18",
    "pedro-coc-model",
    "cocsegmentation-villages",
    "nothing-clash-of-clans",
]


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


def family_key(stem: str) -> str:
    return re.sub(r"\.rf\.[a-f0-9]+$", "", stem, flags=re.IGNORECASE)


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


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = (cx - w / 2) * width
    y1 = (cy - h / 2) * height
    x2 = (cx + w / 2) * width
    y2 = (cy + h / 2) * height
    return (
        round(max(0.0, min(width, x1)), 2),
        round(max(0.0, min(height, y1)), 2),
        round(max(0.0, min(width, x2)), 2),
        round(max(0.0, min(height, y2)), 2),
    )


def load_targets(manifest: dict) -> set[str]:
    return {
        *manifest["target_classes"]["reserved_th19_yolo"],
        *manifest["target_classes"]["unlearnable_or_empty"],
        *manifest["target_classes"]["very_weak"],
        *manifest["target_classes"]["weak"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=40)
    parser.add_argument(
        "--output-report",
        type=Path,
        default=TRAINING / "reports" / "visual-collection-c1c2-20261118.json",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    detector_classes = json.loads(CLASSES.read_text(encoding="utf-8"))["detector_classes"]
    class_ids = {row["id"]: index for index, row in enumerate(detector_classes)}
    targets = load_targets(manifest)

    staged: dict[str, list[dict]] = defaultdict(list)
    box_totals: Counter[str] = Counter()
    family_seen: dict[str, set[str]] = defaultdict(set)
    hash_seen: set[str] = set()
    reference_seeds: list[dict] = []
    known_detector: set[str] = set()
    detector_images = TRAINING / "dataset" / "detector" / "images"
    if detector_images.exists():
        for split in ("train", "val", "test"):
            split_dir = detector_images / split
            if not split_dir.exists():
                continue
            for image_path in split_dir.rglob("*"):
                if image_path.suffix.lower() in IMAGE_SUFFIXES:
                    known_detector.add(sha256(image_path))

    for source_key in SOURCE_KEYS:
        source_root = TRAINING / "imports" / source_key
        if not source_root.exists() or source_key not in registry:
            continue
        info = registry[source_key]
        mapping: dict[str, str] = info.get("mapping") or {}
        wanted_src = {src for src, dst in mapping.items() if dst in targets}
        if not wanted_src:
            continue
        source_yaml = next(iter([*source_root.glob("*.yaml"), *source_root.glob("*.yml")]), None)
        if source_yaml is None:
            continue
        names = names_list(yaml.safe_load(source_yaml.read_text(encoding="utf-8"))["names"])
        wanted_ids = {index for index, name in enumerate(names) if name in wanted_src}

        for split in ("train", "val", "test"):
            dirs = split_dirs(source_root, split)
            if not dirs:
                continue
            image_dir, label_dir = dirs
            for image_path in sorted(image_dir.rglob("*")):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                label_path = (label_dir / image_path.relative_to(image_dir)).with_suffix(".txt")
                if not label_path.exists():
                    continue
                focus_boxes: list[tuple[str, list[float]]] = []
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        class_id = int(parts[0])
                        values = [float(value) for value in parts[1:5]]
                    except ValueError:
                        continue
                    if class_id not in wanted_ids:
                        continue
                    dst = mapping[names[class_id]]
                    focus_boxes.append((dst, values))
                if not focus_boxes:
                    continue

                focus_counts = Counter(name for name, _ in focus_boxes)
                # Une image peut servir plusieurs classes cibles ; on la rattache
                # à la classe prioritaire la plus rare encore sous quota.
                # TH10–18 d'abord ; classes YOLO 50–53 / TH19 en tout dernier
                priority_order = [
                    "wall",
                    "tornado-trap",
                    "giga-bomb",
                    "town-hall-guardian",
                    "workshop",
                    "pet-house",
                    "blacksmith",
                    "hero-hall",
                    "crafted-defense",
                    "spiky-tower",
                    "firecracker-tower",
                    "guardian-shrine",
                    "squad-barracks",
                ]
                primary = next(
                    (
                        name
                        for name in priority_order
                        if name in focus_counts and len(staged[name]) < args.max_per_class
                    ),
                    None,
                )
                if primary is None:
                    continue

                digest = sha256(image_path)
                family = family_key(image_path.stem)
                if digest in hash_seen:
                    continue
                if family in family_seen[primary] and primary != "town-hall-guardian":
                    # garder 1 variante .rf par famille pour la file d'annotation
                    continue
                if len(family_seen[primary]) >= args.max_per_class:
                    continue

                dest_dir = INBOX / primary
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_image = dest_dir / f"{source_key}__{split}__{image_path.name}"
                shutil.copy2(image_path, dest_image)
                with Image.open(dest_image) as image:
                    width, height = image.size

                boxes_payload = []
                for dst_name, (cx, cy, bw, bh) in focus_boxes:
                    if dst_name not in class_ids:
                        continue
                    x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, width, height)
                    boxes_payload.append(
                        {
                            "class_id": class_ids[dst_name],
                            "class_name": dst_name,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "status": "pending",
                            "origin": "public-source-yolo",
                        }
                    )
                    box_totals[dst_name] += 1

                provenance = {
                    "village_group": f"public-{source_key}-{family}",
                    "class_focus": primary,
                    "town_hall": None,
                    "capture_mode": "public-roboflow-yolo",
                    "license": info.get("license"),
                    "source_url": info.get("url"),
                    "author": info.get("author"),
                    "source_key": source_key,
                    "source_split": split,
                    "source_image": image_path.relative_to(ROOT).as_posix(),
                    "sha256": digest,
                    "usage": "train_candidate_after_audit",
                    "already_in_detector_dataset": digest in known_detector,
                    "levels_verified": False,
                    "consent": True,
                    "notes": [
                        "Brouillon issu des labels YOLO publics mappés.",
                        "Annotation exhaustive obligatoire avant import_annotation_session.",
                        "Ne pas inventer de dataIds JSON.",
                    ],
                }
                stem = dest_image.stem
                (dest_dir / f"{stem}.provenance.json").write_text(
                    json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                session = {
                    "version": 1,
                    "image": {
                        "name": dest_image.name,
                        "width": width,
                        "height": height,
                    },
                    "metadata": {
                        "village_group": provenance["village_group"],
                        "source": source_key,
                        "license": info.get("license"),
                        "exhaustive": False,
                        "draft": True,
                        "class_focus": primary,
                    },
                    "expected_counts": None,
                    "boxes": boxes_payload,
                }
                (dest_dir / f"{stem}.session.json").write_text(
                    json.dumps(session, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

                hash_seen.add(digest)
                family_seen[primary].add(family)
                staged[primary].append(
                    {
                        "image": dest_image.relative_to(ROOT).as_posix(),
                        "source_key": source_key,
                        "family": family,
                        "focus_boxes": int(focus_counts[primary]),
                        "all_target_boxes": len(boxes_payload),
                    }
                )

                if primary == "town-hall-guardian":
                    ref_dir = REFERENCE / "town-hall-guardian"
                    ref_dir.mkdir(parents=True, exist_ok=True)
                    ref_image = ref_dir / dest_image.name
                    if not ref_image.exists():
                        shutil.copy2(dest_image, ref_image)
                        (ref_dir / f"{ref_image.stem}.provenance.json").write_text(
                            json.dumps(
                                {
                                    **provenance,
                                    "usage": "manual_reference_only",
                                    "notes": [
                                        "Seed Guardian unique (Pedro CC BY). Insuffisant pour apprendre.",
                                        "Hors dataset/detector tant que < 20 boîtes / 8 villages.",
                                    ],
                                },
                                ensure_ascii=False,
                                indent=2,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                        reference_seeds.append(ref_image.relative_to(ROOT).as_posix())

    th19_watch = {
        "checked_on": date.today().isoformat(),
        "classes": list(manifest["target_classes"]["reserved_th19_yolo"]),
        "public_labeled_datasets_found": 0,
        "images_staged_for_train": 0,
        "notes": [
            "Aucun dataset Roboflow/HuggingFace/GitHub annoté Spiky/Firecracker/Shrine/Squad Barracks au 2026-11-18.",
            "Blog Supercell = reference-only (copyright) — non copié dans inbox train.",
            "Fandom Spiky Tower = HTTP 403 au fetch ; pages encore instables.",
        ],
        "reference_only_dirs": [
            str((REFERENCE / "th19-visual" / class_id).relative_to(ROOT)).replace("\\", "/")
            for class_id in manifest["target_classes"]["reserved_th19_yolo"]
        ],
    }

    summary = {
        "updated": date.today().isoformat(),
        "phase": "C1-C2",
        "train_now": False,
        "invent_dataids": False,
        "policy": manifest.get("policy", {}),
        "staged_images_by_class": {key: len(rows) for key, rows in sorted(staged.items())},
        "staged_images_total": sum(len(rows) for rows in staged.values()),
        "draft_target_boxes": dict(sorted(box_totals.items())),
        "draft_target_boxes_total": int(sum(box_totals.values())),
        "unique_families_by_class": {key: len(values) for key, values in sorted(family_seen.items())},
        "reference_seeds": reference_seeds,
        "th19_watch": th19_watch,
        "notes": [
            "Images déjà présentes dans detector (doublons SHA) : refile annotateur pour exhaustivité, pas nouvel import brut.",
            "Sessions .session.json = brouillons pending ; import_annotation_session refusera tant que exhaustive≠true.",
            "Aucun fine-tune lancé.",
        ],
        "details": {key: rows for key, rows in sorted(staged.items())},
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: summary[k] for k in (
        "updated",
        "staged_images_total",
        "staged_images_by_class",
        "draft_target_boxes",
        "reference_seeds",
        "th19_watch",
    )}, ensure_ascii=False, indent=2))
    print(f"Rapport: {args.output_report}")


if __name__ == "__main__":
    main()
