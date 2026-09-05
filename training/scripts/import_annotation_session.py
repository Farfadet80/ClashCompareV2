"""Valide puis importe une session humaine exhaustive dans le split train.

Le script refuse les pseudo-labels en attente, les métadonnées manquantes, les
quantités export incompatibles et toute image déjà présente dans un split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
DATASET = TRAINING / "dataset" / "detector"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
PUBLIC_LICENSES = {"CC BY 4.0", "CC BY-SA 4.0", "CC0 1.0", "Public Domain"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized[:60] or "village"


def existing_hash_locations() -> dict[str, list[str]]:
    locations: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        for image in (DATASET / "images" / split).rglob("*"):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            locations.setdefault(file_sha256(image), []).append(
                image.relative_to(ROOT).as_posix()
            )
    return locations


def validate_session(session: dict, image_path: Path) -> tuple[list[str], dict]:
    errors: list[str] = []
    classes_payload = json.loads((TRAINING / "classes.json").read_text(encoding="utf-8"))
    classes = classes_payload["detector_classes"]
    metadata = session.get("metadata") or {}
    image_meta = session.get("image") or {}

    if session.get("version") != 1:
        errors.append("version de session attendue: 1")
    if not metadata.get("village_group"):
        errors.append("metadata.village_group manquant")
    if not metadata.get("source"):
        errors.append("metadata.source manquant")
    license_name = str(metadata.get("license") or "").strip()
    if not (
        license_name in PUBLIC_LICENSES
        or license_name.lower().startswith("consentement local")
    ):
        errors.append(f"licence/consentement non autorisé: {license_name!r}")
    if metadata.get("exhaustive") is not True:
        errors.append("session non déclarée exhaustive")

    with Image.open(image_path) as image:
        width, height = image.size
    if image_meta.get("width") != width or image_meta.get("height") != height:
        errors.append(
            f"dimensions session {image_meta.get('width')}x{image_meta.get('height')} "
            f"≠ image {width}x{height}"
        )

    accepted: list[dict] = []
    counts: Counter[str] = Counter()
    pending_count = 0
    for index, box in enumerate(session.get("boxes") or []):
        if box.get("status") != "accepted":
            pending_count += 1
            continue
        class_id = box.get("class_id")
        if not isinstance(class_id, int) or not 0 <= class_id < len(classes):
            errors.append(f"boîte {index}: class_id invalide")
            continue
        expected_name = classes[class_id]["id"]
        if box.get("class_name") != expected_name:
            errors.append(f"boîte {index}: class_name ne correspond pas à class_id")
        coords = [box.get(key) for key in ("x1", "y1", "x2", "y2")]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in coords):
            errors.append(f"boîte {index}: coordonnées invalides")
            continue
        x1, y1, x2, y2 = (float(value) for value in coords)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            errors.append(f"boîte {index}: hors image ou inversée")
            continue
        if x2 - x1 < 2 or y2 - y1 < 2:
            errors.append(f"boîte {index}: taille inférieure à 2 px")
            continue
        accepted.append(
            {
                "class_id": class_id,
                "class_name": expected_name,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
        counts[expected_name] += 1

    if pending_count:
        errors.append(f"{pending_count} boîte(s) encore en attente")
    if not accepted:
        errors.append("aucune boîte acceptée")
    expected_counts = session.get("expected_counts") or {}
    known_names = {item["id"] for item in classes}
    mismatches = {
        name: {"expected": int(expected), "annotated": counts.get(name, 0)}
        for name, expected in expected_counts.items()
        if name in known_names and counts.get(name, 0) != int(expected)
    }
    if mismatches:
        errors.append(f"quantités incompatibles: {mismatches}")

    result = {
        "width": width,
        "height": height,
        "classes_version": classes_payload.get("version"),
        "accepted_boxes": accepted,
        "counts": dict(sorted(counts.items())),
    }
    return errors, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.image.is_file() or args.image.suffix.lower() not in IMAGE_SUFFIXES:
        raise SystemExit(f"Image invalide: {args.image}")
    session = json.loads(args.session.read_text(encoding="utf-8"))
    errors, validated = validate_session(session, args.image)
    digest = file_sha256(args.image)
    duplicates = existing_hash_locations().get(digest, [])
    if duplicates:
        errors.append(f"image déjà présente dans le dataset: {duplicates}")
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    metadata = session["metadata"]
    stem = f"human-{slug(metadata['village_group'])}-{digest[:12]}"
    target_image = DATASET / "images" / "train" / f"{stem}{args.image.suffix.lower()}"
    target_label = DATASET / "labels" / "train" / f"{stem}.txt"
    provenance = DATASET / "provenance" / f"{stem}.json"
    width, height = validated["width"], validated["height"]
    labels = []
    for box in validated["accepted_boxes"]:
        x = ((box["x1"] + box["x2"]) / 2) / width
        y = ((box["y1"] + box["y2"]) / 2) / height
        w = (box["x2"] - box["x1"]) / width
        h = (box["y2"] - box["y1"]) / height
        labels.append(
            f"{box['class_id']} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        )

    payload = {
        "ok": True,
        "dry_run": args.dry_run,
        "target_image": target_image.relative_to(ROOT).as_posix(),
        "target_label": target_label.relative_to(ROOT).as_posix(),
        "sha256": digest,
        "village_group": metadata["village_group"],
        "source": metadata["source"],
        "license": metadata["license"],
        "exhaustive": True,
        "counts": validated["counts"],
        "box_count": len(labels),
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "session": str(args.session.resolve()),
    }
    if not args.dry_run:
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        provenance.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.image, target_image)
        target_label.write_text("\n".join(labels) + "\n", encoding="utf-8")
        provenance.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
