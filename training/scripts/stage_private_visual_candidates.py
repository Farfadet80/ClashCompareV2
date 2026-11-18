"""Staging collecte privée élargie → training/inbox/private/ (images hors git).

- Ne pousse rien : les binaires sont gitignorés.
- N'importe pas dans dataset/detector.
- N'invente aucun dataId.
- Peut copier des imports CC BY murs en file privée locale, ou des drops manuels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import urllib.request
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
PRIVATE = TRAINING / "inbox" / "private"
CLASSES = TRAINING / "classes.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

WALL_IMPORTS = {
    "clashia-detectwall-v7-curated": {
        "license": "CC BY 4.0",
        "url": "https://universe.roboflow.com/clashia/detectwall",
        "map_prefix": "wall",
    },
    "clashia-detectwall-v7": {
        "license": "CC BY 4.0",
        "url": "https://universe.roboflow.com/clashia/detectwall",
        "map_prefix": "wall",
    },
    "peters-walls-16-17-v2": {
        "license": "CC BY 4.0",
        "url": "https://universe.roboflow.com/peters-workspace-y70qq/coc-mauern",
        "map_all_to": "wall",
    },
    "kiet-walls-v6-curated": {
        "license": "CC BY 4.0",
        "url": "https://universe.roboflow.com/clash-of-clan/clash-of-clan-vzsst/dataset/6",
        "map_prefix": "wall",
    },
    "kiet-walls-v6": {
        "license": "CC BY 4.0",
        "url": "https://universe.roboflow.com/clash-of-clan/clash-of-clan-vzsst/dataset/6",
        "map_prefix": "wall",
    },
}

# P1/P2 TH10–18 — hors classes YOLO 50–53 / dataIds TH19
TH10_18_PRIORITY = [
    "wall",
    "tornado-trap",
    "giga-bomb",
    "workshop",
    "pet-house",
    "blacksmith",
    "hero-hall",
    "crafted-defense",
    "town-hall-guardian",
]
TH19_SKIP = {
    "spiky-tower",
    "firecracker-tower",
    "guardian-shrine",
    "squad-barracks",
}
PRIORITY_SOURCE_KEYS = [
    "coc-all-traps",
    "glenn-th18-gigabomb",
    "evans-home-village-building-detector",
    "base-finder-th18",
    "pedro-coc-model",
    "cocsegmentation-villages",
    "nothing-clash-of-clans",
]
REGISTRY = TRAINING / "sources" / "public-datasets.json"

TH19_REFERENCE = [
    {
        "url": "https://clashofclans.inbox.supercell.com/lqe2co20rkhw/6MB58yE8JHC3zXYk1r1fq6/ee9223ff980ca16b5da9e6999f9177ac/th19_900x506.jpg",
        "name": "supercell-clash-of-titans-th19_900x506.jpg",
        "license": "Supercell copyright — reference-only",
        "note": "OG blog Clash of Titans ; hors train YOLO.",
    }
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names_list(value: list | dict) -> list[str]:
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


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, width: int, height: int):
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


def class_ids() -> dict[str, int]:
    rows = json.loads(CLASSES.read_text(encoding="utf-8"))["detector_classes"]
    return {row["id"]: index for index, row in enumerate(rows)}


def write_pair(
    dest_dir: Path,
    image_path: Path,
    dest_name: str,
    boxes: list[dict],
    provenance: dict,
    metadata: dict,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_image = dest_dir / dest_name
    if not dest_image.exists():
        shutil.copy2(image_path, dest_image)
    stem = dest_image.stem
    (dest_dir / f"{stem}.provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with Image.open(dest_image) as image:
        width, height = image.size
    session = {
        "version": 1,
        "image": {"name": dest_image.name, "width": width, "height": height},
        "metadata": metadata,
        "expected_counts": None,
        "boxes": boxes,
    }
    (dest_dir / f"{stem}.session.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return dest_image


def map_wall_name(raw: str, cfg: dict) -> str | None:
    lowered = raw.lower().replace("_", " ").strip()
    if cfg.get("map_all_to") == "wall":
        if "wall" in lowered or "mauer" in lowered:
            return "wall"
        return None
    prefix = cfg.get("map_prefix", "wall")
    if lowered.startswith(prefix) or lowered.startswith("wall "):
        return "wall"
    return None


def stage_wall_imports(max_per_source: int) -> dict:
    ids = class_ids()
    staged: dict[str, list[str]] = defaultdict(list)
    hash_seen: set[str] = set()
    for source_key, cfg in WALL_IMPORTS.items():
        source_root = TRAINING / "imports" / source_key
        if not source_root.exists():
            continue
        source_yaml = next(iter([*source_root.glob("*.yaml"), *source_root.glob("*.yml")]), None)
        if source_yaml is None:
            continue
        names = names_list(yaml.safe_load(source_yaml.read_text(encoding="utf-8"))["names"])
        wanted = {
            index: map_wall_name(name, cfg)
            for index, name in enumerate(names)
            if map_wall_name(name, cfg)
        }
        if not wanted:
            continue
        count = 0
        for split in ("train", "val", "test"):
            dirs = split_dirs(source_root, split)
            if not dirs:
                continue
            image_dir, label_dir = dirs
            for image_path in sorted(image_dir.rglob("*")):
                if count >= max_per_source:
                    break
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                label_path = (label_dir / image_path.relative_to(image_dir)).with_suffix(".txt")
                if not label_path.exists():
                    continue
                digest = sha256(image_path)
                if digest in hash_seen:
                    continue
                with Image.open(image_path) as image:
                    width, height = image.size
                boxes: list[dict] = []
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        class_id = int(parts[0])
                        values = [float(v) for v in parts[1:5]]
                    except ValueError:
                        continue
                    dst = wanted.get(class_id)
                    if not dst or dst not in ids:
                        continue
                    x1, y1, x2, y2 = yolo_to_xyxy(*values, width, height)
                    boxes.append(
                        {
                            "class_id": ids[dst],
                            "class_name": dst,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "status": "pending",
                            "origin": "private-wall-import-yolo",
                        }
                    )
                if not boxes:
                    continue
                dest_name = f"{source_key}__{split}__{image_path.name}"
                provenance = {
                    "village_group": f"private-{source_key}-{family_key(image_path.stem)}",
                    "class_focus": "wall",
                    "license": cfg["license"],
                    "source_url": cfg["url"],
                    "source_key": source_key,
                    "source_split": split,
                    "sha256": digest,
                    "usage": "private_local_annotation_queue",
                    "publish": False,
                    "notes": [
                        "Copie locale privée pour élargir la file murs.",
                        "Ne pas republier / ne pas committer l'image.",
                        "Annotation exhaustive requise avant import detector.",
                    ],
                }
                metadata = {
                    "village_group": provenance["village_group"],
                    "source": source_key,
                    "license": cfg["license"],
                    "exhaustive": False,
                    "draft": True,
                    "class_focus": "wall",
                    "private": True,
                }
                dest = write_pair(
                    PRIVATE / "wall",
                    image_path,
                    dest_name,
                    boxes,
                    provenance,
                    metadata,
                )
                hash_seen.add(digest)
                staged["wall"].append(dest.relative_to(ROOT).as_posix())
                count += 1
    return staged


def stage_drop(class_focus: str) -> list[str]:
    ids = class_ids()
    if class_focus not in ids:
        raise SystemExit(f"Classe inconnue: {class_focus}")
    if class_focus in TH19_SKIP:
        raise SystemExit(f"TH19 différé — ne pas stager {class_focus}")
    drop = PRIVATE / "drop"
    drop.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    dest_dir = PRIVATE / class_focus
    for image_path in sorted(drop.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        digest = sha256(image_path)
        dest_name = f"drop__{image_path.name}"
        provenance = {
            "village_group": f"private-drop-{family_key(image_path.stem)}",
            "class_focus": class_focus,
            "license": "consentement local — dataset privé",
            "sha256": digest,
            "usage": "private_local_annotation_queue",
            "publish": False,
            "notes": [
                "Capture déposée manuellement dans inbox/private/drop/.",
                "Ne jamais pousser l'image sur le remote public.",
            ],
        }
        metadata = {
            "village_group": provenance["village_group"],
            "source": "private-drop",
            "license": provenance["license"],
            "exhaustive": False,
            "draft": True,
            "class_focus": class_focus,
            "private": True,
        }
        dest = write_pair(dest_dir, image_path, dest_name, [], provenance, metadata)
        out.append(dest.relative_to(ROOT).as_posix())
    return out


def stage_priority_imports(max_per_class: int) -> dict[str, list[str]]:
    """Élargit la file privée TH10–18 depuis imports CC BY locaux (images hors git)."""
    ids = class_ids()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    staged: dict[str, list[str]] = defaultdict(list)
    hash_seen: set[str] = set()
    family_seen: dict[str, set[str]] = defaultdict(set)

    for source_key in PRIORITY_SOURCE_KEYS:
        source_root = TRAINING / "imports" / source_key
        if not source_root.exists() or source_key not in registry:
            continue
        info = registry[source_key]
        mapping: dict[str, str] = info.get("mapping") or {}
        # ignore TH19 + ne garder que P1/P2 TH10–18
        wanted_src = {
            src: dst
            for src, dst in mapping.items()
            if dst in TH10_18_PRIORITY and dst not in TH19_SKIP
        }
        if not wanted_src:
            continue
        # Pedro Little_Guardian = UI troupes — skip guardian
        if source_key == "pedro-coc-model":
            wanted_src = {k: v for k, v in wanted_src.items() if v != "town-hall-guardian"}
            if not wanted_src:
                continue
        source_yaml = next(iter([*source_root.glob("*.yaml"), *source_root.glob("*.yml")]), None)
        if source_yaml is None:
            continue
        names = names_list(yaml.safe_load(source_yaml.read_text(encoding="utf-8"))["names"])
        wanted_ids = {
            index: wanted_src[name]
            for index, name in enumerate(names)
            if name in wanted_src
        }
        if not wanted_ids:
            continue

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
                digest = sha256(image_path)
                if digest in hash_seen:
                    continue
                with Image.open(image_path) as image:
                    width, height = image.size
                boxes: list[dict] = []
                focus_counts: Counter = Counter()
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        class_id = int(parts[0])
                        values = [float(v) for v in parts[1:5]]
                    except ValueError:
                        continue
                    dst = wanted_ids.get(class_id)
                    if not dst or dst not in ids:
                        continue
                    x1, y1, x2, y2 = yolo_to_xyxy(*values, width, height)
                    boxes.append(
                        {
                            "class_id": ids[dst],
                            "class_name": dst,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "status": "pending",
                            "origin": "private-priority-import-yolo",
                        }
                    )
                    focus_counts[dst] += 1
                if not boxes:
                    continue
                primary = next(
                    (
                        name
                        for name in TH10_18_PRIORITY
                        if name in focus_counts and len(staged[name]) < max_per_class
                    ),
                    None,
                )
                if primary is None:
                    continue
                family = family_key(image_path.stem)
                if family in family_seen[primary]:
                    continue
                dest_name = f"{source_key}__{split}__{image_path.name}"
                provenance = {
                    "village_group": f"private-{source_key}-{family}",
                    "class_focus": primary,
                    "license": info.get("license"),
                    "source_url": info.get("url"),
                    "source_key": source_key,
                    "source_split": split,
                    "sha256": digest,
                    "usage": "private_local_annotation_queue",
                    "publish": False,
                    "scope": "TH10-18",
                    "notes": [
                        "Copie locale privée TH10–18 (CC BY) — ne pas republier l'image.",
                        "Classes YOLO 50–53 / TH19 ignorées volontairement.",
                        "Annotation exhaustive requise avant import detector.",
                    ],
                }
                metadata = {
                    "village_group": provenance["village_group"],
                    "source": source_key,
                    "license": info.get("license"),
                    "exhaustive": False,
                    "draft": True,
                    "class_focus": primary,
                    "private": True,
                    "scope": "TH10-18",
                }
                dest = write_pair(
                    PRIVATE / primary,
                    image_path,
                    dest_name,
                    boxes,
                    provenance,
                    metadata,
                )
                hash_seen.add(digest)
                family_seen[primary].add(family)
                staged[primary].append(dest.relative_to(ROOT).as_posix())
    return staged


def fetch_th19_reference() -> list[str]:
    written: list[str] = []
    for item in TH19_REFERENCE:
        dest_dir = PRIVATE / "reference-only" / "th19-visual"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / item["name"]
        if not dest.exists():
            try:
                urllib.request.urlretrieve(item["url"], dest)
            except Exception as exc:  # noqa: BLE001 — consigne locale
                print(f"WARN fetch TH19 ref: {exc}")
                continue
        side = {
            "license": item["license"],
            "source_url": item["url"],
            "usage": "reference_only",
            "publish": False,
            "notes": [item["note"], "Hors dataset/detector."],
            "sha256": sha256(dest) if dest.exists() else None,
        }
        (dest.with_suffix(dest.suffix + ".provenance.json")).write_text(
            json.dumps(side, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(dest.relative_to(ROOT).as_posix())
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-drop", action="store_true")
    parser.add_argument("--class-focus", default="wall")
    parser.add_argument("--from-wall-imports", action="store_true")
    parser.add_argument(
        "--from-priority-imports",
        action="store_true",
        help="Stager P1/P2 TH10–18 depuis imports CC BY locaux (ignore classes 50–53)",
    )
    parser.add_argument("--max-per-source", type=int, default=40)
    parser.add_argument("--max-per-class", type=int, default=40)
    parser.add_argument(
        "--fetch-th19-reference",
        action="store_true",
        help="Réf visuelle TH19 hors train seulement (fin de roadmap)",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=TRAINING / "reports" / f"private-collection-{date.today().isoformat().replace('-', '')}.json",
    )
    args = parser.parse_args()

    PRIVATE.mkdir(parents=True, exist_ok=True)
    (PRIVATE / "drop").mkdir(exist_ok=True)
    (PRIVATE / "exports-json").mkdir(exist_ok=True)

    summary: dict = {
        "updated": date.today().isoformat(),
        "publish_images": False,
        "train_now": False,
        "invent_dataids": False,
        "scope": "TH10-18",
        "th19_deferred": True,
        "staged": {},
        "notes": [
            "Images privées gitignorées ; JSON de session/provenance peuvent rester locaux.",
            "Pas d'import detector / pas de fine-tune dans ce script.",
            "Priorité TH10–18 ; classes YOLO 50–53 / dataIds TH19 ignorées.",
        ],
    }

    if args.from_wall_imports:
        summary["staged"]["wall_imports"] = stage_wall_imports(args.max_per_source)
    if args.from_priority_imports:
        summary["staged"]["priority_imports"] = stage_priority_imports(args.max_per_class)
    if args.from_drop:
        summary["staged"]["drop"] = stage_drop(args.class_focus)
    if args.fetch_th19_reference:
        summary["staged"]["th19_reference"] = fetch_th19_reference()

    if not any(
        [
            args.from_wall_imports,
            args.from_priority_imports,
            args.from_drop,
            args.fetch_th19_reference,
        ]
    ):
        parser.error(
            "Choisir --from-wall-imports et/ou --from-priority-imports "
            "et/ou --from-drop et/ou --fetch-th19-reference"
        )

    class_counts = {}
    for name in TH10_18_PRIORITY:
        folder = PRIVATE / name
        class_counts[name] = (
            len(list(folder.glob("*.session.json"))) if folder.exists() else 0
        )
    summary["private_sessions_by_class"] = class_counts
    wall_sessions = list((PRIVATE / "wall").glob("*.session.json")) if (PRIVATE / "wall").exists() else []
    summary["private_wall_sessions"] = len(wall_sessions)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Rapport: {args.output_report}")


if __name__ == "__main__":
    main()
