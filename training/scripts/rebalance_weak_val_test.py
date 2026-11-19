"""Rééquilibre val/test pour classes faibles (wall/giga/tornado/crafted).

Déplace des images train → val/test en respectant village_group disjoint
(aucune fuite cross-split). Préfère les villages Mode photo legacy multi-classe
(non focus-crop) quand disponibles. Pas de fine-tune.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "training" / "dataset" / "detector"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def load_classes() -> dict[str, int]:
    payload = json.loads((ROOT / "training" / "classes.json").read_text(encoding="utf-8"))
    return {item["id"]: index for index, item in enumerate(payload["detector_classes"])}


def image_for_label(split: str, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        path = DATASET / "images" / split / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def provenance_for(stem: str) -> dict:
    path = DATASET / "provenance" / f"{stem}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def village_groups_by_split() -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "val", "test"):
        for label in (DATASET / "labels" / split).rglob("*.txt"):
            prov = provenance_for(label.stem)
            vg = prov.get("village_group") or label.stem
            mapping[split].add(str(vg))
    return mapping


def candidates_for(class_name: str, class_id: int) -> list[dict]:
    rows: list[dict] = []
    for label in (DATASET / "labels" / "train").rglob("*.txt"):
        lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        hits = [line for line in lines if int(line.split()[0]) == class_id]
        if not hits:
            continue
        image = image_for_label("train", label.stem)
        if image is None:
            continue
        prov = provenance_for(label.stem)
        vg = str(prov.get("village_group") or label.stem)
        src = str(prov.get("source") or "legacy")
        n_classes = len({int(line.split()[0]) for line in lines})
        focus_crop = "focus-crop" in vg or "focus-crop" in src
        rows.append(
            {
                "stem": label.stem,
                "label": label,
                "image": image,
                "village_group": vg,
                "source": src,
                "boxes": len(hits),
                "n_classes": n_classes,
                "focus_crop": focus_crop,
                "score": (0 if focus_crop else 100) + n_classes * 2 + len(hits),
            }
        )
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


def move_one(row: dict, dest_split: str) -> dict:
    stem = row["stem"]
    src_img: Path = row["image"]
    src_lab: Path = row["label"]
    dest_img = DATASET / "images" / dest_split / src_img.name
    dest_lab = DATASET / "labels" / dest_split / f"{stem}.txt"
    dest_img.parent.mkdir(parents=True, exist_ok=True)
    dest_lab.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_img), str(dest_img))
    shutil.move(str(src_lab), str(dest_lab))
    prov_path = DATASET / "provenance" / f"{stem}.json"
    if prov_path.is_file():
        payload = json.loads(prov_path.read_text(encoding="utf-8"))
        payload["split"] = dest_split
        payload["target_image"] = dest_img.relative_to(ROOT).as_posix()
        payload["target_label"] = dest_lab.relative_to(ROOT).as_posix()
        payload["rebalanced_on"] = date.today().isoformat()
        payload["rebalance_reason"] = "p0-weak-val-test"
        prov_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "stem": stem,
        "dest_split": dest_split,
        "village_group": row["village_group"],
        "source": row["source"],
        "n_classes": row["n_classes"],
        "boxes": row["boxes"],
        "focus_crop": row["focus_crop"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wall-val", type=int, default=0)
    parser.add_argument("--wall-test", type=int, default=0)
    parser.add_argument("--giga-val", type=int, default=0)
    parser.add_argument("--giga-test", type=int, default=0)
    parser.add_argument("--tornado-val", type=int, default=0)
    parser.add_argument("--tornado-test", type=int, default=0)
    parser.add_argument("--crafted-val", type=int, default=0)
    parser.add_argument("--crafted-test", type=int, default=0)
    parser.add_argument("--min-classes", type=int, default=10)
    parser.add_argument("--prefer-mode-photo", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--report-name",
        default="",
        help="Nom fichier rapport (défaut auto daté)",
    )
    args = parser.parse_args()

    class_ids = load_classes()
    occupied = village_groups_by_split()
    blocked = occupied["val"] | occupied["test"]
    moved: list[dict] = []
    used_groups: set[str] = set()

    plan = [
        ("wall", "val", args.wall_val),
        ("wall", "test", args.wall_test),
        ("giga-bomb", "val", args.giga_val),
        ("giga-bomb", "test", args.giga_test),
        ("tornado-trap", "val", args.tornado_val),
        ("tornado-trap", "test", args.tornado_test),
        ("crafted-defense", "val", args.crafted_val),
        ("crafted-defense", "test", args.crafted_test),
    ]
    for class_name, dest_split, need in plan:
        if need <= 0:
            continue
        got = 0
        for row in candidates_for(class_name, class_ids[class_name]):
            if got >= need:
                break
            if row["n_classes"] < args.min_classes and class_name != "wall":
                continue
            if args.prefer_mode_photo and row["focus_crop"] and class_name != "wall":
                continue
            vg = row["village_group"]
            if vg in blocked or vg in used_groups:
                continue
            if args.dry_run:
                entry = {**row, "dest_split": dest_split, "class_name": class_name, "dry_run": True}
            else:
                entry = {**move_one(row, dest_split), "class_name": class_name}
            moved.append(entry)
            used_groups.add(vg)
            blocked.add(vg)
            got += 1

    report = {
        "updated": date.today().isoformat(),
        "purpose": "Rebalance val/test weak classes disjoint village_group",
        "dry_run": args.dry_run,
        "min_classes": args.min_classes,
        "moved_count": len(moved),
        "moved": [
            {
                "class_name": row.get("class_name"),
                "stem": row.get("stem"),
                "dest_split": row.get("dest_split"),
                "village_group": row.get("village_group"),
                "source": row.get("source"),
                "n_classes": row.get("n_classes"),
                "boxes": row.get("boxes"),
                "focus_crop": row.get("focus_crop"),
            }
            for row in moved
        ],
    }
    name = args.report_name or (
        f"perfect-rebalance-weak-val-test-{date.today().isoformat().replace('-', '')}.json"
    )
    out = ROOT / "training" / "reports" / name
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out), "moved_count": len(moved)}, indent=2))


if __name__ == "__main__":
    main()
