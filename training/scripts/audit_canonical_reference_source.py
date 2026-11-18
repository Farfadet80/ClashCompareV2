"""Audit a structured, level-by-level Clash of Clans image reference source.

This script deliberately does not copy the assets into the training dataset.
Canonical transparent PNGs are useful references, but they are not independent
gameplay screenshots and must not be counted as validation/test observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


SOURCE_ID_ALIASES = {
    "bob-hut": "bobs-hut",
}


def iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_strings(child)


def load_source_records(source: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((source / "data" / "home").rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("id"), str):
            payload["_source_json"] = path.relative_to(source).as_posix()
            records[payload["id"]] = payload
    return records


def image_metadata(source: Path, relative_path: str) -> dict[str, Any]:
    path = source / relative_path
    result: dict[str, Any] = {
        "path": relative_path,
        "exists": path.is_file(),
    }
    if not path.is_file():
        return result
    raw = path.read_bytes()
    with Image.open(path) as image:
        result.update(
            {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "has_alpha": "A" in image.getbands(),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return result


def source_revision(source: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def audit(project: Path, source: Path) -> dict[str, Any]:
    catalog = json.loads((project / "data" / "buildings.json").read_text(encoding="utf-8"))
    records = load_source_records(source)
    rows: list[dict[str, Any]] = []
    hashes: defaultdict[str, set[str]] = defaultdict(set)
    duplicate_level_records: dict[str, list[int]] = {}
    inconsistent_level_records: list[dict[str, Any]] = []
    reference_uses: defaultdict[str, set[str]] = defaultdict(set)

    expected_total = 0
    covered_total = 0
    existing_images = 0
    missing_images: list[str] = []

    for building in catalog["buildings"]:
        building_id = building["id"]
        expected_levels = sorted(int(level) for level in building.get("levels", []))
        expected_total += len(expected_levels)
        source_id = SOURCE_ID_ALIASES.get(building_id, building_id)
        record = records.get(source_id)
        levels_by_number: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        for level in (record or {}).get("levels", []):
            if isinstance(level, dict) and isinstance(level.get("level"), int):
                levels_by_number[int(level["level"])].append(level)
        duplicates = sorted(number for number, values in levels_by_number.items() if len(values) > 1)
        if duplicates:
            duplicate_level_records[building_id] = duplicates

        matched_levels: list[int] = []
        level_references: dict[str, list[dict[str, Any]]] = {}
        for level_number in expected_levels:
            refs = []
            candidate_paths: set[str] = set()
            level_records = levels_by_number.get(level_number, [])
            for level_record in level_records:
                paths = {
                    value
                    for value in iter_strings(level_record.get("images", {}))
                    if value.lower().endswith(".png")
                }
                coherent = {path for path in paths if Path(path).stem == f"level-{level_number}"}
                if paths and not coherent:
                    inconsistent_level_records.append(
                        {"id": building_id, "declared_level": level_number, "paths": sorted(paths)}
                    )
                candidate_paths.update(coherent)
                # Some buildings intentionally keep the same appearance across
                # adjacent levels (for example Blacksmith 1/2). Accept that alias
                # only when the declared level itself is not duplicated.
                if not coherent and len(level_records) == 1:
                    candidate_paths.update(paths)
            for relative_path in sorted(candidate_paths):
                if not relative_path.lower().endswith(".png"):
                    continue
                metadata = image_metadata(source, relative_path)
                refs.append(metadata)
                reference_uses[relative_path].add(f"{building_id}:{level_number}")
                if metadata["exists"]:
                    existing_images += 1
                    hashes[metadata["sha256"]].add(metadata["path"])
                else:
                    missing_images.append(relative_path)
            level_references[str(level_number)] = refs
            if refs and all(item["exists"] for item in refs):
                matched_levels.append(level_number)

        covered_total += len(matched_levels)

        rows.append(
            {
                "id": building_id,
                "category": building.get("category"),
                "expected_levels": expected_levels,
                "covered_levels": matched_levels,
                "missing_levels": sorted(set(expected_levels) - set(matched_levels)),
                "source_json": (record or {}).get("_source_json"),
                "references": level_references,
            }
        )

    duplicate_groups = [sorted(paths) for paths in hashes.values() if len(paths) > 1]
    reference_reuse_groups = {
        path: sorted(labels) for path, labels in reference_uses.items() if len(labels) > 1
    }
    return {
        "generated": date.today().isoformat(),
        "source": {
            "id": "chiefpansancolt-clash-of-clans-data",
            "url": "https://github.com/chiefpansancolt/clash-of-clans-data",
            "revision": source_revision(source),
            "repository_license": "MIT",
            "upstream_game_asset_policy": "Supercell Fan Content Policy",
            "use_mode": "canonical_reference_only",
        },
        "summary": {
            "catalog_buildings": len(rows),
            "expected_building_levels": expected_total,
            "covered_building_levels": covered_total,
            "coverage_percent": round(100 * covered_total / expected_total, 2) if expected_total else 0,
            "referenced_existing_pngs": existing_images,
            "missing_referenced_pngs": len(missing_images),
            "duplicate_file_groups": len(duplicate_groups),
            "buildings_with_duplicate_level_records": len(duplicate_level_records),
            "inconsistent_level_records": len(inconsistent_level_records),
            "shared_reference_groups": len(reference_reuse_groups),
        },
        "missing_referenced_pngs": sorted(set(missing_images)),
        "duplicate_file_groups": duplicate_groups,
        "duplicate_level_records": duplicate_level_records,
        "inconsistent_level_records": inconsistent_level_records,
        "shared_reference_groups": reference_reuse_groups,
        "buildings": rows,
        "training_policy": [
            "Do not add these canonical PNGs to validation or test splits.",
            "Do not report them as independent gameplay observations.",
            "Do not generate or redistribute modified derivatives without confirming upstream asset rights.",
            "Use them to verify class/level names and to guide review of real gameplay crops.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit(args.project.resolve(), args.source.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
