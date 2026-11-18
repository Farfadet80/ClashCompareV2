"""Audit des niveaux annotés, sans confondre crops et captures indépendantes."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "training/dataset/level-splits.jsonl"
CATALOG = ROOT / "data/buildings.json"
DETECTOR_DATASET = ROOT / "training/dataset.yaml"
DETECTOR_COVERAGE = ROOT / "training/reports/class-coverage.json"
LEVEL_SOURCES = ROOT / "training/sources/level-datasets.json"
REPORTS = ROOT / "training/reports"
SPLITS = ("train", "val", "test")
SOURCE_IMAGE = re.compile(r"^(.*)-(?:train|val|test)-([0-9a-f]{12})-\d+$")


def family(row: dict) -> str:
    """Regroupe les crops d'une image source, même si son split a changé."""
    stem = Path(row["path"]).stem
    match = SOURCE_IMAGE.match(stem)
    if match:
        return f"{match.group(1)}:{match.group(2)}"
    return re.sub(r"-\d+$", "", stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORTS / f"level-readiness-{date.today():%Y%m%d}.json")
    args = parser.parse_args()

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))["buildings"]
    catalog_names = {item["id"] for item in catalog}
    detector_names_raw = yaml.safe_load(DETECTOR_DATASET.read_text(encoding="utf-8"))["names"]
    detector_names = set(detector_names_raw.values() if isinstance(detector_names_raw, dict) else detector_names_raw)
    detector_coverage = json.loads(DETECTOR_COVERAGE.read_text(encoding="utf-8"))
    source_registry = json.loads(LEVEL_SOURCES.read_text(encoding="utf-8"))
    reference_only = {source_id for source_id, metadata in source_registry.items() if metadata.get("role") == "reference_only"}
    expected = {(item["id"], int(level)) for item in catalog for level in item["levels"]}
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    all_families = {split: set() for split in SPLITS}
    total = 0
    excluded_reference_crops = 0
    missing_files = []
    for line_number, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") not in SPLITS:
            continue
        total += 1
        if row.get("source") in reference_only:
            excluded_reference_crops += 1
            continue
        key = (row["building"], int(row["level"]))
        groups[key].append(row)
        all_families[row["split"]].add(family(row))
        if not (ROOT / row["path"]).is_file():
            missing_files.append({"line": line_number, "path": row["path"]})

    rows = []
    for building, level in sorted(expected | groups.keys()):
        samples = groups[(building, level)]
        by_split = {split: [row for row in samples if row["split"] == split] for split in SPLITS}
        families = {split: {family(row) for row in by_split[split]} for split in SPLITS}
        sources = {split: sorted({row.get("source", "unknown") for row in by_split[split]}) for split in SPLITS}
        overlap = sorted((families["train"] & families["val"]) | (families["train"] & families["test"]) | (families["val"] & families["test"]))
        counts = {split: len(by_split[split]) for split in SPLITS}
        family_counts = {split: len(families[split]) for split in SPLITS}
        if (building, level) not in expected:
            status = "outside_catalog"
        elif not samples:
            status = "missing"
        elif overlap:
            status = "split_family_overlap"
        elif not counts["test"]:
            status = "no_test"
        elif counts["train"] < 50 or counts["val"] < 10:
            status = "sparse_train_or_val"
        elif counts["test"] < 10 or family_counts["test"] < 3:
            status = "sparse_test"
        else:
            status = "data_candidate"
        rows.append({
            "building": building,
            "level": level,
            "status": status,
            "crops": counts,
            "source_images": family_counts,
            "sources": sources,
            "overlapping_source_images": overlap[:20],
        })

    status_counts = Counter(row["status"] for row in rows)
    global_overlap = sorted((all_families["train"] & all_families["val"]) | (all_families["train"] & all_families["test"]) | (all_families["val"] & all_families["test"]))
    by_building = []
    for item in catalog:
        selected = [row for row in rows if row["building"] == item["id"] and row["status"] != "outside_catalog"]
        by_building.append({
            "building": item["id"],
            "catalog_levels": len(selected),
            "data_candidates": sum(row["status"] == "data_candidate" for row in selected),
            "missing_levels": [row["level"] for row in selected if row["status"] == "missing"],
            "levels_without_test": [row["level"] for row in selected if row["status"] == "no_test"],
        })
    payload = {
        "date": date.today().isoformat(),
        "manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "manifest_crops": total,
        "eligible_crops": total - excluded_reference_crops,
        "excluded_reference_only_crops": excluded_reference_crops,
        "reference_only_sources": sorted(reference_only),
        "catalog_buildings": len(catalog),
        "catalog_building_levels": len(expected),
        "detector": {
            "classes": len(detector_names),
            "catalog_classes_absent": sorted(catalog_names - detector_names),
            "classes_without_training_boxes": detector_coverage.get("unlearnable", []),
            "classes_without_evaluation_boxes": detector_coverage.get("no_eval_split", []),
            "coverage_report": str(DETECTOR_COVERAGE.relative_to(ROOT)).replace("\\", "/"),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "cross_building_source_family_overlap_count": len(global_overlap),
        "cross_building_source_family_overlap_examples": global_overlap[:50],
        "missing_files": missing_files[:50],
        "missing_file_count": len(missing_files),
        "criteria": {
            "data_candidate": "Au moins 50 train, 10 val et 10 test, 3 images sources test, sans recouvrement de familles entre splits.",
            "caution": "Un candidat n'est PAS un modèle validé : les images sources peuvent être des copies, des sprites ou des vues trop similaires. Un vrai test gameplay indépendant et un audit visuel restent nécessaires.",
            "source_image": "Identifiant d'image dérivé du nom du crop importé ; approximation, pas une preuve d'indépendance.",
            "manifest": "Comptes du manifeste brut ; les validations dérivées lors de préparations ultérieures ne sont pas incluses.",
            "reference_only": "Les crops de sources visuelles isolées sont conservés au manifeste pour traçabilité, mais exclus des comptes de préparation par défaut.",
        },
        "by_building": by_building,
        "levels": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "catalog_buildings": payload["catalog_buildings"],
        "catalog_building_levels": payload["catalog_building_levels"],
        "detector": payload["detector"],
        "manifest_crops": total,
        "excluded_reference_only_crops": excluded_reference_crops,
        "status_counts": payload["status_counts"],
        "missing_file_count": len(missing_files),
        "cross_building_source_family_overlap_count": len(global_overlap),
        "report": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
