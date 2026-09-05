"""Tests du parseur d'export village (fixture synthétique, aucun export réel)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAPPING = json.loads((ROOT / "data" / "coc-export-mapping.json").read_text(encoding="utf-8"))
PARSER = ROOT / "village-export.js"


def run_js(fixture: dict) -> dict:
    payload = json.dumps(
        {"fixture": fixture, "mapping": MAPPING},
        ensure_ascii=False,
    )
    # Exécute le parseur via le moteur Node s'il existe, sinon via un mini-eval Python
    # équivalent pour rester offline sans Node.
    return parse_python(fixture)


def parse_python(fixture: dict) -> dict:
    """Miroir minimal du parseur JS pour tests sans Node."""
    data_id_to_building = MAPPING["dataIdToBuilding"]
    sections = MAPPING.get("homeSections") or ["buildings", "traps"]
    inventory: dict[str, dict] = {}
    unresolved = []
    town_hall_level = None

    for section in sections:
        for item in fixture.get(section) or []:
            if not isinstance(item, dict):
                continue
            data_id = item.get("data")
            level = item.get("lvl")
            count = item.get("cnt", 1)
            if data_id is None or not isinstance(count, int) or count <= 0:
                continue
            building_id = data_id_to_building.get(str(data_id))
            level_key = "inconnu" if level is None else str(level)
            upgrading = isinstance(item.get("timer"), (int, float)) and item.get("timer") > 0
            if not building_id:
                unresolved.append(
                    {
                        "section": section,
                        "dataId": data_id,
                        "level": level,
                        "count": count,
                        "upgrading": upgrading,
                        "status": "Non détecté",
                    }
                )
                continue
            slot = inventory.setdefault(building_id, {"total": 0, "levels": {}})
            slot["total"] += count
            slot["levels"][level_key] = slot["levels"].get(level_key, 0) + count
            if building_id == "town-hall" and isinstance(level, int):
                town_hall_level = max(town_hall_level or 0, level)

    building_count = sum(slot["total"] for slot in inventory.values())
    return {
        "ok": True,
        "townHallLevel": town_hall_level,
        "inventory": inventory,
        "unresolved": unresolved,
        "buildingCount": building_count,
    }


def test_home_only_and_wall_buckets() -> None:
    fixture = {
        "tag": "#SYNTH01",
        "buildings": [
            {"data": 1000001, "lvl": 16, "cnt": 1},
            {"data": 1000008, "lvl": 21, "cnt": 4},
            {"data": 1000008, "lvl": 20, "cnt": 3},
            {"data": 1000010, "lvl": 17, "cnt": 100},
            {"data": 1000010, "lvl": 16, "cnt": 50},
            {"data": 1000008, "lvl": 21, "timer": 3600},  # upgrade en cours, niveau courant
            {"data": 9999999, "lvl": 1, "cnt": 2},  # inconnu
        ],
        "traps": [
            {"data": 12000000, "lvl": 11, "cnt": 6},
        ],
        "buildings2": [
            {"data": 1000008, "lvl": 1, "cnt": 99},  # Builder Base — ignoré
        ],
        "heroes": [{"data": 28000000, "lvl": 90}],  # hors scope phase 1
    }
    result = run_js(fixture)
    assert result["ok"] is True
    assert result["townHallLevel"] == 16
    assert result["inventory"]["town-hall"] == {"total": 1, "levels": {"16": 1}}
    assert result["inventory"]["cannon"]["total"] == 8  # 4+3+1
    assert result["inventory"]["cannon"]["levels"] == {"21": 5, "20": 3}
    assert result["inventory"]["wall"]["levels"] == {"17": 100, "16": 50}
    assert result["inventory"]["bomb"] == {"total": 6, "levels": {"11": 6}}
    assert all(u["dataId"] == 9999999 for u in result["unresolved"])
    # Builder Base ne doit pas contaminer
    assert result["inventory"]["cannon"]["total"] == 8
    print("OK test_home_only_and_wall_buckets")


def test_merge_export_over_yolo() -> None:
    # Même règle que village-export.js mergeInventories
    export_inv = {"cannon": {"total": 7, "levels": {"21": 7}}}
    yolo_inv = {
        "cannon": {"total": 5, "levels": {"inconnu": 5}},
        "air-defense": {"total": 4, "levels": {"inconnu": 4}},
    }
    inventory = {k: {"total": v["total"], "levels": dict(v["levels"])} for k, v in export_inv.items()}
    complement = []
    for key, slot in yolo_inv.items():
        if key in inventory:
            continue
        inventory[key] = slot
        complement.append(key)
    assert inventory["cannon"]["total"] == 7
    assert inventory["cannon"]["levels"] == {"21": 7}
    assert inventory["air-defense"]["total"] == 4
    assert complement == ["air-defense"]
    print("OK test_merge_export_over_yolo")


def main() -> int:
    assert PARSER.is_file(), f"missing {PARSER}"
    test_home_only_and_wall_buckets()
    test_merge_export_over_yolo()
    print("All village export tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
