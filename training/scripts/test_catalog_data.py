"""Contrôles catalogue + mapping + plafonds HDV (TH19, 2026-11-18)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    payload = json.loads((ROOT / "data" / "buildings.json").read_text(encoding="utf-8"))
    rows = payload["buildings"]
    by_id = {row["id"]: row for row in rows}
    assert len(by_id) == len(rows), "IDs bâtiment dupliqués"
    assert payload["updated_through"] == "2026-11-18"
    assert payload.get("max_town_hall") == 19

    expected_max = {
        "town-hall": 19,
        "air-defense": 17,
        "hidden-tesla": 18,
        "x-bow": 14,
        "builders-hut": 9,
        "multi-archer-tower": 5,
        "multi-gear-tower": 4,
        "ricochet-cannon": 5,
        "wall": 20,
        "clan-castle": 15,
        "laboratory": 17,
        "hero-hall": 13,
        "gold-storage": 20,
        "elixir-storage": 20,
        "dark-elixir-storage": 14,
        "spring-trap": 14,
        "air-bomb": 14,
        "skeleton-trap": 6,
        "spiky-tower": 2,
        "firecracker-tower": 2,
        "guardian-shrine": 1,
        "squad-barracks": 12,
        "town-hall-guardian": 10,
        "scattershot": 7,
        "bomb-tower": 13,
        "spell-tower": 4,
        "army-camp": 14,
        "barracks": 19,
        "spell-factory": 10,
        "dark-spell-factory": 8,
    }
    for building, expected in expected_max.items():
        levels = by_id[building]["levels"]
        assert levels == list(range(1, expected + 1)), (
            f"{building}: niveaux {levels}, attendu 1..{expected}"
        )

    guardians = by_id["town-hall-guardian"]
    assert guardians["variants"] == ["smasher", "longshot", "logger"]
    assert guardians["variant_max_levels"]["smasher"] == 10
    assert guardians["variant_max_levels"]["longshot"] == 5

    crafted = by_id["crafted-defense"]
    assert crafted["current_variants"] == [
        "cake-a-pult",
        "hero-hunter",
        "hot-candle",
    ]
    assert set(crafted["current_variants"]).issubset(crafted["variants"])

    mapping_payload = json.loads(
        (ROOT / "data" / "coc-export-mapping.json").read_text(encoding="utf-8")
    )
    mapping = mapping_payload["dataIdToBuilding"]
    assert set(mapping.values()).issubset(by_id), "Le mapping export cible un ID inconnu"
    assert mapping["1000064"] == "bob-hut"
    assert mapping["1000093"] == "helper-hut"
    pending = {row["id"] for row in mapping_payload.get("pendingNewBuildings", [])}
    assert pending >= {
        "spiky-tower",
        "firecracker-tower",
        "guardian-shrine",
        "squad-barracks",
    }
    for row in mapping_payload["pendingNewBuildings"]:
        assert row.get("dataId") is None, "Ne pas inventer de dataId sans export observé"

    caps = json.loads((ROOT / "data" / "th-caps.json").read_text(encoding="utf-8"))
    assert caps["max_town_hall"] == 19
    assert caps["by_th"]["19"]["air-defense"] == 17
    assert caps["by_th"]["19"]["wall"] == 20
    assert caps["by_th"]["19"]["spiky-tower"] == 2
    assert caps["by_th"]["18"]["air-defense"] == 16
    assert caps["by_th"]["16"]["laboratory"] == 14

    print(f"OK catalogue: {len(rows)} classes, TH19 + th-caps vérifiés")


if __name__ == "__main__":
    main()
