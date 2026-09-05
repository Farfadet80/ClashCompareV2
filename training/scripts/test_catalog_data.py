"""Contrôles simples des données catalogue vérifiées au 5 septembre 2026."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    payload = json.loads((ROOT / "data" / "buildings.json").read_text(encoding="utf-8"))
    rows = payload["buildings"]
    by_id = {row["id"]: row for row in rows}
    assert len(by_id) == len(rows), "IDs bâtiment dupliqués"

    expected_max = {
        "air-sweeper": 7,
        "inferno-tower": 12,
        "dark-elixir-drill": 11,
        "workshop": 9,
        "pet-house": 12,
        "blacksmith": 10,
        "hero-hall": 12,
        "multi-gear-tower": 3,
        "revenge-tower": 2,
        "super-wizard-tower": 2,
        "bomb": 14,
        "spring-trap": 13,
        "air-bomb": 13,
        "giant-bomb": 12,
        "seeking-air-mine": 8,
        "skeleton-trap": 5,
        "tornado-trap": 3,
        "giga-bomb": 4,
        "town-hall-guardian": 5,
    }
    for building, expected in expected_max.items():
        levels = by_id[building]["levels"]
        assert levels == list(range(1, expected + 1)), (
            f"{building}: niveaux {levels}, attendu 1..{expected}"
        )

    guardians = by_id["town-hall-guardian"]["variants"]
    assert guardians == ["smasher", "longshot", "logger"]

    crafted = by_id["crafted-defense"]
    assert crafted["current_variants"] == [
        "cake-a-pult",
        "hero-hunter",
        "hot-candle",
    ]
    assert set(crafted["current_variants"]).issubset(crafted["variants"])
    assert by_id["bob-hut"]["levels"] == [1]
    assert by_id["helper-hut"]["levels"] == [1]
    assert payload["updated_through"] == "2026-09-05"

    mapping_payload = json.loads(
        (ROOT / "data" / "coc-export-mapping.json").read_text(encoding="utf-8")
    )
    mapping = mapping_payload["dataIdToBuilding"]
    assert set(mapping.values()).issubset(by_id), "Le mapping export cible un ID inconnu"
    assert mapping["1000064"] == "bob-hut"
    assert mapping["1000093"] == "helper-hut"
    observed_home_ids = {
        "1000000", "1000001", "1000002", "1000003", "1000004", "1000005",
        "1000006", "1000007", "1000008", "1000009", "1000010", "1000011",
        "1000012", "1000013", "1000014", "1000015", "1000019", "1000020",
        "1000021", "1000023", "1000024", "1000026", "1000027", "1000028",
        "1000029", "1000031", "1000032", "1000059", "1000064", "1000067",
        "1000068", "1000070", "1000071", "1000072", "1000077", "1000084",
        "1000085", "1000093", "1000097", "12000000", "12000001", "12000002",
        "12000005", "12000006", "12000008", "12000016",
    }
    assert not (observed_home_ids - set(mapping)), (
        f"IDs export HDV16 non mappés: {sorted(observed_home_ids - set(mapping))}"
    )

    print(f"OK catalogue: {len(rows)} classes, niveaux récents vérifiés")


if __name__ == "__main__":
    main()
