"""Smoke % maxage HDV sur inventaire JSON (sans navigateur)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def compute_maxage(inventory: dict, town_hall: int, caps: dict) -> dict:
    th_caps = caps["by_th"].get(str(town_hall))
    if not th_caps:
        return {"ok": False, "percent": None}
    sum_ratio = 0.0
    weight = 0
    remaining = 0
    for building_id, slot in inventory.items():
        max_lvl = th_caps.get(building_id)
        if max_lvl is None:
            continue
        for level_key, count in (slot.get("levels") or {}).items():
            if level_key == "inconnu":
                continue
            level = int(level_key)
            count = int(count)
            ratio = min(level, max_lvl) / max_lvl
            sum_ratio += ratio * count
            weight += count
            if level < max_lvl:
                remaining += count
    if weight == 0:
        return {"ok": False, "percent": None, "remaining": 0}
    return {
        "ok": True,
        "percent": round(1000 * sum_ratio / weight) / 10,
        "remaining": remaining,
        "weight": weight,
    }


def main() -> None:
    caps = json.loads((ROOT / "data" / "th-caps.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (
            ROOT
            / "training/curation/active-learning/jimmy-th16-20260905/ground-truth-export-summary.json"
        ).read_text(encoding="utf-8")
    )
    inventory = summary["inventory"]
    result = compute_maxage(inventory, 16, caps)
    assert result["ok"], result
    assert 0 < result["percent"] <= 100
    assert result["remaining"] > 0
    # Lab 14/14 à TH16 → ne doit pas compter comme reste
    lab_only = {"laboratory": {"total": 1, "levels": {"14": 1}}}
    full = compute_maxage(lab_only, 16, caps)
    assert full["ok"] and full["percent"] == 100.0 and full["remaining"] == 0
    # TH10–15 désormais supportés (ClashRecord)
    th10 = compute_maxage(
        {"cannon": {"total": 6, "levels": {"13": 6}}, "wall": {"total": 10, "levels": {"11": 10}}},
        10,
        caps,
    )
    assert th10["ok"] and th10["percent"] == 100.0 and th10["remaining"] == 0
    th12_partial = compute_maxage(
        {"workshop": {"total": 1, "levels": {"2": 1}}},
        12,
        caps,
    )
    assert th12_partial["ok"] and th12_partial["percent"] < 100 and th12_partial["remaining"] == 1
    # Hors table (sous TH10)
    assert compute_maxage(inventory, 9, caps)["ok"] is False
    print(
        f"OK maxage TH16 Jimmy ~{result['percent']} % "
        f"({result['remaining']} unités sous plafond, {result['weight']} scorées); "
        f"TH10/12 OK"
    )


if __name__ == "__main__":
    main()
