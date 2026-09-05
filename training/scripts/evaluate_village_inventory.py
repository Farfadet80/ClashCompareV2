"""Compare les quantités YOLO à un inventaire officiel agrégé.

Ce contrôle ne mesure pas l'IoU : une quantité correcte peut masquer une mauvaise
boîte. Les scores sont donc des bornes optimistes utiles au triage, pas une mAP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def totals(inventory: dict) -> dict[str, int]:
    return {name: int(slot.get("total", 0)) for name, slot in inventory.items()}


def score(expected: dict[str, int], predicted: dict[str, int]) -> dict:
    names = sorted(set(expected) | set(predicted))
    expected_total = sum(expected.values())
    predicted_total = sum(predicted.values())
    matched = sum(min(expected.get(name, 0), predicted.get(name, 0)) for name in names)
    return {
        "expected": expected_total,
        "predicted": predicted_total,
        "matched_upper_bound": matched,
        "count_precision_upper_bound": round(matched / predicted_total, 4)
        if predicted_total
        else 0,
        "count_recall_upper_bound": round(matched / expected_total, 4)
        if expected_total
        else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("inference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ground = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    inference = json.loads(args.inference.read_text(encoding="utf-8"))
    expected = totals(ground["inventory"])
    predicted = totals(inference["inventory"])
    non_wall = {name: count for name, count in expected.items() if name != "wall"}

    deltas = []
    for name in sorted(set(expected) | set(predicted)):
        wanted = expected.get(name, 0)
        found = predicted.get(name, 0)
        if wanted != found:
            deltas.append(
                {
                    "building": name,
                    "expected": wanted,
                    "predicted": found,
                    "delta": found - wanted,
                }
            )

    payload = {
        "ground_truth": str(args.ground_truth),
        "inference": str(args.inference),
        "warning": (
            "Scores optimistes par quantités uniquement ; ils ne valident ni les boîtes "
            "ni l'identité instance par instance."
        ),
        "all_objects": score(expected, predicted),
        "excluding_walls": score(non_wall, predicted),
        "exact_count_classes": sum(
            1 for name, wanted in expected.items() if predicted.get(name, 0) == wanted
        ),
        "expected_classes": len(expected),
        "deltas": sorted(deltas, key=lambda row: (-abs(row["delta"]), row["building"])),
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    else:
        print(text)


if __name__ == "__main__":
    main()
