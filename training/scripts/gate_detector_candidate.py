"""Bloque la promotion d'un détecteur qui ne bat pas la baseline comparable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COMPARABLE_FIELDS = ("split", "imgsz", "augment_tta")
PROMOTION_METRICS = ("map50", "map50_95")


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Métriques absentes: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [
        key
        for key in (*COMPARABLE_FIELDS, *PROMOTION_METRICS)
        if key not in payload
    ]
    if missing:
        raise SystemExit(f"Métriques incomplètes dans {path}: {', '.join(missing)}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Gain absolu minimal requis sur chaque métrique de promotion.",
    )
    args = parser.parse_args()

    baseline = load_metrics(args.baseline.resolve())
    candidate = load_metrics(args.candidate.resolve())

    mismatches = [
        field
        for field in COMPARABLE_FIELDS
        if baseline[field] != candidate[field]
    ]
    if mismatches:
        details = ", ".join(
            f"{field}: {baseline[field]!r} != {candidate[field]!r}"
            for field in mismatches
        )
        raise SystemExit(f"Protocoles non comparables: {details}")

    comparison = {}
    passed = True
    for metric in PROMOTION_METRICS:
        reference = float(baseline[metric])
        value = float(candidate[metric])
        delta = value - reference
        metric_passed = delta > args.min_delta
        comparison[metric] = {
            "baseline": reference,
            "candidate": value,
            "delta": delta,
            "passed": metric_passed,
        }
        passed = passed and metric_passed

    result = {
        "comparable": True,
        "passed": passed,
        "min_delta": args.min_delta,
        "protocol": {field: baseline[field] for field in COMPARABLE_FIELDS},
        "metrics": comparison,
        "decision": "candidate_eligible_for_test" if passed else "keep_baseline",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
