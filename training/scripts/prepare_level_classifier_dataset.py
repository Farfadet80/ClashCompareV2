"""Prépare un dataset Ultralytics Classification équilibré depuis le manifeste."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("building")
    parser.add_argument("--min-train", type=int, default=50)
    parser.add_argument("--min-val", type=int, default=10)
    parser.add_argument("--max-train", type=int, default=1000)
    parser.add_argument("--max-val", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = TRAINING / "dataset" / "level-splits.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    groups: dict[tuple[int, str], list[Path]] = defaultdict(list)
    for row in rows:
        if row["building"] == args.building and row["split"] in {"train", "val"}:
            groups[(int(row["level"]), row["split"])].append(ROOT / row["path"])

    levels = sorted({level for level, _ in groups})
    eligible = [
        level for level in levels
        if len(groups[(level, "train")]) >= args.min_train
        and len(groups[(level, "val")]) >= args.min_val
    ]
    if len(eligible) < 2:
        raise SystemExit(
            f"Pas assez de niveaux prêts pour {args.building}. "
            f"Il faut au moins {args.min_train} train et {args.min_val} val par niveau."
        )

    output = TRAINING / "classification" / args.building
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Le dossier existe déjà: {output}. Supprime-le explicitement avant de le régénérer.")

    rng = random.Random(args.seed)
    summary: dict[str, dict[str, int]] = {}
    for level in eligible:
        summary[str(level)] = {}
        for split, maximum in (("train", args.max_train), ("val", args.max_val)):
            sources = sorted(set(groups[(level, split)]))
            rng.shuffle(sources)
            sources = sources[:maximum]
            destination = output / split / f"level-{level}"
            destination.mkdir(parents=True, exist_ok=True)
            for index, source in enumerate(sources):
                shutil.copy2(source, destination / f"{index:05d}-{source.name}")
            summary[str(level)][split] = len(sources)

    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Dataset préparé: {output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

