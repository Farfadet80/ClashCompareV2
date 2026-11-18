"""Prépare un dataset Ultralytics Classification équilibré depuis le manifeste."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"


def source_family(path: Path) -> str:
    """Identifiant de la capture source, sans l'index du crop final."""
    return re.sub(r"-\d+$", "", path.stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("building")
    parser.add_argument(
        "--output-name",
        default=None,
        help="Nom du dossier de sortie; le bâtiment par défaut.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=None,
        help="Niveaux à conserver; tous les niveaux éligibles par défaut.",
    )
    parser.add_argument("--min-train", type=int, default=50)
    parser.add_argument("--min-val", type=int, default=10)
    parser.add_argument(
        "--derive-val-levels",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Niveaux sans validation publiée pour lesquels réserver des familles "
            "d'images train entières comme validation."
        ),
    )
    parser.add_argument("--derived-val-ratio", type=float, default=0.2)
    parser.add_argument("--max-train", type=int, default=1000)
    parser.add_argument("--max-val", type=int, default=200)
    parser.add_argument("--max-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-reference-only",
        action="store_true",
        help="Inclure explicitement les sources de référence visuelle (jamais pour un test indépendant).",
    )
    args = parser.parse_args()

    manifest = TRAINING / "dataset" / "level-splits.jsonl"
    source_registry = json.loads((TRAINING / "sources" / "level-datasets.json").read_text(encoding="utf-8"))
    reference_only = {
        source_id for source_id, metadata in source_registry.items()
        if metadata.get("role") == "reference_only"
    }
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    groups: dict[tuple[int, str], list[Path]] = defaultdict(list)
    for row in rows:
        if row.get("source") in reference_only and (
            not args.include_reference_only or row.get("split") != "train"
        ):
            continue
        if row["building"] == args.building and row["split"] in {"train", "val", "test"}:
            groups[(int(row["level"]), row["split"])].append(ROOT / row["path"])

    # Plusieurs crops peuvent venir de la même capture de village. Ils doivent
    # rester dans le même split, sinon la validation mesure surtout la mémoire
    # du décor. Le nom produit par import_level_crops.py se termine par l'index
    # du crop; on retire cet index pour retrouver la famille source.
    derived_validation: dict[str, dict[str, int]] = {}
    for level in args.derive_val_levels or []:
        validation = groups[(level, "val")]
        training = groups[(level, "train")]
        if validation or not training:
            continue
        families: dict[str, list[Path]] = defaultdict(list)
        for path in training:
            family = source_family(path)
            families[family].append(path)
        if len(families) < 2:
            continue
        ordered = sorted(
            families,
            key=lambda family: hashlib.sha256(
                f"{args.seed}:{level}:{family}".encode("utf-8")
            ).hexdigest(),
        )
        target = max(args.min_val, round(len(training) * args.derived_val_ratio))
        selected: list[str] = []
        selected_count = 0
        for family in ordered[:-1]:
            selected.append(family)
            selected_count += len(families[family])
            if selected_count >= target:
                break
        selected_families = set(selected)
        selected_paths: set[Path] = set()
        # Une même capture peut contenir plusieurs niveaux. Toute la famille
        # bascule en validation, pas seulement les crops du niveau demandé.
        for (other_level, split), paths in list(groups.items()):
            if split != "train":
                continue
            moved = [path for path in paths if source_family(path) in selected_families]
            if not moved:
                continue
            selected_paths.update(moved)
            groups[(other_level, "train")] = [path for path in paths if path not in moved]
            groups[(other_level, "val")].extend(moved)
        derived_validation[str(level)] = {
            "families": len(selected),
            "crops": len(selected_paths),
        }

    requested_levels = set(args.levels) if args.levels else None
    levels = sorted(
        level
        for level, _ in groups
        if requested_levels is None or level in requested_levels
    )
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

    output = TRAINING / "classification" / (args.output_name or args.building)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Le dossier existe déjà: {output}. Supprime-le explicitement avant de le régénérer.")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".preparing"
    try:
        lock_path.touch(exist_ok=False)
    except FileExistsError:
        raise SystemExit(f"Une autre préparation utilise déjà le dossier: {output}") from None

    rng = random.Random(args.seed)
    selected_sources: dict[tuple[int, str], list[Path]] = {}
    for level in eligible:
        for split, maximum in (
            ("train", args.max_train),
            ("val", args.max_val),
            ("test", args.max_test),
        ):
            sources = sorted(set(groups[(level, split)]))
            rng.shuffle(sources)
            selected_sources[(level, split)] = sources[:maximum]

    # Un crop identique ne doit jamais traverser les splits. On privilégie le
    # test, puis la validation, puis l'entraînement.
    seen_hashes: set[str] = set()
    exact_duplicates_removed: dict[str, int] = defaultdict(int)
    for split in ("test", "val", "train"):
        for level in eligible:
            unique: list[Path] = []
            for source in selected_sources[(level, split)]:
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                if digest in seen_hashes:
                    exact_duplicates_removed[split] += 1
                    continue
                seen_hashes.add(digest)
                unique.append(source)
            selected_sources[(level, split)] = unique

    summary: dict[str, dict[str, int]] = {}
    for level in eligible:
        summary[str(level)] = {}
        for split in ("train", "val", "test"):
            sources = selected_sources[(level, split)]
            destination = output / split / f"level-{level}"
            destination.mkdir(parents=True, exist_ok=True)
            for source in sources:
                # Nom déterministe : deux préparations concurrentes écrasent le
                # même fichier au lieu de multiplier silencieusement les crops.
                source_key = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
                shutil.copy2(source, destination / f"{source_key}-{source.name}")
            summary[str(level)][split] = len(sources)

    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "split-provenance.json").write_text(
        json.dumps({
            "derived_validation": derived_validation,
            "exact_duplicates_removed": dict(exact_duplicates_removed),
        }, indent=2),
        encoding="utf-8",
    )
    lock_path.unlink()
    print(f"Dataset préparé: {output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
