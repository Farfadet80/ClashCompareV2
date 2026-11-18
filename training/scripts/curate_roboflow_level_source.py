"""Nettoie un export Roboflow de niveaux avant extraction des crops.

Roboflow peut exporter plusieurs augmentations d'une même image source. Ce script
regroupe ces variantes grâce au nom ``*.rf.*``, choisit la variante médiane la
plus représentative, ignore les images sans classe de niveau connue et refuse
les familles qui traversent plusieurs splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def original_key(path: Path) -> str:
    return path.stem.split(".rf.", 1)[0]


def difference_hash(path: Path) -> int:
    with Image.open(path) as opened:
        pixels = list(opened.convert("L").resize((17, 16)).get_flattened_data())
    value = 0
    for row in range(16):
        offset = row * 17
        for column in range(16):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def choose_medoid(paths: list[Path]) -> Path:
    if len(paths) == 1:
        return paths[0]
    hashes = {path: difference_hash(path) for path in paths}
    return min(
        paths,
        key=lambda candidate: (
            sum((hashes[candidate] ^ hashes[other]).bit_count() for other in paths),
            candidate.name,
        ),
    )


def foreground_score(path: Path) -> int:
    """Approxime la surface visible d'un sprite sur fond noir.

    Certaines versions Roboflow ne publient que des variantes augmentées avec
    de grands masques noirs. Dans ce cas, le meilleur exemplaire est celui qui
    conserve le plus de pixels non noirs, plutôt que le médoïde perceptuel.
    """
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.thumbnail((160, 160))
        return sum(1 for pixel in image.get_flattened_data() if max(pixel) > 24)


def choose_most_foreground(paths: list[Path]) -> Path:
    return max(paths, key=lambda path: (foreground_score(path), path.name))


def names_list(value: list[str] | dict[int | str, str]) -> list[str]:
    if isinstance(value, list):
        return value
    return [value[key] for key in sorted(value, key=lambda item: int(item))]


def mapped_lines(
    label_path: Path,
    names: list[str],
    mapping: dict[str, object],
    image_path: Path | None = None,
    min_aspect: float = 0.0,
    max_aspect: float = float("inf"),
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    if not label_path.exists():
        return result
    image_size: tuple[int, int] | None = None
    if image_path is not None:
        with Image.open(image_path) as opened:
            image_size = opened.size
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            continue
        if not (0 <= class_id < len(names) and names[class_id] in mapping):
            continue

        # Les exports YOLO de segmentation contiennent une classe suivie des
        # points x/y du polygone. Le classificateur de niveaux a seulement
        # besoin du crop : on convertit donc ce contour en boîte englobante.
        if len(parts) == 5:
            box = tuple(float(value) for value in parts[1:5])
        elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
            coordinates = [float(value) for value in parts[1:]]
            xs = coordinates[0::2]
            ys = coordinates[1::2]
            left, right = min(xs), max(xs)
            top, bottom = min(ys), max(ys)
            box = ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
        else:
            continue
        if not all(0.0 <= value <= 1.0 for value in box) or box[2] <= 0 or box[3] <= 0:
            continue
        if image_size is not None:
            width, height = image_size
            box_width = box[2] * width
            box_height = box[3] * height
            aspect = box_width / max(1.0, box_height)
            if box_width < 10 or box_height < 10 or not min_aspect <= aspect <= max_aspect:
                continue
        normalized_box = " ".join(f"{value:.8f}" for value in box)
        result.append((names[class_id], f"{class_id} {normalized_box}"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("source_key")
    parser.add_argument("--min-aspect", type=float, default=0.45)
    parser.add_argument("--max-aspect", type=float, default=2.2)
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="Conserve chaque image lorsque la version Roboflow déclare zéro augmentation.",
    )
    parser.add_argument(
        "--prefer-most-foreground",
        action="store_true",
        help="Choisit la variante au sprite le moins masqué sur fond noir.",
    )
    parser.add_argument(
        "--single-label-images",
        action="store_true",
        help="Écarte les planches multi-bâtiments, utiles comme référence mais trop ambiguës pour l'entraînement.",
    )
    parser.add_argument(
        "--max-boxes-per-class-per-image",
        type=int,
        default=0,
        help="Limite déterministe des objets corrélés d'une même classe dans une image (0 = sans limite).",
    )
    parser.add_argument(
        "--dedupe-cross-split-dhash",
        type=int,
        default=-1,
        help="Écarte les quasi-doublons entre splits jusqu'à cette distance dHash (-1 = désactivé).",
    )
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Le dossier de sortie existe déjà et n'est pas vide: {args.output}")

    registry = json.loads((TRAINING / "sources" / "level-datasets.json").read_text(encoding="utf-8"))
    if args.source_key not in registry:
        raise SystemExit(f"Source inconnue: {args.source_key}")
    mapping = registry[args.source_key]["mapping"]
    config_path = next(iter(args.source.glob("*.yaml")), None) or next(iter(args.source.glob("*.yml")), None)
    if config_path is None:
        raise SystemExit("data.yaml absent")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    names = names_list(config["names"])

    groups: dict[tuple[str, str], list[tuple[Path, Path]]] = defaultdict(list)
    key_splits: dict[str, set[str]] = defaultdict(set)
    for source_split in ("train", "valid", "val", "test"):
        image_dir = args.source / source_split / "images"
        label_dir = args.source / source_split / "labels"
        if not image_dir.exists():
            continue
        target_split = "valid" if source_split == "val" else source_split
        for image_path in sorted(image_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = (label_dir / image_path.relative_to(image_dir)).with_suffix(".txt")
            labels = mapped_lines(label_path, names, mapping)
            if not labels or (args.single_label_images and len(labels) != 1):
                continue
            key = image_path.stem if args.keep_all else original_key(image_path)
            key_splits[key].add(target_split)
            groups[(target_split, key)].append((image_path, label_path))

    leaking = {key: sorted(splits) for key, splits in key_splits.items() if len(splits) > 1}
    if leaking:
        raise SystemExit(f"Familles présentes dans plusieurs splits: {json.dumps(leaking, indent=2)}")

    candidates: list[dict[str, object]] = []
    for (split, key), pairs in sorted(groups.items()):
        chooser = choose_most_foreground if args.prefer_most_foreground else choose_medoid
        chosen_image = chooser([image for image, _ in pairs])
        chosen_label = next(label for image, label in pairs if image == chosen_image)
        accepted = mapped_lines(
            chosen_label,
            names,
            mapping,
            chosen_image,
            args.min_aspect,
            args.max_aspect,
        )
        if not accepted:
            continue
        if args.max_boxes_per_class_per_image > 0:
            by_class: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for item in accepted:
                by_class[item[0]].append(item)
            accepted = []
            for class_name, items in sorted(by_class.items()):
                ranked = sorted(
                    items,
                    key=lambda item: hashlib.sha1(
                        f"{key}\0{item[1]}".encode("utf-8")
                    ).hexdigest(),
                )
                accepted.extend(ranked[: args.max_boxes_per_class_per_image])
        candidates.append({
            "key": key,
            "split": split,
            "image": chosen_image,
            "label": chosen_label,
            "accepted": accepted,
            "variants": len(pairs),
        })

    split_priority = {"test": 0, "valid": 1, "train": 2}
    accepted_candidates: list[dict[str, object]] = []
    exact_hashes: set[str] = set()
    fingerprints: list[tuple[str, int]] = []
    dropped_exact_duplicates = 0
    dropped_cross_split_near_duplicates = 0
    for candidate in sorted(
        candidates,
        key=lambda item: (split_priority.get(str(item["split"]), 9), str(item["key"])),
    ):
        image_path = candidate["image"]
        assert isinstance(image_path, Path)
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest in exact_hashes:
            dropped_exact_duplicates += 1
            continue
        fingerprint = difference_hash(image_path)
        split = str(candidate["split"])
        if args.dedupe_cross_split_dhash >= 0 and any(
            other_split != split
            and (fingerprint ^ other_fingerprint).bit_count() <= args.dedupe_cross_split_dhash
            for other_split, other_fingerprint in fingerprints
        ):
            dropped_cross_split_near_duplicates += 1
            continue
        exact_hashes.add(digest)
        fingerprints.append((split, fingerprint))
        accepted_candidates.append(candidate)

    class_counts: Counter[str] = Counter()
    selected: list[dict[str, object]] = []
    for candidate in sorted(accepted_candidates, key=lambda item: (str(item["split"]), str(item["key"]))):
        split = str(candidate["split"])
        key = str(candidate["key"])
        chosen_image = candidate["image"]
        chosen_label = candidate["label"]
        accepted = candidate["accepted"]
        assert isinstance(chosen_image, Path)
        assert isinstance(chosen_label, Path)
        assert isinstance(accepted, list)
        destination_images = args.output / split / "images"
        destination_labels = args.output / split / "labels"
        destination_images.mkdir(parents=True, exist_ok=True)
        destination_labels.mkdir(parents=True, exist_ok=True)
        shutil.copy2(chosen_image, destination_images / chosen_image.name)
        (destination_labels / chosen_label.name).write_text(
            "\n".join(line for _, line in accepted) + "\n", encoding="utf-8"
        )
        classes = [name for name, _ in accepted]
        class_counts.update(classes)
        selected.append({
            "original_key": key,
            "split": split,
            "selected": chosen_image.name,
            "variants": candidate["variants"],
            "mapped_classes": classes,
        })

    output_config = dict(config)
    output_config["train"] = "../train/images"
    output_config["val"] = "../valid/images"
    output_config["test"] = "../test/images"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "data.yaml").write_text(
        yaml.safe_dump(output_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    report = {
        "source": str(args.source),
        "source_key": args.source_key,
        "selection_policy": "most_foreground" if args.prefer_most_foreground else "dhash_medoid",
        "selected_images": len(selected),
        "mapped_boxes": dict(sorted(class_counts.items())),
        "dropped_exact_duplicate_images": dropped_exact_duplicates,
        "dropped_cross_split_near_duplicate_images": dropped_cross_split_near_duplicates,
        "selections": selected,
    }
    (args.output / "curation-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "selections"}, indent=2))


if __name__ == "__main__":
    main()
