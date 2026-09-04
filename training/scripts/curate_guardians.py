"""Crée des planches de contrôle autour des Town Halls TH18 annotés.

Ce script ne crée aucune annotation Guardian automatiquement. Il extrait seulement
des zones candidates afin qu'elles puissent être contrôlées visuellement avant
d'être ajoutées au dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def names_list(value: list[str] | dict[int | str, str]) -> list[str]:
    if isinstance(value, list):
        return value
    return [value[key] for key in sorted(value, key=lambda item: int(item))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--class-name", default="Town Hall 18")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=float, default=2.5)
    parser.add_argument("--tile", type=int, default=320)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()

    config_path = next(iter(args.source.glob("*.yaml")), None)
    if config_path is None:
        raise SystemExit("data.yaml introuvable")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    names = names_list(config["names"])
    class_id = names.index(args.class_name)
    candidates: list[dict[str, object]] = []

    for split in ("train", "valid", "val", "test"):
        image_dir = args.source / split / "images"
        label_dir = args.source / split / "labels"
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
            if not label_path.exists():
                continue
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5 or int(parts[0]) != class_id:
                    continue
                values = list(map(float, parts[1:]))
                if len(values) == 4:
                    x, y, w, h = values
                elif len(values) >= 6 and len(values) % 2 == 0:
                    xs, ys = values[0::2], values[1::2]
                    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
                    x, y = (left + right) / 2, (top + bottom) / 2
                    w, h = right - left, bottom - top
                else:
                    continue
                candidates.append({
                    "image": str(image_path),
                    "split": split,
                    "town_hall_box": [x, y, w, h],
                })

    args.output.mkdir(parents=True, exist_ok=True)
    tile = args.tile
    if not candidates:
        raise SystemExit(f"Aucune annotation trouvée pour {args.class_name}")
    rows = math.ceil(len(candidates) / args.columns)
    sheet = Image.new("RGB", (args.columns * tile, rows * tile), "#202020")
    crops_dir = args.output / "crops"
    pages_dir = args.output / "pages"
    crops_dir.mkdir(exist_ok=True)
    pages_dir.mkdir(exist_ok=True)
    canvases: list[Image.Image] = []

    for index, item in enumerate(candidates):
        source = Image.open(str(item["image"])).convert("RGB")
        x, y, w, h = item["town_hall_box"]  # type: ignore[misc]
        width, height = source.size
        crop_w = max(w * width * args.context, 80)
        crop_h = max(h * height * args.context, 80)
        cx, cy = x * width, y * height
        left = max(0, int(cx - crop_w / 2))
        top = max(0, int(cy - crop_h / 2))
        right = min(width, int(cx + crop_w / 2))
        bottom = min(height, int(cy + crop_h / 2))
        crop = source.crop((left, top, right, bottom))
        scale = min(tile / crop.width, (tile - 24) / crop.height)
        crop = crop.resize(
            (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
            Image.Resampling.LANCZOS,
        )
        canvas = Image.new("RGB", (tile, tile), "#202020")
        canvas.paste(crop, ((tile - crop.width) // 2, 24 + (tile - 24 - crop.height) // 2))
        canvas_draw = ImageDraw.Draw(canvas)
        canvas_draw.text((8, 6), f"{index:03d} {item['split']}", fill="white")
        canvas.save(crops_dir / f"{index:03d}.jpg", quality=94)
        canvases.append(canvas)
        sheet.paste(canvas, ((index % args.columns) * tile, (index // args.columns) * tile))
        item["crop_pixels"] = [left, top, right, bottom]

    sheet.save(args.output / "town-hall-18-contact-sheet.jpg", quality=92)
    page_size = args.columns * args.columns
    for start in range(0, len(canvases), page_size):
        page = Image.new("RGB", (args.columns * tile, args.columns * tile), "#202020")
        for offset, canvas in enumerate(canvases[start : start + page_size]):
            page.paste(canvas, ((offset % args.columns) * tile, (offset // args.columns) * tile))
        page.save(pages_dir / f"page-{start // page_size + 1:02d}.jpg", quality=94)
    (args.output / "candidates.json").write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{len(candidates)} zones TH18 extraites dans {args.output}")


if __name__ == "__main__":
    main()
