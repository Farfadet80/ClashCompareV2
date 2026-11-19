"""Extrait des tuiles Mode photo mur-only exhaustives (P0).

Depuis sessions mur clashia/kiet (villages), découpe des tuiles où:
- ≥ min_walls boîtes wall acceptées
- V5 ne détecte aucune autre classe (conf ≥ other_conf) dans la tuile
→ session exhaustive honnête mono-contexte mur, domaine Mode photo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from ultralytics import YOLO  # noqa: E402


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sessions",
        nargs="*",
        type=Path,
        default=[],
        help="Sessions mur source (défaut: p0-mode-photo wall*)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "training" / "inbox" / "private" / "p0-wall-tiles",
    )
    parser.add_argument("--tile", type=int, default=320)
    parser.add_argument("--stride", type=int, default=160)
    parser.add_argument("--min-walls", type=int, default=8)
    parser.add_argument("--other-conf", type=float, default=0.30)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "building-detector.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-tiles-per-image", type=int, default=6)
    parser.add_argument("--max-images", type=int, default=4)
    args = parser.parse_args()

    import torch

    device = args.device if args.device != "0" or torch.cuda.is_available() else "cpu"
    class_payload = json.loads((ROOT / "training" / "classes.json").read_text(encoding="utf-8"))
    wall_id = next(
        i for i, item in enumerate(class_payload["detector_classes"]) if item["id"] == "wall"
    )

    if args.sessions:
        sessions = [p if p.is_absolute() else ROOT / p for p in args.sessions]
    else:
        sessions = sorted(
            (ROOT / "training" / "inbox" / "private" / "p0-mode-photo").glob("wall__*.session.json")
        )[: args.max_images]

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    rows = []

    for session_path in sessions:
        session = json.loads(session_path.read_text(encoding="utf-8"))
        meta = session.get("metadata") or {}
        image_name = (session.get("image") or {}).get("name")
        image_path = session_path.with_name(image_name)
        if not image_path.is_file():
            continue
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        walls = [
            b
            for b in (session.get("boxes") or [])
            if b.get("status") == "accepted" and b.get("class_name") == "wall"
        ]
        arr = np.asarray(image)
        produced = 0
        for y0 in range(0, max(1, height - args.tile + 1), args.stride):
            for x0 in range(0, max(1, width - args.tile + 1), args.stride):
                if produced >= args.max_tiles_per_image:
                    break
                x1, y1 = x0 + args.tile, y0 + args.tile
                if x1 > width or y1 > height:
                    continue
                tile_walls = []
                for box in walls:
                    bx1, by1, bx2, by2 = (
                        float(box["x1"]),
                        float(box["y1"]),
                        float(box["x2"]),
                        float(box["y2"]),
                    )
                    # keep walls fully inside tile
                    if bx1 >= x0 and by1 >= y0 and bx2 <= x1 and by2 <= y1:
                        tile_walls.append(
                            {
                                "class_id": wall_id,
                                "class_name": "wall",
                                "x1": bx1 - x0,
                                "y1": by1 - y0,
                                "x2": bx2 - x0,
                                "y2": by2 - y0,
                                "status": "accepted",
                                "origin": "wall-tile",
                            }
                        )
                if len(tile_walls) < args.min_walls:
                    continue
                crop = arr[y0:y1, x0:x1]
                # V5: aucune autre classe dans la tuile
                result = model.predict(
                    crop,
                    imgsz=args.imgsz,
                    conf=args.other_conf,
                    device=device,
                    augment=False,
                    max_det=200,
                    verbose=False,
                )[0]
                other = []
                if result.boxes is not None:
                    for det in result.boxes:
                        name = result.names[int(det.cls.item())]
                        if name != "wall":
                            other.append(name)
                if other:
                    continue
                png = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(".jpg", png, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if not ok:
                    continue
                digest = sha256_bytes(buf.tobytes())[:12]
                stem = f"wall-tile-{meta.get('village_group', 'vg')[:40]}-{digest}"
                stem = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in stem)[:80]
                out_img = args.output / f"{stem}.jpg"
                out_img.write_bytes(buf.tobytes())
                out_session = {
                    "version": 1,
                    "image": {"name": out_img.name, "width": args.tile, "height": args.tile},
                    "metadata": {
                        "village_group": f"wall-tile-{digest}",
                        "source": f"wall-tile:{meta.get('source')}",
                        "license": meta.get("license") or "CC BY 4.0",
                        "exhaustive": True,
                        "draft": False,
                        "class_focus": "wall",
                        "private": True,
                        "p0_wall_tile": True,
                        "parent_village_group": meta.get("village_group"),
                        "tile_xyxy": [x0, y0, x1, y1],
                        "scope": "TH10-18",
                    },
                    "expected_counts": {"wall": len(tile_walls)},
                    "boxes": tile_walls,
                }
                out_json = args.output / f"{stem}.session.json"
                out_json.write_text(
                    json.dumps(out_session, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                rows.append(
                    {
                        "session": str(out_json.relative_to(ROOT)).replace("\\", "/"),
                        "image": str(out_img.relative_to(ROOT)).replace("\\", "/"),
                        "walls": len(tile_walls),
                        "parent": meta.get("village_group"),
                    }
                )
                produced += 1
            if produced >= args.max_tiles_per_image:
                break

    report = {
        "updated": date.today().isoformat(),
        "purpose": "Tuiles Mode photo mur-only exhaustives (domaine village, sans poison multi-classe)",
        "device": device,
        "tile": args.tile,
        "min_walls": args.min_walls,
        "tiles": len(rows),
        "rows": rows,
    }
    report_path = (
        ROOT
        / "training"
        / "reports"
        / f"p0-wall-modephoto-tiles-{date.today().isoformat().replace('-', '')}.json"
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "tiles": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
