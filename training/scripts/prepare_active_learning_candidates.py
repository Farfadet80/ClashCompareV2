"""Prépare des candidats d'annotation active learning (revue humaine obligatoire).

Ne crée aucune annotation YOLO automatique. Écrit des JSON + overlays pour
accélérer l'annotation Mode photo des classes faibles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from ultralytics import YOLO  # noqa: E402

PRIORITY = (
    "wall",
    "town-hall-guardian",
    "tornado-trap",
    "giga-bomb",
    "seeking-air-mine",
    "skeleton-trap",
    "air-bomb",
    "spring-trap",
    "bomb",
    "hidden-tesla",
    "workshop",
    "pet-house",
    "hero-hall",
    "crafted-defense",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def safe_output_stem(path: Path, limit: int = 72) -> str:
    stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in path.stem)
    if len(stem) <= limit:
        return stem
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{stem[: limit - 11]}-{digest}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inbox",
        type=Path,
        default=ROOT / "training" / "inbox" / "screenshots",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "training" / "curation" / "active-learning",
    )
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "building-detector.pt")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--image",
        action="append",
        type=Path,
        default=[],
        help="Capture externe à ajouter (option répétable).",
    )
    parser.add_argument("--town-hall-level", type=int)
    parser.add_argument("--local-training-consent", action="store_true")
    args = parser.parse_args()

    inbox_images = [
        path
        for path in sorted(args.inbox.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.name != "README.md"
    ]
    images = list(dict.fromkeys([*args.image, *inbox_images]))
    if not images:
        print(f"Aucune capture dans {args.inbox}")
        print("Dépose des Mode photo, puis relance ce script.")
        return

    import torch

    device = args.device if args.device != "0" or torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.model))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    sources_dir = args.output / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for image_path in images:
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise SystemExit(f"Capture invalide: {image_path}")
        image = Image.open(image_path).convert("RGB")
        output_stem = safe_output_stem(image_path)
        source_copy = sources_dir / f"{output_stem}.jpg"
        image.save(source_copy, format="JPEG", quality=95, subsampling=0)
        result = model.predict(
            image,
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
            augment=True,
            max_det=1000,
            verbose=False,
        )[0]
        detections = []
        canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        if result.boxes is not None:
            for box in result.boxes:
                name = result.names[int(box.cls.item())]
                conf = float(box.conf.item())
                left, top, right, bottom = (int(round(v)) for v in box.xyxy[0].tolist())
                detections.append(
                    {
                        "building": name,
                        "confidence": round(conf, 4),
                        "box": [left, top, right, bottom],
                        "priority": name in PRIORITY,
                    }
                )
                color = (40, 180, 255) if name in PRIORITY else (40, 220, 40)
                cv2.rectangle(canvas, (left, top), (right, bottom), color, 2)
                cv2.putText(
                    canvas,
                    f"{name} {conf:.0%}",
                    (left, max(16, top - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        out_json = args.output / f"{output_stem}.candidates.json"
        out_img = args.output / f"{output_stem}.review.jpg"
        payload = {
            "source": str(source_copy.relative_to(ROOT)),
            "original_source": str(image_path.resolve()),
            "confirmed_town_hall_level": args.town_hall_level,
            "local_training_consent": args.local_training_consent,
            "conf": args.conf,
            "note": (
                "Candidats uniquement. Annoter exhaustivement à la main avant "
                "import dataset. Ne jamais promouvoir des pseudo-labels bruts."
            ),
            "priority_hits": sum(1 for item in detections if item["priority"]),
            "detections": sorted(detections, key=lambda item: (-item["priority"], -item["confidence"])),
        }
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        cv2.imwrite(str(out_img), canvas)
        summary.append(
            {
                "image": image_path.name,
                "detections": len(detections),
                "priority_hits": payload["priority_hits"],
            }
        )

    (args.output / "summary.json").write_text(
        json.dumps({"images": summary, "priority_classes": list(PRIORITY)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{len(images)} image(s) → {args.output}")
    print("Revue humaine obligatoire avant toute annotation / import.")


if __name__ == "__main__":
    main()
