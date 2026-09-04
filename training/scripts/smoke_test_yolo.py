"""Entraîne YOLO une époque sur un petit dataset synthétique temporaire."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from ultralytics import YOLO


def make_image(path: Path, offset: int) -> None:
    image = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((32 + offset, 32, 95 + offset, 95), fill="white")
    image.save(path)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="clashcompare-yolo-") as temp:
        root = Path(temp)
        for split, count in (("train", 2), ("val", 1)):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            for index in range(count):
                make_image(root / "images" / split / f"sample_{index}.png", index * 2)
                (root / "labels" / split / f"sample_{index}.txt").write_text(
                    "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
                )
        dataset = root / "dataset.yaml"
        dataset.write_text(
            yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val", "names": {0: "object"}}),
            encoding="utf-8",
        )
        model = YOLO("yolo11n.yaml")
        model.train(
            data=str(dataset), epochs=1, imgsz=128, batch=2, device=0,
            workers=0, plots=False, project=str(root / "runs"), name="smoke", verbose=False,
        )
        torch.cuda.synchronize()
        print("Smoke test entraînement YOLO CUDA: OK")


if __name__ == "__main__":
    main()
