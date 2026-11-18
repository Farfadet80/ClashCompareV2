"""Entraîne un classifieur de niveaux déjà préparé."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("building")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction du jeu d'entraînement (utile pour un test rapide du pipeline).",
    )
    parser.add_argument("--name", default=None, help="Nom du run; le bâtiment par défaut.")
    args = parser.parse_args()

    data = ROOT / "training" / "classification" / args.building
    if not (data / "train").exists() or not (data / "val").exists():
        raise SystemExit(f"Dataset non préparé: {data}")

    model = YOLO("yolo11n-cls.pt")
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=0,
        patience=5,
        fraction=args.fraction,
        project=str(ROOT / "training" / "runs" / "levels"),
        name=args.name or args.building,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
