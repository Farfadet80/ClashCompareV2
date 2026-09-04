"""Évalue un détecteur YOLO et enregistre des métriques globales et par classe."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from ultralytics import YOLO


def resolved_config(source: Path, destination: Path) -> dict:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["path"] = str((source.parent / config["path"]).resolve())
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "building-detector.pt")
    parser.add_argument("--data", type=Path, default=TRAINING / "dataset.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="TTA Ultralytics (augment=True) pendant model.val",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    model_path = args.model.resolve()
    data_path = args.data.resolve()
    if not model_path.exists():
        raise SystemExit(f"Modèle absent: {model_path}")
    if not data_path.exists():
        raise SystemExit(f"Configuration absente: {data_path}")

    run_name = args.name or f"{model_path.stem}-{args.split}"
    output_root = TRAINING / "runs" / "evaluations"
    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="clashcompare-eval-") as temp:
        runtime_yaml = Path(temp) / "dataset.yaml"
        config = resolved_config(data_path, runtime_yaml)
        split_dir = Path(config["path"]) / config[args.split]
        if not split_dir.exists() or not any(split_dir.iterdir()):
            raise SystemExit(f"Split {args.split} absent ou vide: {split_dir}")

        model = YOLO(str(model_path))
        metrics = model.val(
            data=str(runtime_yaml),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=0,
            plots=True,
            augment=args.augment,
            project=str(output_root),
            name=run_name,
            exist_ok=True,
            verbose=False,
        )

    class_indexes = [int(value) for value in metrics.box.ap_class_index.tolist()]
    per_class = []
    for position, class_id in enumerate(class_indexes):
        per_class.append(
            {
                "class_id": class_id,
                "class_name": model.names[class_id],
                "precision": float(metrics.box.p[position]),
                "recall": float(metrics.box.r[position]),
                "map50": float(metrics.box.ap50[position]),
                "map50_95": float(metrics.box.maps[class_id]),
            }
        )

    payload = {
        "model": str(model_path),
        "data": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "augment_tta": bool(args.augment),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "speed_ms": {key: float(value) for key, value in metrics.speed.items()},
        "per_class": per_class,
    }
    output = Path(metrics.save_dir) / "metrics.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "per_class"}, ensure_ascii=False, indent=2))
    print(f"Métriques par classe: {output}")


if __name__ == "__main__":
    main()
