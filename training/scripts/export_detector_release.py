"""Fige un détecteur versionné, exporte ONNX et vérifie la parité d'inférence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detections(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    return (
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.conf.cpu().numpy(),
        result.boxes.cls.cpu().numpy().astype(int),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=TRAINING / "runs" / "building-detector-v3-49classes" / "weights" / "best.pt")
    parser.add_argument("--release", default="building-detector-v3-49classes")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--sample", type=Path)
    args = parser.parse_args()

    model_path = args.model.resolve()
    if not model_path.exists():
        raise SystemExit(f"Modèle absent: {model_path}")
    sample = args.sample
    if sample is None:
        image_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
        candidates = [
            path
            for path in sorted((TRAINING / "dataset" / "detector" / "images" / "val").glob("*"))
            if path.suffix.lower() in image_suffixes
        ]
        if not candidates:
            raise SystemExit("Aucune image de validation disponible pour le test de parité")
        sample = candidates[0]

    release_dir = ROOT / "models" / "releases" / args.release
    release_dir.mkdir(parents=True, exist_ok=True)
    pt_target = release_dir / f"{args.release}.pt"
    shutil.copy2(model_path, pt_target)

    pt_model = YOLO(str(model_path))
    exported = Path(pt_model.export(format="onnx", imgsz=args.imgsz, device="cpu", dynamic=False, simplify=True))
    onnx_target = release_dir / f"{args.release}.onnx"
    shutil.copy2(exported, onnx_target)

    onnx_model = onnx.load(str(onnx_target))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(onnx_target), providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape

    pt_result = YOLO(str(pt_target)).predict(sample, imgsz=args.imgsz, conf=0.25, device=0, verbose=False)[0]
    onnx_result = YOLO(str(onnx_target)).predict(sample, imgsz=args.imgsz, conf=0.25, device="cpu", verbose=False)[0]
    pt_boxes, pt_conf, pt_cls = detections(pt_result)
    onnx_boxes, onnx_conf, onnx_cls = detections(onnx_result)
    same_count = len(pt_cls) == len(onnx_cls)
    same_classes = same_count and np.array_equal(pt_cls, onnx_cls)
    max_box_delta = float(np.max(np.abs(pt_boxes - onnx_boxes))) if same_count and len(pt_boxes) else 0.0
    max_conf_delta = float(np.max(np.abs(pt_conf - onnx_conf))) if same_count and len(pt_conf) else 0.0
    parity_ok = same_classes and max_box_delta <= 2.0 and max_conf_delta <= 0.02

    metadata = {
        "release": args.release,
        "source_model": str(model_path),
        "pt": {"path": str(pt_target), "sha256": sha256(pt_target), "bytes": pt_target.stat().st_size},
        "onnx": {"path": str(onnx_target), "sha256": sha256(onnx_target), "bytes": onnx_target.stat().st_size, "input_shape": input_shape},
        "imgsz": args.imgsz,
        "classes": {str(key): value for key, value in pt_model.names.items()},
        "sample": str(sample.resolve()),
        "parity": {
            "ok": parity_ok,
            "pt_detections": len(pt_cls),
            "onnx_detections": len(onnx_cls),
            "same_classes": same_classes,
            "max_box_delta_px": max_box_delta,
            "max_confidence_delta": max_conf_delta,
        },
        "coverage_warning": "town-hall-guardian n'a aucune annotation et ne doit pas être présenté comme appris.",
    }
    metadata_path = release_dir / "release.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if not parity_ok:
        raise SystemExit("Échec de parité PT/ONNX")


if __name__ == "__main__":
    main()
