"""Analyse une capture avec le détecteur puis les classifieurs de niveaux disponibles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from ultralytics import YOLO

_detector_cache: dict[str, YOLO] = {}
_classifier_cache: dict[str, YOLO] = {}

# Les classifieurs disponibles sont fermés sur des plages anciennes
# (air-defense 8–11, town-hall 10–14). Sur un TH16+ ils donnent un niveau
# faux mais très confiant : aucun n'est donc autorisé en production.
EXPERIMENTAL_LEVEL_CLASSIFIERS = frozenset({"air-defense", "town-hall"})
PRODUCTION_LEVEL_CLASSIFIERS: frozenset[str] = frozenset()

# Classes petites / pièges : seuil optionnel (--small-conf). Bake-off VAL
# 2026-09-05 : baisser ce seuil vs conf détruit la précision → défaut = conf.
SMALL_OBJECT_CLASSES = frozenset(
    {
        "hidden-tesla",
        "bomb",
        "spring-trap",
        "air-bomb",
        "giant-bomb",
        "seeking-air-mine",
        "skeleton-trap",
        "tornado-trap",
        "giga-bomb",
    }
)
DEFAULT_BASE_CONF = 0.25
# Bake-off VAL 2026-09-05 (evaluate_inference_policy) : baisser small_conf
# remonte un peu le rappel pièges mais détruit la précision focus → défaut = conf.
DEFAULT_SMALL_CONF = 0.25
DEFAULT_MAX_DET = 1000


def resolve_device(device: str) -> str:
    if device != "0":
        return device
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def load_detector(path: Path) -> YOLO:
    key = str(path.resolve())
    if key not in _detector_cache:
        _detector_cache[key] = YOLO(key)
    return _detector_cache[key]


def load_classifier(path: Path) -> YOLO:
    key = str(path.resolve())
    if key not in _classifier_cache:
        _classifier_cache[key] = YOLO(key)
    return _classifier_cache[key]


def inventory_from_detections(detections: list[dict]) -> dict[str, dict]:
    inventory: dict[str, dict] = {}
    for item in detections:
        building = item["building"]
        slot = inventory.setdefault(building, {"total": 0, "levels": {}})
        slot["total"] += 1
        level_key = str(item["level"]) if item["level"] is not None else "inconnu"
        slot["levels"][level_key] = slot["levels"].get(level_key, 0) + 1
    return dict(sorted(inventory.items(), key=lambda pair: (-pair[1]["total"], pair[0])))


def safe_output_stem(source: Path, limit: int = 72) -> str:
    """Évite les chemins Windows trop longs pour les captures jointes Cursor."""
    stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in source.stem)
    if len(stem) <= limit:
        return stem
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{stem[: limit - 11]}-{digest}"


def analyze_image(
    source: Path,
    detector_path: Path | None = None,
    conf: float = DEFAULT_BASE_CONF,
    small_conf: float = DEFAULT_SMALL_CONF,
    max_det: int = DEFAULT_MAX_DET,
    level_conf: float = 0.60,
    imgsz: int = 800,
    tta: bool = True,
    device: str = "0",
    output_dir: Path | None = None,
) -> dict:
    detector_path = detector_path or (ROOT / "models" / "building-detector.pt")
    if not source.exists():
        raise FileNotFoundError(f"Image absente: {source}")
    if not detector_path.exists():
        raise FileNotFoundError(f"Détecteur absent: {detector_path}")
    if not 0.0 <= level_conf <= 1.0:
        raise ValueError("--level-conf doit être compris entre 0 et 1")
    if not 0.0 <= conf <= 1.0 or not 0.0 <= small_conf <= 1.0:
        raise ValueError("--conf et --small-conf doivent être compris entre 0 et 1")
    if max_det < 1:
        raise ValueError("--max-det doit être >= 1")

    device = resolve_device(device)
    image = Image.open(source).convert("RGB")
    detector = load_detector(detector_path)
    predict_conf = min(conf, small_conf)
    result = detector.predict(
        image,
        imgsz=imgsz,
        conf=predict_conf,
        device=device,
        augment=tta,
        max_det=max_det,
        verbose=False,
    )[0]

    detections: list[dict] = []
    crops_by_building: dict[str, list[tuple[int, Image.Image]]] = defaultdict(list)
    if result.boxes is not None:
        for box in result.boxes:
            building = result.names[int(box.cls.item())]
            confidence = float(box.conf.item())
            threshold = small_conf if building in SMALL_OBJECT_CLASSES else conf
            if confidence < threshold:
                continue
            left, top, right, bottom = (int(round(value)) for value in box.xyxy[0].tolist())
            left, top = max(0, left), max(0, top)
            right, bottom = min(image.width, right), min(image.height, bottom)
            item = {
                "building": building,
                "level": None,
                "box": [left, top, right, bottom],
                "confidence": round(confidence, 5),
                "level_confidence": None,
            }
            index = len(detections)
            detections.append(item)
            if right > left and bottom > top:
                crops_by_building[building].append((index, image.crop((left, top, right, bottom))))

    for building, indexed_crops in crops_by_building.items():
        if building not in PRODUCTION_LEVEL_CLASSIFIERS:
            continue
        classifier_path = ROOT / "models" / f"level-{building}.pt"
        if not classifier_path.exists():
            continue
        classifier = load_classifier(classifier_path)
        predictions = classifier.predict(
            [np.asarray(crop) for _, crop in indexed_crops],
            imgsz=224,
            batch=32,
            device=device,
            verbose=False,
        )
        for (index, _), prediction in zip(indexed_crops, predictions, strict=True):
            top1 = int(prediction.probs.top1)
            name = classifier.names[top1]
            confidence = float(prediction.probs.top1conf)
            detections[index]["level_confidence"] = round(confidence, 5)
            if confidence >= level_conf:
                detections[index]["level"] = int(name.removeprefix("level-"))

    payload = {
        "source": str(source.resolve()),
        "engine": "building-detector-v5s-infer800-tta" if tta else "building-detector-v5s-infer800",
        "imgsz": imgsz,
        "tta": bool(tta),
        "conf": conf,
        "small_conf": small_conf,
        "max_det": max_det,
        "small_object_classes": sorted(SMALL_OBJECT_CLASSES),
        "device": str(device),
        "count": len(detections),
        "detections": detections,
        "inventory": inventory_from_detections(detections),
        "coverage_note": (
            "YOLO ne remplace pas l'export JSON pour un adversaire sans capture. "
            "Murs et town-hall-guardian restent très sous-annotés : préférer l'export "
            "JSON quand disponible. Types absents → Non détecté, jamais inventés."
        ),
        "level_note": (
            "Niveaux YOLO désactivés : les classifieurs disponibles "
            f"({', '.join(sorted(EXPERIMENTAL_LEVEL_CLASSIFIERS))}) ne couvrent pas "
            "les niveaux récents et peuvent être confiants hors distribution. "
            "Utiliser l'export JSON pour les niveaux vérifiés."
        ),
        "level_classifiers": sorted(PRODUCTION_LEVEL_CLASSIFIERS),
        "experimental_level_classifiers": sorted(EXPERIMENTAL_LEVEL_CLASSIFIERS),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_stem = safe_output_stem(source)
        json_path = output_dir / f"{output_stem}.json"
        image_path = output_dir / f"{output_stem}-annotated.jpg"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        for item in detections:
            left, top, right, bottom = item["box"]
            level = f" niv.{item['level']}" if item["level"] is not None else ""
            label = f"{item['building']}{level} {item['confidence']:.0%}"
            cv2.rectangle(canvas, (left, top), (right, bottom), (40, 220, 40), 2)
            cv2.putText(
                canvas,
                label,
                (left, max(16, top - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (40, 220, 40),
                1,
                cv2.LINE_AA,
            )
        if not cv2.imwrite(str(image_path), canvas):
            raise OSError(f"Écriture image annotée impossible: {image_path}")
        payload["json_path"] = str(json_path)
        payload["annotated_path"] = str(image_path)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Capture de village à analyser")
    parser.add_argument("--detector", type=Path, default=ROOT / "models" / "building-detector.pt")
    parser.add_argument("--conf", type=float, default=DEFAULT_BASE_CONF)
    parser.add_argument(
        "--small-conf",
        type=float,
        default=DEFAULT_SMALL_CONF,
        help="Seuil plus bas pour pièges / Tesla / giga-bomb",
    )
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    parser.add_argument("--level-conf", type=float, default=0.60, help="Confiance minimale avant d'afficher un niveau")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="TTA Ultralytics (augment=True). Promue 2026-09-04 : +mAP50/mAP50-95 sur le test réservé.",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "training" / "runs" / "inference")
    args = parser.parse_args()

    payload = analyze_image(
        args.source,
        detector_path=args.detector,
        conf=args.conf,
        small_conf=args.small_conf,
        max_det=args.max_det,
        level_conf=args.level_conf,
        imgsz=args.imgsz,
        tta=args.tta,
        device=args.device,
        output_dir=args.output_dir,
    )
    print(f"{payload['count']} bâtiments détectés")
    print(f"JSON: {payload.get('json_path')}")
    print(f"Image: {payload.get('annotated_path')}")


if __name__ == "__main__":
    main()
