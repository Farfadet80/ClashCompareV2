"""Évalue YOLO sur images entières puis tuiles, avec fusion NMS classe par classe.

La métrique est celle d'Ultralytics : AP interpolée aux IoU 0.50:0.95 et
précision/rappel au seuil de confiance qui maximise le F1. Une passe globale
est aussi évaluée seule afin de contrôler la comparabilité avec ``model.val``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, UnidentifiedImageError
from torchvision.ops import batched_nms

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics, box_iou

IOU_THRESHOLDS = torch.linspace(0.50, 0.95, 10)
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def axis_tiles(length: int, tile_size: int, overlap: float) -> list[tuple[int, int, float, float]]:
    """Retourne (début, fin, début/fin de zone dont le centre est propriétaire)."""
    if length <= tile_size:
        return [(0, length, 0.0, float(length))]
    stride = max(1, round(tile_size * (1.0 - overlap)))
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    result = []
    for index, start in enumerate(starts):
        end = min(start + tile_size, length)
        owner_start = 0.0 if index == 0 else (starts[index - 1] + tile_size + start) / 2.0
        owner_end = float(length) if index == len(starts) - 1 else (end + starts[index + 1]) / 2.0
        result.append((start, end, owner_start, owner_end))
    return result


def result_tensor(result) -> torch.Tensor:
    """Convertit un résultat Ultralytics en colonnes x1,y1,x2,y2,conf,classe."""
    if result.boxes is None or len(result.boxes) == 0:
        return torch.empty((0, 6), dtype=torch.float32)
    return torch.cat(
        (
            result.boxes.xyxy.detach().cpu().float(),
            result.boxes.conf.detach().cpu().float().unsqueeze(1),
            result.boxes.cls.detach().cpu().float().unsqueeze(1),
        ),
        dim=1,
    )


def merge_predictions(predictions: list[torch.Tensor], iou: float, max_det: int) -> torch.Tensor:
    nonempty = [prediction for prediction in predictions if prediction.numel()]
    if not nonempty:
        return torch.empty((0, 6), dtype=torch.float32)
    merged = torch.cat(nonempty, dim=0)
    keep = batched_nms(merged[:, :4], merged[:, 4], merged[:, 5], iou)
    return merged[keep[:max_det]]


def load_targets(label_path: Path, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    if label_path.exists():
        rows = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return torch.empty(0, dtype=torch.float32), torch.empty((0, 4), dtype=torch.float32)
    labels = torch.tensor([[float(value) for value in row[:5]] for row in rows], dtype=torch.float32)
    classes = labels[:, 0]
    xywh = labels[:, 1:5]
    boxes = torch.empty_like(xywh)
    boxes[:, 0] = (xywh[:, 0] - xywh[:, 2] / 2.0) * width
    boxes[:, 1] = (xywh[:, 1] - xywh[:, 3] / 2.0) * height
    boxes[:, 2] = (xywh[:, 0] + xywh[:, 2] / 2.0) * width
    boxes[:, 3] = (xywh[:, 1] + xywh[:, 3] / 2.0) * height
    return classes, boxes


def match_predictions(pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor) -> np.ndarray:
    """Même appariement glouton qu'Ultralytics BaseValidator.match_predictions."""
    correct = np.zeros((pred_classes.shape[0], IOU_THRESHOLDS.shape[0]), dtype=bool)
    if pred_classes.numel() == 0 or true_classes.numel() == 0:
        return correct
    class_iou = (iou * (true_classes[:, None] == pred_classes)).numpy()
    for index, threshold in enumerate(IOU_THRESHOLDS.tolist()):
        matches = np.array(np.nonzero(class_iou >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[class_iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), index] = True
    return correct


def update_metrics(
    metrics: DetMetrics,
    prediction: torch.Tensor,
    target_classes: torch.Tensor,
    target_boxes: torch.Tensor,
    image_name: str,
) -> None:
    pred_classes = prediction[:, 5] if prediction.numel() else torch.empty(0)
    iou = box_iou(target_boxes, prediction[:, :4]) if prediction.numel() else torch.empty((len(target_boxes), 0))
    metrics.update_stats(
        {
            "tp": match_predictions(pred_classes, target_classes, iou),
            "conf": prediction[:, 4].numpy() if prediction.numel() else np.zeros(0),
            "pred_cls": pred_classes.numpy() if prediction.numel() else np.zeros(0),
            "target_cls": target_classes.numpy(),
            "target_img": np.unique(target_classes.numpy()),
            "im_name": image_name,
        }
    )


def metrics_payload(metrics: DetMetrics, names: dict[int, str]) -> dict:
    metrics.process(plot=False)
    class_indexes = [int(value) for value in metrics.box.ap_class_index]
    per_class = []
    for position, class_id in enumerate(class_indexes):
        per_class.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "instances": int(metrics.nt_per_class[class_id]),
                "precision": float(metrics.box.p[position]),
                "recall": float(metrics.box.r[position]),
                "map50": float(metrics.box.ap50[position]),
                "map50_95": float(metrics.box.maps[class_id]),
            }
        )
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "building-detector.pt")
    parser.add_argument("--data", type=Path, default=TRAINING / "dataset.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true", help="Garde-fou obligatoire pour lire le test réservé")
    parser.add_argument("--imgsz", type=int, default=640, help="Taille d'inférence globale et par tuile")
    parser.add_argument("--tile-size", type=int, default=640, help="Côté maximal d'une tuile en pixels source")
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--merge-iou", type=float, default=0.55)
    parser.add_argument("--conf", type=float, default=0.001, help="0.001 requis pour une AP comparable à model.val")
    parser.add_argument("--predict-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    if args.split == "test" and not args.allow_test:
        raise SystemExit("Test réservé bloqué : ajouter --allow-test seulement après décision explicite sur VAL.")
    if not 0.0 <= args.overlap < 1.0:
        raise SystemExit("--overlap doit être dans [0, 1[")
    if args.conf > 0.001:
        raise SystemExit("--conf > 0.001 rendrait le calcul d'AP non comparable à la validation Ultralytics.")

    config = yaml.safe_load(args.data.resolve().read_text(encoding="utf-8"))
    dataset_root = (args.data.resolve().parent / config["path"]).resolve()
    image_dir = dataset_root / config[args.split]
    label_dir = dataset_root / str(config[args.split]).replace("images", "labels", 1)
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise SystemExit(f"Aucune image dans {image_dir}")

    raw_names = config["names"]
    names = {int(key): value for key, value in raw_names.items()} if isinstance(raw_names, dict) else dict(enumerate(raw_names))
    model = YOLO(str(args.model.resolve()))
    global_metrics = DetMetrics(names=names)
    tiled_metrics = DetMetrics(names=names)
    global_seconds = 0.0
    tiled_seconds = 0.0
    tile_count = 0

    for image_index, image_path in enumerate(image_paths, start=1):
        try:
            image = Image.open(image_path).convert("RGB")
        except UnidentifiedImageError as exc:
            raise RuntimeError(f"Image illisible: {image_path}") from exc
        width, height = image.size
        target_classes, target_boxes = load_targets(label_dir / f"{image_path.stem}.txt", width, height)

        started = time.perf_counter()
        global_result = model.predict(
            image,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.predict_iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )[0]
        global_seconds += time.perf_counter() - started
        global_prediction = result_tensor(global_result)
        update_metrics(global_metrics, global_prediction, target_classes, target_boxes, image_path.name)

        x_tiles = axis_tiles(width, args.tile_size, args.overlap)
        y_tiles = axis_tiles(height, args.tile_size, args.overlap)
        windows = [(x, y) for y in y_tiles for x in x_tiles]
        shifted_predictions = [global_prediction]
        if len(windows) > 1:
            crops = [image.crop((x[0], y[0], x[1], y[1])) for x, y in windows]
            started = time.perf_counter()
            tile_results = model.predict(
                crops,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.predict_iou,
                max_det=args.max_det,
                batch=1,
                device=args.device,
                verbose=False,
            )
            tiled_seconds += time.perf_counter() - started
            tile_count += len(crops)
            for result, (x_tile, y_tile) in zip(tile_results, windows, strict=True):
                prediction = result_tensor(result)
                if not prediction.numel():
                    continue
                x0, _, owner_x0, owner_x1 = x_tile
                y0, _, owner_y0, owner_y1 = y_tile
                prediction[:, [0, 2]] += x0
                prediction[:, [1, 3]] += y0
                centers_x = (prediction[:, 0] + prediction[:, 2]) / 2.0
                centers_y = (prediction[:, 1] + prediction[:, 3]) / 2.0
                owned = (
                    (centers_x >= owner_x0)
                    & (centers_x <= owner_x1)
                    & (centers_y >= owner_y0)
                    & (centers_y <= owner_y1)
                )
                shifted_predictions.append(prediction[owned])

        merged = merge_predictions(shifted_predictions, args.merge_iou, args.max_det)
        update_metrics(tiled_metrics, merged, target_classes, target_boxes, image_path.name)
        print(f"[{image_index:03d}/{len(image_paths)}] {image_path.name}: {len(global_prediction)} -> {len(merged)}", flush=True)

    global_payload = metrics_payload(global_metrics, names)
    tiled_payload = metrics_payload(tiled_metrics, names)
    output_root = TRAINING / "runs" / "evaluations"
    run_name = args.name or f"{args.model.stem}-{args.split}-tiles{args.tile_size}-o{args.overlap:g}"
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": (
            "Passe globale + tuiles en pixels source; attribution des prédictions par zone de centre; "
            "fusion torchvision batched_nms classe par classe; métriques Ultralytics AP IoU 0.50:0.95."
        ),
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "images": len(image_paths),
        "config": {
            "imgsz": args.imgsz,
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "merge_iou": args.merge_iou,
            "conf": args.conf,
            "predict_iou": args.predict_iou,
            "max_det": args.max_det,
            "device": args.device,
        },
        "runtime": {
            "global_total_seconds": global_seconds,
            "tiles_total_seconds": tiled_seconds,
            "total_seconds": global_seconds + tiled_seconds,
            "tile_count": tile_count,
            "milliseconds_per_image": 1000.0 * (global_seconds + tiled_seconds) / len(image_paths),
        },
        "global_only_control": global_payload,
        "global_plus_tiles": tiled_payload,
        "delta_tiles_minus_control": {
            key: tiled_payload[key] - global_payload[key]
            for key in ("precision", "recall", "map50", "map50_95")
        },
    }
    output_path = output_dir / "metrics.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    concise = {key: value for key, value in payload.items() if key not in ("global_only_control", "global_plus_tiles")}
    concise["global_only_control"] = {key: global_payload[key] for key in ("precision", "recall", "map50", "map50_95")}
    concise["global_plus_tiles"] = {key: tiled_payload[key] for key in ("precision", "recall", "map50", "map50_95")}
    print(json.dumps(concise, ensure_ascii=False, indent=2))
    print(f"Métriques complètes: {output_path}")


if __name__ == "__main__":
    main()
