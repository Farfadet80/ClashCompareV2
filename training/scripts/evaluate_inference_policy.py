"""Compare des politiques d'inférence sur le split VAL (IoU 0.5).

Ne consulte jamais le TEST réservé. Sert à choisir conf/max_det sans inventer
de niveaux ni promouvoir un modèle.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_SMALL = (
    "hidden-tesla",
    "bomb",
    "spring-trap",
    "air-bomb",
    "giant-bomb",
    "seeking-air-mine",
    "skeleton-trap",
    "tornado-trap",
    "giga-bomb",
)


def names_list(value: list[str] | dict[int | str, str]) -> list[str]:
    if isinstance(value, list):
        return value
    return [value[key] for key in sorted(value, key=lambda item: int(item))]


def load_gt(label_path: Path, width: int, height: int) -> list[tuple[int, list[float]]]:
    boxes: list[tuple[int, list[float]]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])
        left = (xc - w / 2) * width
        top = (yc - h / 2) * height
        right = (xc + w / 2) * width
        bottom = (yc + h / 2) * height
        boxes.append((class_id, [left, top, right, bottom]))
    return boxes


def iou(a: list[float], b: list[float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def match(
    preds: list[tuple[int, float, list[float]]],
    gts: list[tuple[int, list[float]]],
    iou_thr: float,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Return tp/fp counts placeholders via matched flags per class."""
    gt_used = [False] * len(gts)
    tp: dict[int, int] = defaultdict(int)
    fp: dict[int, int] = defaultdict(int)
    fn: dict[int, int] = defaultdict(int)
    ordered = sorted(preds, key=lambda item: item[1], reverse=True)
    for class_id, _conf, box in ordered:
        best_iou = 0.0
        best_idx = -1
        for index, (gt_class, gt_box) in enumerate(gts):
            if gt_used[index] or gt_class != class_id:
                continue
            score = iou(box, gt_box)
            if score > best_iou:
                best_iou = score
                best_idx = index
        if best_idx >= 0 and best_iou >= iou_thr:
            gt_used[best_idx] = True
            tp[class_id] += 1
        else:
            fp[class_id] += 1
    for index, (gt_class, _) in enumerate(gts):
        if not gt_used[index]:
            fn[gt_class] += 1
    return tp, fp, fn  # type: ignore[return-value]


def predict_policy(
    model: YOLO,
    image: Image.Image,
    names: list[str],
    base_conf: float,
    small_conf: float,
    small_ids: set[int],
    max_det: int,
    imgsz: int,
    tta: bool,
    device: str,
) -> list[tuple[int, float, list[float]]]:
    low_conf = min(base_conf, small_conf)
    result = model.predict(
        image,
        imgsz=imgsz,
        conf=low_conf,
        device=device,
        augment=tta,
        max_det=max_det,
        verbose=False,
    )[0]
    preds: list[tuple[int, float, list[float]]] = []
    if result.boxes is None:
        return preds
    for box in result.boxes:
        class_id = int(box.cls.item())
        conf = float(box.conf.item())
        threshold = small_conf if class_id in small_ids else base_conf
        if conf < threshold:
            continue
        left, top, right, bottom = (float(value) for value in box.xyxy[0].tolist())
        preds.append((class_id, conf, [left, top, right, bottom]))
    return preds


def summarize(
    names: list[str],
    tp: dict[int, int],
    fp: dict[int, int],
    fn: dict[int, int],
    focus: set[int],
) -> dict:
    per_class = []
    for class_id, name in enumerate(names):
        t = tp.get(class_id, 0)
        fpos = fp.get(class_id, 0)
        fneg = fn.get(class_id, 0)
        precision = t / (t + fpos) if (t + fpos) else None
        recall = t / (t + fneg) if (t + fneg) else None
        per_class.append(
            {
                "class_id": class_id,
                "class_name": name,
                "tp": t,
                "fp": fpos,
                "fn": fneg,
                "precision": precision,
                "recall": recall,
            }
        )
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    focus_tp = sum(tp.get(i, 0) for i in focus)
    focus_fp = sum(fp.get(i, 0) for i in focus)
    focus_fn = sum(fn.get(i, 0) for i in focus)
    return {
        "precision": total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0,
        "recall": total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0,
        "focus_precision": focus_tp / (focus_tp + focus_fp) if (focus_tp + focus_fp) else 0.0,
        "focus_recall": focus_tp / (focus_tp + focus_fn) if (focus_tp + focus_fn) else 0.0,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "building-detector.pt")
    parser.add_argument("--split", default="val", choices=("val",))
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--name", default="inference-policy-val")
    args = parser.parse_args()

    import torch

    device = args.device
    if device == "0" and not torch.cuda.is_available():
        device = "cpu"

    config = yaml.safe_load((TRAINING / "dataset.yaml").read_text(encoding="utf-8"))
    names = names_list(config["names"])
    small_ids = {names.index(name) for name in DEFAULT_SMALL if name in names}
    image_dir = TRAINING / "dataset" / "detector" / "images" / args.split
    label_dir = TRAINING / "dataset" / "detector" / "labels" / args.split
    images = [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        raise SystemExit(f"Aucune image dans {image_dir}")

    model = YOLO(str(args.model.resolve()))
    policies = [
        {"name": "baseline-conf25", "base_conf": 0.25, "small_conf": 0.25},
        {"name": "global-conf15", "base_conf": 0.15, "small_conf": 0.15},
        {"name": "dual-traps12", "base_conf": 0.25, "small_conf": 0.12},
        {"name": "dual-traps10", "base_conf": 0.25, "small_conf": 0.10},
        {"name": "dual-traps08", "base_conf": 0.22, "small_conf": 0.08},
    ]

    results = []
    for policy in policies:
        tp: dict[int, int] = defaultdict(int)
        fp: dict[int, int] = defaultdict(int)
        fn: dict[int, int] = defaultdict(int)
        for image_path in images:
            image = Image.open(image_path).convert("RGB")
            gts = load_gt(label_dir / f"{image_path.stem}.txt", image.width, image.height)
            preds = predict_policy(
                model,
                image,
                names,
                base_conf=policy["base_conf"],
                small_conf=policy["small_conf"],
                small_ids=small_ids,
                max_det=args.max_det,
                imgsz=args.imgsz,
                tta=args.tta,
                device=device,
            )
            t, fpos, fneg = match(preds, gts, args.iou)
            for class_id, value in t.items():
                tp[class_id] += value
            for class_id, value in fpos.items():
                fp[class_id] += value
            for class_id, value in fneg.items():
                fn[class_id] += value
        summary = summarize(names, tp, fp, fn, small_ids)
        summary["policy"] = policy
        results.append(summary)
        print(
            json.dumps(
                {
                    "policy": policy["name"],
                    "precision": round(summary["precision"], 4),
                    "recall": round(summary["recall"], 4),
                    "focus_precision": round(summary["focus_precision"], 4),
                    "focus_recall": round(summary["focus_recall"], 4),
                },
                ensure_ascii=False,
            )
        )

    # Prefer higher focus recall without dropping overall precision by >1.5 pts
    # vs baseline, and without dropping focus precision by >5 pts.
    baseline = results[0]
    best = baseline
    for candidate in results[1:]:
        if candidate["precision"] + 1e-9 < baseline["precision"] - 0.015:
            continue
        if candidate["focus_precision"] + 1e-9 < baseline["focus_precision"] - 0.05:
            continue
        if candidate["focus_recall"] > best["focus_recall"] + 1e-9:
            best = candidate
        elif (
            abs(candidate["focus_recall"] - best["focus_recall"]) < 1e-9
            and candidate["precision"] > best["precision"]
        ):
            best = candidate

    payload = {
        "model": str(args.model.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "tta": args.tta,
        "max_det": args.max_det,
        "iou": args.iou,
        "small_classes": [names[i] for i in sorted(small_ids)],
        "results": [
            {
                "policy": item["policy"],
                "precision": item["precision"],
                "recall": item["recall"],
                "focus_precision": item["focus_precision"],
                "focus_recall": item["focus_recall"],
                "tp": item["tp"],
                "fp": item["fp"],
                "fn": item["fn"],
            }
            for item in results
        ],
        "selected": best["policy"],
        "decision": (
            "adopt_selected_policy"
            if best["policy"]["name"] != baseline["policy"]["name"]
            else "keep_baseline_policy"
        ),
    }
    out_dir = TRAINING / "runs" / "evaluations" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "policy.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": payload["selected"], "decision": payload["decision"], "report": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
