"""Auto-annotator Mode photo — ACTIVE FT1 → sessions inbox privée.

Usage typique:
  .\\.venv\\Scripts\\python.exe training\\scripts\\auto_annotate_villages.py `
    --input training\\inbox\\private\\drop `
    --output training\\inbox\\private\\auto-annotate

Règles dures:
- exhaustive=false toujours (jamais ready / never mark human-reviewed)
- classes YOLO 50–53 (TH19) ignorées
- murs: jamais auto-acceptés (toujours pending + flag si douteux)
- pas d'import detector ; pas de train
- images restent dans inbox/private (gitignorées)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from ultralytics import YOLO  # noqa: E402

TH19 = {
    "spiky-tower",
    "firecracker-tower",
    "guardian-shrine",
    "squad-barracks",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# Caps surface relative (évite boîtes géantes absurdes)
AREA_CAP = {
    "wall": 0.06,
    "tornado-trap": 0.18,
    "giga-bomb": 0.22,
    "bomb": 0.25,
    "spring-trap": 0.25,
    "air-bomb": 0.25,
    "giant-bomb": 0.25,
    "seeking-air-mine": 0.25,
    "skeleton-trap": 0.25,
    "hidden-tesla": 0.20,
}
DEFAULT_AREA_CAP = 0.80

# Murs: modèle encore faible → jamais auto-accept
WALL_ALWAYS_PENDING = True
WALL_MIN_CONF_KEEP = 0.18
WALL_DOUBT_CONF = 0.35
WALL_MAX_COUNT_SOFT = 350
DEFAULT_WALL_FANDOM_INDEX = (
    TRAINING / "inbox" / "private" / "reference-only" / "wall-fandom" / "index.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_stem(text: str, limit: int = 72) -> str:
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    if len(stem) <= limit:
        return stem
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{stem[: limit - 11]}-{digest}"


def load_classes() -> tuple[dict[str, int], dict[int, str]]:
    payload = json.loads((TRAINING / "classes.json").read_text(encoding="utf-8"))
    class_ids = {item["id"]: index for index, item in enumerate(payload["detector_classes"])}
    class_names = {index: item["id"] for index, item in enumerate(payload["detector_classes"])}
    return class_ids, class_names


def load_active_meta() -> dict:
    path = ROOT / "models" / "ACTIVE.json"
    if not path.exists():
        return {"release": "unknown", "imgsz": 800, "tta": True}
    return json.loads(path.read_text(encoding="utf-8"))


def load_wall_fandom_index(path: Path | None) -> dict | None:
    """Charge l'index Fandom murs (référence visuelle locale)."""
    if path is None:
        path = DEFAULT_WALL_FANDOM_INDEX
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    levels = []
    for row in payload.get("levels") or []:
        rel = row.get("path")
        if not rel:
            continue
        local = ROOT / rel
        if not local.is_file():
            continue
        levels.append(
            {
                "level": int(row["level"]),
                "path": str(Path(rel).as_posix()),
                "file": row.get("file") or local.name,
                "th19_adjacent": bool(row.get("th19_adjacent") or int(row["level"]) >= 19),
            }
        )
    if not levels:
        return None
    return {
        "index": str(path.relative_to(ROOT)).replace("\\", "/"),
        "source_page": payload.get("source_page"),
        "levels": sorted(levels, key=lambda item: item["level"]),
        "level_count": len(levels),
    }


def build_wall_ref_strip(wall_index: dict, thumb: int = 64) -> np.ndarray | None:
    """Bandeau horizontal Wall1..N pour collage sous l'overlay de revue."""
    tiles: list[np.ndarray] = []
    for row in wall_index.get("levels") or []:
        # Focus revue TH10–18 : on montre 1–18 en priorité, 19–20 en petit rappel
        level = int(row["level"])
        if level > 20:
            continue
        path = ROOT / row["path"]
        if not path.is_file():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            alpha = img[:, :, 3:4] / 255.0
            rgb = img[:, :, :3]
            img = (rgb * alpha + 255 * (1.0 - alpha)).astype(np.uint8)
        h, w = img.shape[:2]
        scale = thumb / max(h, w)
        resized = cv2.resize(
            img,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        canvas = np.full((thumb, thumb, 3), 30, dtype=np.uint8)
        y0 = (thumb - resized.shape[0]) // 2
        x0 = (thumb - resized.shape[1]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        cv2.putText(
            canvas,
            str(level),
            (2, thumb - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        tiles.append(canvas)
    if not tiles:
        return None
    return np.hstack(tiles)


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise SystemExit(f"Pas une image: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise SystemExit(f"Input introuvable: {input_path}")
    images = sorted(
        p
        for p in input_path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
        and ".review." not in p.name
    )
    return images


def area_frac(xyxy: list[float], width: int, height: int) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, width * height)


def nms_xyxy(boxes: list[dict], iou_thr: float) -> list[dict]:
    """NMS simple par classe (conf décroissante)."""
    if not boxes:
        return []
    by_class: dict[str, list[dict]] = {}
    for box in boxes:
        by_class.setdefault(box["class_name"], []).append(box)
    kept: list[dict] = []
    for rows in by_class.values():
        rows = sorted(rows, key=lambda b: float(b.get("confidence") or 0), reverse=True)
        selected: list[dict] = []
        for cand in rows:
            cx = [cand["x1"], cand["y1"], cand["x2"], cand["y2"]]
            conflict = False
            for prev in selected:
                px = [prev["x1"], prev["y1"], prev["x2"], prev["y2"]]
                ix1, iy1 = max(cx[0], px[0]), max(cx[1], px[1])
                ix2, iy2 = min(cx[2], px[2]), min(cx[3], px[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inter <= 0:
                    continue
                area_c = max(0.0, (cx[2] - cx[0]) * (cx[3] - cx[1]))
                area_p = max(0.0, (px[2] - px[0]) * (px[3] - px[1]))
                iou = inter / max(1e-9, area_c + area_p - inter)
                if iou >= iou_thr:
                    conflict = True
                    break
            if not conflict:
                selected.append(cand)
        kept.extend(selected)
    return kept


def score_village(boxes: list[dict], flags: list[str]) -> float:
    if not boxes:
        return 0.0
    confs = [float(b.get("confidence") or 0) for b in boxes]
    mean_conf = sum(confs) / len(confs)
    n_classes = len({b["class_name"] for b in boxes})
    class_bonus = min(0.25, 0.02 * n_classes)
    penalty = 0.08 * len([f for f in flags if "wall" in f or "tesla" in f or "sparse" in f])
    return round(max(0.0, min(1.0, 0.55 * mean_conf + 0.20 + class_bonus - penalty)), 4)


def annotate_one(
    model: YOLO,
    image_path: Path,
    out_dir: Path,
    class_ids: dict[str, int],
    class_names: dict[int, str],
    active: dict,
    conf: float,
    auto_accept_conf: float,
    imgsz: int,
    device: str,
    tta: bool,
    iou_nms: float,
    write_overlay: bool,
    wall_fandom_index: dict | None = None,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    digest = sha256_file(image_path)
    stem = safe_stem(f"auto__{image_path.stem}")
    dest_img = out_dir / f"{stem}{image_path.suffix.lower()}"
    dest_session = out_dir / f"{stem}.session.json"
    dest_overlay = out_dir / f"{stem}.review.jpg"
    dest_prov = out_dir / f"{stem}.provenance.json"

    if dest_session.exists():
        return {
            "ok": True,
            "skipped_existing": True,
            "session": str(dest_session.relative_to(ROOT)).replace("\\", "/"),
            "image_src": str(image_path),
        }

    if not dest_img.exists() or dest_img.stat().st_size != image_path.stat().st_size:
        shutil.copy2(image_path, dest_img)

    result = model.predict(
        image,
        imgsz=imgsz,
        conf=conf,
        device=device,
        augment=tta,
        max_det=1000,
        verbose=False,
    )[0]

    raw: list[dict] = []
    skipped_th19 = 0
    skipped_area = 0
    skipped_wall_low = 0
    if result.boxes is not None:
        for det in result.boxes:
            name = class_names[int(det.cls.item())]
            if name in TH19:
                skipped_th19 += 1
                continue
            if name not in class_ids:
                continue
            conf_v = float(det.conf.item())
            x1, y1, x2, y2 = (float(v) for v in det.xyxy[0].tolist())
            x1 = max(0.0, min(float(width), x1))
            y1 = max(0.0, min(float(height), y1))
            x2 = max(0.0, min(float(width), x2))
            y2 = max(0.0, min(float(height), y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            xyxy = [x1, y1, x2, y2]
            frac = area_frac(xyxy, width, height)
            if frac > AREA_CAP.get(name, DEFAULT_AREA_CAP):
                skipped_area += 1
                continue
            if name == "wall" and conf_v < WALL_MIN_CONF_KEEP:
                skipped_wall_low += 1
                continue
            raw.append(
                {
                    "class_id": class_ids[name],
                    "class_name": name,
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "confidence": round(conf_v, 4),
                    "origin": "auto-annotate-active",
                }
            )

    filtered = nms_xyxy(raw, iou_thr=iou_nms)
    flags: list[str] = []
    boxes: list[dict] = []
    wall_confs: list[float] = []

    for box in filtered:
        name = box["class_name"]
        conf_v = float(box["confidence"])
        status = "pending"
        note = f"auto conf={conf_v:.3f}"
        if name == "wall":
            wall_confs.append(conf_v)
            status = "pending" if WALL_ALWAYS_PENDING else (
                "accepted" if conf_v >= auto_accept_conf else "pending"
            )
            if conf_v < WALL_DOUBT_CONF:
                note += " | wall_douteux"
                box["quality_flag"] = "wall_douteux"
            else:
                box["quality_flag"] = "wall_pending_revue"
            note += " | mur jamais auto-accepté"
            if wall_fandom_index:
                note += " | comparer refs Fandom Wall1..N"
        elif conf_v >= auto_accept_conf:
            status = "accepted"
            note += f" | auto-accept>={auto_accept_conf}"
        else:
            note += " | pending revue"
        box["status"] = status
        box["review_note"] = note
        boxes.append(box)

    wall_count = sum(1 for b in boxes if b["class_name"] == "wall")
    if wall_count == 0:
        flags.append("walls_missing_suspect")
    if wall_count > WALL_MAX_COUNT_SOFT:
        flags.append("walls_count_high")
    if wall_confs and (sum(wall_confs) / len(wall_confs)) < WALL_DOUBT_CONF:
        flags.append("walls_low_mean_conf")
    n_classes = len({b["class_name"] for b in boxes})
    if n_classes < 4:
        flags.append("sparse_classes")
    if any(b["class_name"] == "hidden-tesla" and float(b["confidence"]) < 0.35 for b in boxes):
        flags.append("tesla_low_conf_present")

    village_score = score_village(boxes, flags)
    accepted = sum(1 for b in boxes if b["status"] == "accepted")
    pending = sum(1 for b in boxes if b["status"] == "pending")

    metadata = {
        "village_group": f"auto-{digest[:12]}",
        "source": "auto-annotate-active",
        "license": "consentement local — dataset privé / CC BY local copy",
        "exhaustive": False,
        "draft": True,
        "private": True,
        "scope": "TH10-18",
        "auto_annotate": True,
        "auto_annotate_model": active.get("release") or "building-detector.pt",
        "auto_annotate_on": date.today().isoformat(),
        "village_confidence": village_score,
        "quality_flags": flags,
        "class_focus": "wall" if wall_count else "multi",
        "notes": [
            "Pré-labels automatiques — revue humaine obligatoire avant exhaustive.",
            "Ne pas importer wall-tiles / mur-only poison dans detector.",
        ],
    }
    if wall_fandom_index:
        # Réfs 1–18 pour TH10–18 ; 19–20 marquées adjacent (hors focus train actuel)
        refs = [
            {
                "level": row["level"],
                "path": row["path"],
                "th19_adjacent": row.get("th19_adjacent", False),
            }
            for row in wall_fandom_index["levels"]
            if int(row["level"]) <= 20
        ]
        metadata["wall_visual_refs"] = {
            "index": wall_fandom_index.get("index"),
            "source_page": wall_fandom_index.get("source_page"),
            "levels": refs,
            "usage": "reference_only_visual",
        }
        metadata["notes"].append(
            "Référence visuelle Fandom Wall/Home_Village branchée (hors train)."
        )
        if "walls_missing_suspect" in flags or "walls_low_mean_conf" in flags:
            flags.append("consult_wall_fandom_refs")
            metadata["quality_flags"] = flags
    session = {
        "version": 1,
        "image": {"name": dest_img.name, "width": width, "height": height},
        "metadata": metadata,
        "expected_counts": {},
        "boxes": boxes,
    }
    dest_session.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "sha256": digest,
        "source_image": str(image_path.resolve()),
        "publish": False,
        "usage": "private_auto_annotate_queue",
        "model": active.get("release"),
        "imgsz": imgsz,
        "tta": tta,
        "conf": conf,
        "auto_accept_conf": auto_accept_conf,
        "village_confidence": village_score,
        "quality_flags": flags,
        "wall_fandom_index": (wall_fandom_index or {}).get("index"),
    }
    dest_prov.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if write_overlay:
        canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        for box in boxes:
            if box["class_name"] == "wall":
                color = (0, 140, 255)  # orange murs pending
            elif box["status"] == "pending":
                color = (0, 165, 255)
            else:
                color = (40, 220, 40)
            p1 = (int(box["x1"]), int(box["y1"]))
            p2 = (int(box["x2"]), int(box["y2"]))
            cv2.rectangle(canvas, p1, p2, color, 1)
            label = f"{box['class_name'][:14]} {box['confidence']:.2f}"
            cv2.putText(
                canvas,
                label,
                (p1[0], max(12, p1[1] - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
        if wall_fandom_index:
            strip = build_wall_ref_strip(wall_fandom_index, thumb=48)
            if strip is not None:
                # Adapter largeur au canvas
                if strip.shape[1] != canvas.shape[1]:
                    scale = canvas.shape[1] / strip.shape[1]
                    strip = cv2.resize(
                        strip,
                        (canvas.shape[1], max(24, int(strip.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                canvas = np.vstack([canvas, strip])
        cv2.imwrite(str(dest_overlay), canvas)

    return {
        "ok": True,
        "session": str(dest_session.relative_to(ROOT)).replace("\\", "/"),
        "image": str(dest_img.relative_to(ROOT)).replace("\\", "/"),
        "overlay": str(dest_overlay.relative_to(ROOT)).replace("\\", "/") if write_overlay else None,
        "village_confidence": village_score,
        "quality_flags": flags,
        "boxes": len(boxes),
        "accepted": accepted,
        "pending": pending,
        "wall_count": wall_count,
        "n_classes": n_classes,
        "skipped_th19": skipped_th19,
        "skipped_area": skipped_area,
        "skipped_wall_low": skipped_wall_low,
        "wh": [width, height],
        "source_image": str(image_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-annotator Mode photo (ACTIVE) → sessions inbox privée"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Image ou dossier d'images (ex. training/inbox/private/drop)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TRAINING / "inbox" / "private" / "auto-annotate",
        help="Dossier sessions (défaut: inbox/private/auto-annotate)",
    )
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "building-detector.pt")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument(
        "--auto-accept-conf",
        type=float,
        default=0.50,
        help="Seuil auto-accept (murs exclus: toujours pending)",
    )
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--iou-nms", type=float, default=0.45)
    parser.add_argument("--max-images", type=int, default=0, help="0 = toutes")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument(
        "--wall-fandom-index",
        type=Path,
        default=DEFAULT_WALL_FANDOM_INDEX,
        help="Index JSON murs Fandom (réf visuelle). Défaut: reference-only/wall-fandom/index.json",
    )
    parser.add_argument(
        "--no-wall-fandom",
        action="store_true",
        help="Désactive l'attache des références Fandom murs",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Rapport JSON (défaut training/reports/auto-annotate-YYYYMMDD.json)",
    )
    args = parser.parse_args()

    import torch

    device = args.device if args.device != "0" or torch.cuda.is_available() else "cpu"
    tta = False if args.no_tta else True
    active = load_active_meta()
    class_ids, class_names = load_classes()
    wall_fandom_index = None if args.no_wall_fandom else load_wall_fandom_index(args.wall_fandom_index)

    input_path = args.input if args.input.is_absolute() else (ROOT / args.input)
    out_dir = args.output if args.output.is_absolute() else (ROOT / args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".gitkeep").touch(exist_ok=True)

    images = collect_images(input_path.resolve())
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise SystemExit(f"Aucune image dans {input_path}")

    model = YOLO(str(args.model if args.model.is_absolute() else ROOT / args.model))
    rows = []
    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path.name}")
        rows.append(
            annotate_one(
                model,
                image_path,
                out_dir,
                class_ids,
                class_names,
                active,
                conf=args.conf,
                auto_accept_conf=args.auto_accept_conf,
                imgsz=args.imgsz,
                device=device,
                tta=tta,
                iou_nms=args.iou_nms,
                write_overlay=not args.no_overlay,
                wall_fandom_index=wall_fandom_index,
            )
        )

    ok_rows = [r for r in rows if r.get("ok") and not r.get("skipped_existing")]
    skipped = [r for r in rows if r.get("skipped_existing")]
    flag_counter = Counter()
    for row in ok_rows:
        for flag in row.get("quality_flags") or []:
            flag_counter[flag] += 1

    report = {
        "updated": date.today().isoformat(),
        "purpose": "Auto-annotator Mode photo ACTIVE → file revue (exhaustive=false)",
        "train_now": False,
        "publish_images": False,
        "active_release": active.get("release"),
        "model": str(args.model).replace("\\", "/"),
        "input": str(input_path).replace("\\", "/"),
        "output": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "device": device,
        "imgsz": args.imgsz,
        "tta": tta,
        "conf": args.conf,
        "auto_accept_conf": args.auto_accept_conf,
        "wall_policy": "always_pending",
        "wall_fandom_index": (wall_fandom_index or {}).get("index"),
        "wall_fandom_levels": (wall_fandom_index or {}).get("level_count", 0),
        "images_seen": len(images),
        "annotated": len(ok_rows),
        "skipped_existing": len(skipped),
        "mean_village_confidence": round(
            sum(float(r.get("village_confidence") or 0) for r in ok_rows) / max(1, len(ok_rows)),
            4,
        ),
        "quality_flag_counts": dict(flag_counter),
        "totals": {
            "boxes": sum(int(r.get("boxes") or 0) for r in ok_rows),
            "accepted": sum(int(r.get("accepted") or 0) for r in ok_rows),
            "pending": sum(int(r.get("pending") or 0) for r in ok_rows),
            "walls": sum(int(r.get("wall_count") or 0) for r in ok_rows),
        },
        "blocked": [
            "exhaustive=true sans revue humaine",
            "import detector wall-tiles / mur-only poison",
            "fine-tune sans OK Jimmy",
        ],
        "rows": rows,
    }
    report_path = args.report or (
        TRAINING / "reports" / f"auto-annotate-{date.today().isoformat().replace('-', '')}.json"
    )
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "annotated": report["annotated"],
                "skipped_existing": report["skipped_existing"],
                "mean_village_confidence": report["mean_village_confidence"],
                "output": report["output"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
