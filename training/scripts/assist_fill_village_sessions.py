"""Assist V5 pour remplir des villages Mode photo (P0 pré-FT).

Copie des sessions focus (wall/tornado/giga/crafted) vers
training/inbox/private/p0-mode-photo/, fusionne les boîtes focus acceptées
avec des propositions YOLO V5 (pending ou auto-accept haute conf), écrit
overlays de revue. Ne marque JAMAIS exhaustive=True (revue humaine requise).

Pas de fine-tune. Device: CUDA si dispo sinon CPU (pas d'upgrade stack).
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
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from ultralytics import YOLO  # noqa: E402

TH19 = {
    "spiky-tower",
    "firecracker-tower",
    "guardian-shrine",
    "squad-barracks",
}
DEFAULT_FOCI = ("wall", "tornado-trap", "giga-bomb", "crafted-defense")
SMALL_AREA_CAP = {
    "wall": 0.08,
    "tornado-trap": 0.20,
    "giga-bomb": 0.25,
    "bomb": 0.30,
    "spring-trap": 0.30,
    "air-bomb": 0.30,
    "giant-bomb": 0.30,
    "seeking-air-mine": 0.30,
    "skeleton-trap": 0.30,
}


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-9, area_a + area_b - inter)


def box_xyxy(box: dict) -> list[float]:
    return [float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])]


def area_frac(xyxy: list[float], width: int, height: int) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, width * height)


def safe_stem(text: str, limit: int = 72) -> str:
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    if len(stem) <= limit:
        return stem
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{stem[: limit - 11]}-{digest}"


def load_shortlist(path: Path, foci: tuple[str, ...], per_class: int) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected: list[dict] = []
    seen_groups: set[str] = set()
    seen_paths: set[str] = set()
    for focus in foci:
        rows = payload.get("shortlist", {}).get(focus) or []
        kept = 0
        for row in rows:
            session_path = ROOT / row["path"]
            if not session_path.is_file():
                continue
            resolved = str(session_path.resolve())
            if resolved in seen_paths:
                continue
            session = json.loads(session_path.read_text(encoding="utf-8"))
            meta = session.get("metadata") or {}
            vg = str(meta.get("village_group") or session_path.stem)
            # dédupliquer path exact + village_group
            key = f"{focus}:{vg}"
            if key in seen_groups:
                continue
            seen_groups.add(key)
            seen_paths.add(resolved)
            selected.append(
                {
                    "focus": focus,
                    "session_path": session_path,
                    "score": row.get("score"),
                    "village_group": vg,
                    "source": meta.get("source") or row.get("source"),
                }
            )
            kept += 1
            if kept >= per_class:
                break
    return selected


def assist_one(
    model: YOLO,
    class_ids: dict[str, int],
    class_names: dict[int, str],
    item: dict,
    out_dir: Path,
    conf: float,
    auto_accept_conf: float,
    imgsz: int,
    device: str,
    iou_match: float,
) -> dict:
    session_path: Path = item["session_path"]
    session = json.loads(session_path.read_text(encoding="utf-8"))
    image_meta = session.get("image") or {}
    image_path = session_path.with_name(image_meta["name"])
    if not image_path.is_file():
        return {"ok": False, "error": f"image manquante: {image_path}", **item}

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    focus = item["focus"]
    stem = safe_stem(f"{focus}__{image_path.stem}")
    dest_img = out_dir / f"{stem}{image_path.suffix.lower()}"
    dest_session = out_dir / f"{stem}.session.json"
    dest_overlay = out_dir / f"{stem}.review.jpg"
    if not dest_img.exists() or dest_img.stat().st_size != image_path.stat().st_size:
        shutil.copy2(image_path, dest_img)

    kept_boxes: list[dict] = []
    for box in session.get("boxes") or []:
        if box.get("status") != "accepted":
            continue
        name = box.get("class_name")
        if name in TH19:
            continue
        xyxy = box_xyxy(box)
        if xyxy[2] - xyxy[0] < 2 or xyxy[3] - xyxy[1] < 2:
            continue
        kept_boxes.append(
            {
                "class_id": int(box["class_id"]),
                "class_name": name,
                "x1": xyxy[0],
                "y1": xyxy[1],
                "x2": xyxy[2],
                "y2": xyxy[3],
                "status": "accepted",
                "origin": "human-focus",
                "review_note": box.get("review_note") or "focus accepté (source inbox)",
            }
        )

    result = model.predict(
        image,
        imgsz=imgsz,
        conf=conf,
        device=device,
        augment=True,
        max_det=1000,
        verbose=False,
    )[0]

    proposals = 0
    auto_accepted = 0
    skipped_overlap = 0
    skipped_th19 = 0
    skipped_area = 0
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
            cap = SMALL_AREA_CAP.get(name, 0.85)
            if frac > cap:
                skipped_area += 1
                continue
            if any(iou_xyxy(xyxy, box_xyxy(b)) >= iou_match for b in kept_boxes):
                skipped_overlap += 1
                continue
            status = "accepted" if conf_v >= auto_accept_conf else "pending"
            note = (
                f"V5 assist conf={conf_v:.3f} auto-accept>={auto_accept_conf}"
                if status == "accepted"
                else f"V5 assist conf={conf_v:.3f} pending revue"
            )
            kept_boxes.append(
                {
                    "class_id": class_ids[name],
                    "class_name": name,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "status": status,
                    "origin": "v5-assist",
                    "confidence": round(conf_v, 4),
                    "review_note": note,
                }
            )
            proposals += 1
            if status == "accepted":
                auto_accepted += 1

    meta = dict(session.get("metadata") or {})
    meta.update(
        {
            "village_group": item["village_group"],
            "source": item["source"] or meta.get("source") or "p0-assist",
            "license": meta.get("license") or "CC BY 4.0",
            "exhaustive": False,
            "draft": True,
            "class_focus": focus,
            "private": True,
            "p0_assist": True,
            "p0_assist_model": "building-detector-v5",
            "scope": "TH10-18",
        }
    )
    out_session = {
        "version": 1,
        "image": {"name": dest_img.name, "width": width, "height": height},
        "metadata": meta,
        "expected_counts": {},
        "boxes": kept_boxes,
    }
    dest_session.write_text(
        json.dumps(out_session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    for box in kept_boxes:
        color = (40, 180, 255) if box.get("origin") == "v5-assist" else (40, 220, 40)
        if box["status"] == "pending":
            color = (0, 165, 255)
        p1 = (int(box["x1"]), int(box["y1"]))
        p2 = (int(box["x2"]), int(box["y2"]))
        cv2.rectangle(canvas, p1, p2, color, 1)
        label = box["class_name"][:18]
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
    cv2.imwrite(str(dest_overlay), canvas)

    counts = Counter(b["class_name"] for b in kept_boxes if b["status"] == "accepted")
    pending = sum(1 for b in kept_boxes if b["status"] == "pending")
    return {
        "ok": True,
        "focus": focus,
        "village_group": item["village_group"],
        "source": item["source"],
        "session": str(dest_session.relative_to(ROOT)).replace("\\", "/"),
        "image": str(dest_img.relative_to(ROOT)).replace("\\", "/"),
        "overlay": str(dest_overlay.relative_to(ROOT)).replace("\\", "/"),
        "accepted": len([b for b in kept_boxes if b["status"] == "accepted"]),
        "pending": pending,
        "proposals": proposals,
        "auto_accepted": auto_accepted,
        "skipped_overlap": skipped_overlap,
        "skipped_th19": skipped_th19,
        "skipped_area": skipped_area,
        "class_counts": dict(sorted(counts.items())),
        "n_classes": len(counts),
        "wh": [width, height],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shortlist",
        type=Path,
        default=ROOT / "training" / "reports" / "pretrain-annotation-shortlist-20261118.json",
    )
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "training" / "inbox" / "private" / "p0-mode-photo",
    )
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "building-detector.pt")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--auto-accept-conf", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--iou-match", type=float, default=0.45)
    parser.add_argument(
        "--foci",
        default=",".join(DEFAULT_FOCI),
        help="Classes focus CSV",
    )
    args = parser.parse_args()

    import torch

    device = args.device if args.device != "0" or torch.cuda.is_available() else "cpu"
    foci = tuple(x.strip() for x in args.foci.split(",") if x.strip())
    selected = load_shortlist(args.shortlist, foci, args.per_class)
    if not selected:
        raise SystemExit("Aucune session shortlist sélectionnée")

    class_payload = json.loads((ROOT / "training" / "classes.json").read_text(encoding="utf-8"))
    class_ids = {item["id"]: index for index, item in enumerate(class_payload["detector_classes"])}
    class_names = {index: item["id"] for index, item in enumerate(class_payload["detector_classes"])}

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    rows = []
    for item in selected:
        print(f"assist {item['focus']} :: {item['session_path'].name}")
        rows.append(
            assist_one(
                model,
                class_ids,
                class_names,
                item,
                args.output,
                conf=args.conf,
                auto_accept_conf=args.auto_accept_conf,
                imgsz=args.imgsz,
                device=device,
                iou_match=args.iou_match,
            )
        )

    report = {
        "updated": date.today().isoformat(),
        "purpose": "P0 assist V5 Mode photo — exhaustive=false jusqu'à revue",
        "device": device,
        "model": str(args.model.relative_to(ROOT)).replace("\\", "/"),
        "conf": args.conf,
        "auto_accept_conf": args.auto_accept_conf,
        "imgsz": args.imgsz,
        "output": str(args.output.relative_to(ROOT)).replace("\\", "/"),
        "selected": len(selected),
        "ok": sum(1 for r in rows if r.get("ok")),
        "rows": rows,
    }
    report_path = (
        ROOT
        / "training"
        / "reports"
        / f"p0-assist-fill-{date.today().isoformat().replace('-', '')}.json"
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "ok": report["ok"], "device": device}, indent=2))


if __name__ == "__main__":
    main()
