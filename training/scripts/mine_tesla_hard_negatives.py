"""Mine hard-negatives hidden-tesla (P0) sans fine-tune.

Stratégie:
1) Parcourt des villages Mode photo assistés / inbox où aucune tesla n'est
   labellée acceptée.
2) Lance V5 (TTA) et collecte les détections hidden-tesla (FP candidats).
3) Écrit un pack de revue + sessions avec boîtes tesla en status=rejected
   (documentation FP) — n'importe PAS automatiquement dans le detector.

Usage typique après assist P0.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from ultralytics import YOLO  # noqa: E402

TESLA = "hidden-tesla"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "training" / "inbox" / "private" / "p0-mode-photo",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "training" / "inbox" / "private" / "p0-tesla-hardneg",
    )
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "building-detector.pt")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-images", type=int, default=24)
    args = parser.parse_args()

    import torch

    device = args.device if args.device != "0" or torch.cuda.is_available() else "cpu"
    class_payload = json.loads((ROOT / "training" / "classes.json").read_text(encoding="utf-8"))
    class_ids = {item["id"]: i for i, item in enumerate(class_payload["detector_classes"])}
    tesla_id = class_ids[TESLA]

    args.input_dir = args.input_dir.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    rows = []
    sessions = sorted(args.input_dir.glob("*.session.json"))[: args.max_images]
    for session_path in sessions:
        session = json.loads(session_path.read_text(encoding="utf-8"))
        accepted = [
            b for b in (session.get("boxes") or []) if b.get("status") == "accepted"
        ]
        if any(b.get("class_name") == TESLA for b in accepted):
            continue
        image_name = (session.get("image") or {}).get("name")
        image_path = session_path.with_name(image_name)
        if not image_path.is_file():
            continue
        image = Image.open(image_path).convert("RGB")
        result = model.predict(
            image,
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
            augment=True,
            max_det=1000,
            verbose=False,
        )[0]
        fps = []
        if result.boxes is not None:
            for det in result.boxes:
                name = result.names[int(det.cls.item())]
                if name != TESLA:
                    continue
                conf_v = float(det.conf.item())
                x1, y1, x2, y2 = (float(v) for v in det.xyxy[0].tolist())
                fps.append(
                    {
                        "class_id": tesla_id,
                        "class_name": TESLA,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "status": "rejected",
                        "origin": "v5-hardneg-candidate",
                        "confidence": round(conf_v, 4),
                        "review_note": (
                            "FP candidat hidden-tesla sur image sans tesla acceptée — "
                            "hard-neg / revue précision"
                        ),
                    }
                )
        if not fps:
            continue
        stem = session_path.stem.replace(".session", "")
        dest_img = args.output / image_path.name
        if not dest_img.exists():
            dest_img.write_bytes(image_path.read_bytes())
        meta = dict(session.get("metadata") or {})
        meta.update(
            {
                "exhaustive": False,
                "draft": True,
                "p0_tesla_hardneg": True,
                "private": True,
            }
        )
        out_session = {
            "version": 1,
            "image": {
                "name": dest_img.name,
                "width": image.width,
                "height": image.height,
            },
            "metadata": meta,
            "expected_counts": {},
            "boxes": accepted
            + fps,  # keep context labels + rejected tesla FPs
        }
        out_json = args.output / f"{stem}.tesla-hardneg.session.json"
        out_json.write_text(
            json.dumps(out_session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        for box in fps:
            p1 = (int(box["x1"]), int(box["y1"]))
            p2 = (int(box["x2"]), int(box["y2"]))
            cv2.rectangle(canvas, p1, p2, (0, 0, 255), 2)
            cv2.putText(
                canvas,
                f"tesla FP {box['confidence']:.2f}",
                (p1[0], max(14, p1[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        overlay = args.output / f"{stem}.tesla-hardneg.review.jpg"
        cv2.imwrite(str(overlay), canvas)
        rows.append(
            {
                "session": str(out_json.relative_to(ROOT)).replace("\\", "/"),
                "overlay": str(overlay.relative_to(ROOT)).replace("\\", "/"),
                "fp_count": len(fps),
                "max_conf": max(b["confidence"] for b in fps),
                "village_group": meta.get("village_group"),
                "focus": meta.get("class_focus"),
            }
        )

    report = {
        "updated": date.today().isoformat(),
        "purpose": "Hard-neg / FP pack hidden-tesla (pas d'import auto, pas de baisse conf)",
        "device": device,
        "input_dir": str(args.input_dir.relative_to(ROOT)).replace("\\", "/"),
        "output": str(args.output.relative_to(ROOT)).replace("\\", "/"),
        "images_with_fp": len(rows),
        "fp_total": sum(r["fp_count"] for r in rows),
        "rows": sorted(rows, key=lambda r: -r["max_conf"]),
        "notes": [
            "Les boîtes tesla sont rejected (documentation FP).",
            "Pour entraîner un hard-neg utile: importer des villages exhaustifs SANS tesla,",
            "ou annoter manuellement les confusions (pas baisser conf inférence).",
        ],
    }
    report_path = (
        ROOT
        / "training"
        / "reports"
        / f"p0-tesla-hardneg-{date.today().isoformat().replace('-', '')}.json"
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "images_with_fp": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
