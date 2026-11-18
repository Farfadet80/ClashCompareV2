"""Extrait des crops focus acceptés → sessions exhaustives mono-objet (pré-FT).

Pourquoi : importer un village partiellement labellisé en multi-classe est toxique.
Un crop autour d'une boîte focus acceptée = image isolée exhaustive honnête.

Cibles TH10–18 : tornado-trap, giga-bomb, wall (+ P2 faibles optionnel).
Ignore classes YOLO 50–53 / guardian Pedro UI.
N'importe PAS de fine-tune.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
OUT_ROOT = TRAINING / "inbox" / "private" / "focus-crops"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
IMPORT = TRAINING / "scripts" / "import_annotation_session.py"
CLASSES = TRAINING / "classes.json"

DEFAULT_TARGETS = [
    "tornado-trap",
    "giga-bomb",
    "wall",
    "hero-hall",
    "crafted-defense",
    "workshop",
    "pet-house",
    "blacksmith",
]
TH19 = {"spiky-tower", "firecracker-tower", "guardian-shrine", "squad-barracks"}
INBOXES = [
    TRAINING / "inbox" / "public-reference",
    TRAINING / "inbox" / "private",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def class_ids() -> dict[str, int]:
    rows = json.loads(CLASSES.read_text(encoding="utf-8"))["detector_classes"]
    return {row["id"]: index for index, row in enumerate(rows)}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def crop_box(
    box: dict, width: int, height: int, pad: float, min_crop: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(box[k]) for k in ("x1", "y1", "x2", "y2"))
    bw, bh = max(2.0, x2 - x1), max(2.0, y2 - y1)
    px, py = bw * pad, bh * pad
    # Agrandir jusqu'à min_crop pour les petits pièges (tornado/giga)
    need_w = max(min_crop, bw + 2 * px)
    need_h = max(min_crop, bh + 2 * py)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left = int(clamp(cx - need_w / 2.0, 0, width - 2))
    top = int(clamp(cy - need_h / 2.0, 0, height - 2))
    right = int(clamp(left + need_w, left + 2, width))
    bottom = int(clamp(top + need_h, top + 2, height))
    return left, top, right, bottom


def iter_sessions():
    for inbox in INBOXES:
        if not inbox.exists():
            continue
        for path in sorted(inbox.rglob("*.session.json")):
            if "focus-crops" in path.parts:
                continue
            if any(part in TH19 for part in path.parts):
                continue
            yield path


def extract_for_class(
    target: str,
    ids: dict[str, int],
    max_crops: int,
    pad: float,
    min_side: int,
    min_crop: int,
) -> list[dict]:
    out_dir = OUT_ROOT / target
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict] = []
    seen_hash: set[str] = set()

    for session_path in iter_sessions():
        if len(written) >= max_crops:
            break
        session = json.loads(session_path.read_text(encoding="utf-8"))
        meta = session.get("metadata") or {}
        if meta.get("class_focus") != target:
            continue
        if target == "town-hall-guardian" and meta.get("source") == "pedro-coc-model":
            continue

        image_meta = session.get("image") or {}
        image_name = image_meta.get("name")
        if not image_name:
            continue
        image_path = session_path.with_name(image_name)
        if not image_path.exists():
            # parfois image gitignorée absente
            continue

        width = int(image_meta.get("width") or 0)
        height = int(image_meta.get("height") or 0)
        focus_boxes = [
            b
            for b in (session.get("boxes") or [])
            if b.get("class_name") == target and b.get("status") == "accepted"
        ]
        if not focus_boxes:
            continue

        with Image.open(image_path) as im:
            im = im.convert("RGB")
            if not width or not height:
                width, height = im.size
            for index, box in enumerate(focus_boxes):
                if len(written) >= max_crops:
                    break
                obw = float(box["x2"]) - float(box["x1"])
                obh = float(box["y2"]) - float(box["y1"])
                if obw < min_side or obh < min_side:
                    continue
                left, top, right, bottom = crop_box(box, width, height, pad, min_crop)
                crop = im.crop((left, top, right, bottom))
                cw, ch = crop.size
                nx1 = clamp(float(box["x1"]) - left, 0, cw - 1)
                ny1 = clamp(float(box["y1"]) - top, 0, ch - 1)
                nx2 = clamp(float(box["x2"]) - left, nx1 + 2, cw)
                ny2 = clamp(float(box["y2"]) - top, ny1 + 2, ch)
                cov = ((nx2 - nx1) * (ny2 - ny1)) / max(1.0, cw * ch)
                # petits objets dans grand crop : cov peut être basse — OK si >= 0.02
                if cov < 0.02 or cov > 0.98:
                    continue

                buf = __import__("io").BytesIO()
                crop.save(buf, format="JPEG", quality=92)
                digest = sha256_bytes(buf.getvalue())
                if digest in seen_hash:
                    continue
                seen_hash.add(digest)

                stem = (
                    f"crop__{target}__{meta.get('source', 'src')}__"
                    f"{re.sub(r'[^a-zA-Z0-9]+', '-', Path(image_name).stem)[:40]}__{index}"
                )
                dest_img = out_dir / f"{stem}.jpg"
                dest_img.write_bytes(buf.getvalue())

                village_group = f"focus-crop-{target}-{digest[:12]}"
                provenance = {
                    "village_group": village_group,
                    "class_focus": target,
                    "license": meta.get("license") or "CC BY 4.0",
                    "source_session": str(session_path.relative_to(ROOT)).replace("\\", "/"),
                    "source_image": image_name,
                    "sha256": digest,
                    "usage": "pretrain_focus_crop",
                    "publish": False,
                    "notes": [
                        "Crop mono-objet depuis boîte focus accepted.",
                        "Exhaustive pour cette image isolée uniquement.",
                        "Ne pas republier l'image.",
                    ],
                }
                (out_dir / f"{stem}.provenance.json").write_text(
                    json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                session_out = {
                    "version": 1,
                    "image": {"name": dest_img.name, "width": cw, "height": ch},
                    "metadata": {
                        "village_group": village_group,
                        "source": f"focus-crop:{meta.get('source', 'unknown')}",
                        "license": provenance["license"]
                        if str(provenance["license"]).startswith("CC")
                        or str(provenance["license"]).lower().startswith("consentement")
                        else "CC BY 4.0",
                        "exhaustive": True,
                        "draft": False,
                        "class_focus": target,
                        "private": True,
                        "crop_promoted": True,
                        "crop_promote_reason": "focus-crop-extract",
                    },
                    "expected_counts": {target: 1},
                    "boxes": [
                        {
                            "class_id": ids[target],
                            "class_name": target,
                            "x1": round(nx1, 2),
                            "y1": round(ny1, 2),
                            "x2": round(nx2, 2),
                            "y2": round(ny2, 2),
                            "status": "accepted",
                            "origin": "focus-crop-extract",
                            "review_note": "crop isolé depuis focus accepted",
                        }
                    ],
                }
                sess_path = out_dir / f"{stem}.session.json"
                sess_path.write_text(
                    json.dumps(session_out, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                written.append(
                    {
                        "session": str(sess_path.relative_to(ROOT)).replace("\\", "/"),
                        "image": str(dest_img.relative_to(ROOT)).replace("\\", "/"),
                        "target": target,
                        "coverage": round(cov, 4),
                        "wh": [cw, ch],
                    }
                )
    return written


def assign_split(index: int) -> str:
    # Train-heavy (80/10/10) pour combler les seuils very_weak rapidement
    # tout en gardant un peu de val/test.
    mod = index % 10
    if mod == 8:
        return "val"
    if mod == 9:
        return "test"
    return "train"


def import_crops(rows: list[dict], dry_run: bool) -> list[dict]:
    results = []
    for index, row in enumerate(rows):
        split = assign_split(index)
        cmd = [
            str(PY),
            str(IMPORT),
            str(ROOT / row["session"]),
            str(ROOT / row["image"]),
            "--split",
            split,
        ]
        if dry_run:
            cmd.append("--dry-run")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        try:
            body = json.loads(proc.stdout)
        except json.JSONDecodeError:
            body = {
                "ok": False,
                "errors": [proc.stdout or proc.stderr or f"exit {proc.returncode}"],
            }
        results.append({"target": row["target"], "split": split, **body})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--max-per-class", type=int, default=40)
    parser.add_argument("--pad", type=float, default=0.25)
    parser.add_argument(
        "--min-side",
        type=int,
        default=16,
        help="Taille mini de la boîte objet (px)",
    )
    parser.add_argument(
        "--min-crop",
        type=int,
        default=96,
        help="Taille mini du crop image (px) — important pour petits pièges",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-import", action="store_true")
    parser.add_argument(
        "--output-report",
        type=Path,
        default=TRAINING
        / "reports"
        / f"focus-crops-pretrain-{date.today().isoformat().replace('-', '')}.json",
    )
    args = parser.parse_args()

    ids = class_ids()
    targets = [t for t in args.targets if t in ids and t not in TH19]
    extracted: dict[str, list] = {}
    for target in targets:
        extracted[target] = extract_for_class(
            target,
            ids,
            args.max_per_class,
            args.pad,
            args.min_side,
            args.min_crop,
        )

    flat = [row for rows in extracted.values() for row in rows]
    imports: list[dict] = []
    if not args.skip_import and flat:
        imports = import_crops(flat, args.dry_run)

    totals = Counter()
    for row in imports:
        totals[f"attempt_{row.get('split')}"] += 1
        totals["ok" if row.get("ok") else "fail"] += 1
        totals[f"ok_{row.get('target')}"] += 1 if row.get("ok") else 0

    payload = {
        "updated": date.today().isoformat(),
        "dry_run": args.dry_run,
        "finetune": False,
        "targets": targets,
        "extracted_counts": {k: len(v) for k, v in extracted.items()},
        "import_totals": dict(totals),
        "extracted": extracted,
        "imports": imports,
        "notes": [
            "Crops mono-objet pour combler seuils pré-FT sans poison multi-classe.",
            "Pas de fine-tune dans ce script ; gate Jimmy inchangé.",
            "Images focus-crops gitignorées via inbox/private/**/*.jpg.",
        ],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "extracted_counts": payload["extracted_counts"],
                "import_totals": dict(totals),
                "report": str(args.output_report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
