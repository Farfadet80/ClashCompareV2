"""Inventaire local de training/inbox/private/ (images hors git).

Pas d'import detector, pas de train. Sert au rapport de collecte.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "training" / "inbox" / "private"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "training"
        / "reports"
        / f"wall-private-collection-inventory-{date.today().isoformat().replace('-', '')}.json",
    )
    args = parser.parse_args()

    folders: dict[str, dict] = {}
    class_focus = Counter()
    exhaustive = Counter()
    sources = Counter()
    mode_photo_rows = []
    wall_poison_risk = []

    for folder in sorted(p for p in PRIVATE.iterdir() if p.is_dir()):
        sessions = list(folder.rglob("*.session.json"))
        images = [
            p
            for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_SUFFIXES
            and ".review." not in p.name
        ]
        folders[folder.name] = {
            "sessions": len(sessions),
            "images": len(images),
        }
        for session_path in sessions:
            session = json.loads(session_path.read_text(encoding="utf-8"))
            meta = session.get("metadata") or {}
            boxes = session.get("boxes") or []
            focus = meta.get("class_focus") or folder.name
            class_focus[str(focus)] += 1
            exhaustive[str(bool(meta.get("exhaustive")))] += 1
            sources[str(meta.get("source") or "unknown")] += 1
            accepted = [b for b in boxes if b.get("status") == "accepted"]
            pending = [b for b in boxes if b.get("status") == "pending"]
            n_classes = len({b.get("class_name") for b in boxes if b.get("class_name")})
            image = session.get("image") or {}
            wh = [int(image.get("width") or 0), int(image.get("height") or 0)]
            row = {
                "folder": folder.name,
                "session": str(session_path.relative_to(ROOT)).replace("\\", "/"),
                "focus": focus,
                "source": meta.get("source"),
                "exhaustive": bool(meta.get("exhaustive")),
                "draft": bool(meta.get("draft")),
                "boxes": len(boxes),
                "accepted": len(accepted),
                "pending": len(pending),
                "n_classes": n_classes,
                "wh": wh,
            }
            if folder.name == "p0-mode-photo":
                mode_photo_rows.append(row)
                # wall-only / sprite-like = risque poison si import detector
                name = str(image.get("name") or "").lower()
                if n_classes <= 2 or "removebg" in name:
                    wall_poison_risk.append(row)

    payload = {
        "updated": date.today().isoformat(),
        "publish_images": False,
        "train_now": False,
        "private_root": "training/inbox/private",
        "folders": folders,
        "totals": {
            "sessions": sum(v["sessions"] for v in folders.values()),
            "images": sum(v["images"] for v in folders.values()),
        },
        "sessions_by_class_focus": dict(class_focus),
        "exhaustive_flags": dict(exhaustive),
        "top_sources": sources.most_common(20),
        "p0_mode_photo": {
            "count": len(mode_photo_rows),
            "rows": mode_photo_rows,
        },
        "poison_risk_wall_only_or_sprite": {
            "count": len(wall_poison_risk),
            "note": "Ne pas importer dans detector sans revue exhaustive multi-classe",
            "rows": wall_poison_risk,
        },
        "blocked_without_annotator": [
            "exhaustive=true sur villages Mode photo complets",
            "import_annotation_session village multi-classe honnête",
            "murs correctement labellés sur villages (pas seulement crops)",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.output),
                "totals": payload["totals"],
                "p0_mode_photo": len(mode_photo_rows),
                "poison_risk": len(wall_poison_risk),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
