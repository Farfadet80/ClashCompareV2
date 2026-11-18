"""Shortlist sessions inbox prioritaires pour annotation exhaustive (TH10-18).

Ne marque PAS exhaustive. File annotator avant FT. Ignore classes YOLO 50-53.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INBOXES = [
    ROOT / "training" / "inbox" / "public-reference",
    ROOT / "training" / "inbox" / "private",
]
PRIORITY = [
    "wall",
    "tornado-trap",
    "giga-bomb",
    "hero-hall",
    "crafted-defense",
    "workshop",
    "pet-house",
    "blacksmith",
]
TH19 = {"spiky-tower", "firecracker-tower", "guardian-shrine", "squad-barracks"}


def score_session(session: dict) -> tuple[float, dict]:
    meta = session.get("metadata") or {}
    focus = meta.get("class_focus")
    image = session.get("image") or {}
    width = int(image.get("width") or 1)
    height = int(image.get("height") or 1)
    boxes = session.get("boxes") or []
    accepted = [b for b in boxes if b.get("status") == "accepted"]
    focus_acc = [b for b in accepted if b.get("class_name") == focus]
    if not focus_acc or meta.get("exhaustive") is True:
        return -1.0, {}
    cov = sum(
        max(0.0, (float(b["x2"]) - float(b["x1"])) * (float(b["y2"]) - float(b["y1"])))
        for b in focus_acc
    ) / max(1.0, width * height)
    pending = sum(1 for b in boxes if b.get("status") == "pending")
    rejected = sum(1 for b in boxes if b.get("status") == "rejected")
    score = len(focus_acc) * 2.0 + cov * 5.0 - pending * 0.1 - rejected * 0.05
    if width * height <= 640 * 640 * 1.2:
        score += 1.5
    return score, {
        "focus": focus,
        "focus_accepted": len(focus_acc),
        "pending": pending,
        "rejected": rejected,
        "coverage": round(cov, 4),
        "wh": [width, height],
        "source": meta.get("source"),
        "private": bool(meta.get("private")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=15)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "training"
        / "reports"
        / f"pretrain-annotation-shortlist-{date.today().isoformat().replace('-', '')}.json",
    )
    args = parser.parse_args()

    by_class: dict[str, list] = defaultdict(list)
    for inbox in INBOXES:
        if not inbox.exists():
            continue
        for path in inbox.rglob("*.session.json"):
            if any(part in TH19 for part in path.parts):
                continue
            session = json.loads(path.read_text(encoding="utf-8"))
            focus = (session.get("metadata") or {}).get("class_focus")
            if focus not in PRIORITY:
                continue
            score, info = score_session(session)
            if score < 0:
                continue
            by_class[focus].append(
                {
                    "score": round(score, 3),
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    **info,
                }
            )

    shortlist = {
        name: sorted(by_class.get(name) or [], key=lambda r: r["score"], reverse=True)[
            : args.per_class
        ]
        for name in PRIORITY
    }
    payload = {
        "updated": date.today().isoformat(),
        "purpose": "File annotator exhaustive avant fine-tune V5 (pas de train ici)",
        "per_class": args.per_class,
        "annotator": "http://127.0.0.1:8765/training/annotator/",
        "shortlist": shortlist,
        "counts": {k: len(v) for k, v in shortlist.items()},
        "notes": [
            "Villages mur-only (peters): utiles classifieur niveaux, dangereux detector multi-classe.",
            "Classes 50-53 ignorees.",
            "Apres exhaustive: import_annotation_session.py puis audit_split_integrity.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": payload["counts"], "report": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
