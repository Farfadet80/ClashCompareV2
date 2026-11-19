"""Revue / promotion P0 des sessions assistées Mode photo.

Règles:
- Refuse TH19.
- Option --drop-pending : rejette les boîtes pending restantes.
- Option --mark-exhaustive : pose exhaustive=true seulement si pending=0,
  ≥ min-classes acceptées, focus présent, et --i-reviewed.
- N'importe PAS dans le dataset (utiliser import_annotation_session.py).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TH19 = {
    "spiky-tower",
    "firecracker-tower",
    "guardian-shrine",
    "squad-barracks",
}


def promote_one(
    path: Path,
    drop_pending: bool,
    mark_exhaustive: bool,
    min_classes: int,
    require_focus: bool,
) -> dict:
    session = json.loads(path.read_text(encoding="utf-8"))
    meta = session.get("metadata") or {}
    focus = meta.get("class_focus")
    changed = False
    stats: Counter[str] = Counter()

    boxes = session.get("boxes") or []
    cleaned: list[dict] = []
    for box in boxes:
        name = box.get("class_name")
        if name in TH19:
            stats["dropped_th19"] += 1
            changed = True
            continue
        if box.get("status") == "pending" and drop_pending:
            box = dict(box)
            box["status"] = "rejected"
            box["review_note"] = (
                (box.get("review_note") or "") + " | P0: pending rejeté avant exhaustive"
            ).strip(" |")
            changed = True
            stats["rejected_pending"] += 1
        cleaned.append(box)
    session["boxes"] = cleaned

    accepted = [b for b in cleaned if b.get("status") == "accepted"]
    pending = [b for b in cleaned if b.get("status") == "pending"]
    classes = Counter(b.get("class_name") for b in accepted)
    focus_ok = (not require_focus) or any(b.get("class_name") == focus for b in accepted)
    can_exhaustive = (
        mark_exhaustive
        and not pending
        and focus_ok
        and len(classes) >= min_classes
        and len(accepted) > 0
    )
    if can_exhaustive:
        meta["exhaustive"] = True
        meta["draft"] = False
        meta["p0_reviewed"] = True
        meta["p0_reviewed_on"] = date.today().isoformat()
        changed = True
        stats["marked_exhaustive"] += 1
    else:
        meta["exhaustive"] = False
        meta["draft"] = True
        if mark_exhaustive:
            stats["exhaustive_blocked"] += 1
    session["metadata"] = meta
    if changed:
        path.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "focus": focus,
        "village_group": meta.get("village_group"),
        "changed": changed,
        "accepted": len(accepted),
        "pending": len(pending),
        "n_classes": len(classes),
        "exhaustive": bool(meta.get("exhaustive")),
        "class_counts": dict(sorted(classes.items())),
        **stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=Path,
        default=ROOT / "training" / "inbox" / "private" / "p0-mode-photo",
    )
    parser.add_argument("--drop-pending", action="store_true")
    parser.add_argument("--mark-exhaustive", action="store_true")
    parser.add_argument(
        "--i-reviewed",
        action="store_true",
        help="Confirmation revue visuelle/overlays effectuée",
    )
    parser.add_argument("--min-classes", type=int, default=4)
    parser.add_argument("--require-focus", action="store_true", default=True)
    parser.add_argument("--session", action="append", type=Path, default=[])
    args = parser.parse_args()

    if args.mark_exhaustive and not args.i_reviewed:
        raise SystemExit("Refus: --mark-exhaustive exige --i-reviewed")

    sessions = list(args.session) if args.session else sorted(args.dir.glob("*.session.json"))
    rows = [
        promote_one(
            path if path.is_absolute() else ROOT / path,
            drop_pending=args.drop_pending,
            mark_exhaustive=args.mark_exhaustive,
            min_classes=args.min_classes,
            require_focus=args.require_focus,
        )
        for path in sessions
    ]
    report = {
        "updated": date.today().isoformat(),
        "dir": str(args.dir.relative_to(ROOT)).replace("\\", "/"),
        "drop_pending": args.drop_pending,
        "mark_exhaustive": args.mark_exhaustive,
        "rows": rows,
        "exhaustive_count": sum(1 for r in rows if r.get("exhaustive")),
    }
    out = (
        ROOT
        / "training"
        / "reports"
        / f"p0-promote-assisted-{date.today().isoformat().replace('-', '')}.json"
    )
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out), "exhaustive_count": report["exhaustive_count"]}, indent=2))


if __name__ == "__main__":
    main()
