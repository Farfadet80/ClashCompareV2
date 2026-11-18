"""Promeut en exhaustive UNIQUEMENT les crops isolés fiables (TH10–18).

Critères stricts (après revue visuelle 2026-11-18) :
- removebg dans le nom d'image, OU
- sprite mur Clashia (`Wall\\d+` / `train_wall_`) avec 1 seule boîte wall
  et couverture ∈ [0.08, 0.55] sur cadre ≤ 640²*1.2

Interdit (restent draft) :
- screenshots villages (base-finder, cocsegmentation, thN_farm/war, …)
- classes YOLO 50–53 / TH19
- town-hall-guardian Pedro UI

Ne lance PAS import_annotation_session / fine-tune.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
INBOXES = [
    TRAINING / "inbox" / "public-reference",
    TRAINING / "inbox" / "private",
]

ALLOWED_FOCUS = {
    "wall",
    "tornado-trap",
    "giga-bomb",
    "workshop",
    "pet-house",
    "blacksmith",
    "hero-hall",
    "crafted-defense",
    "town-hall-guardian",
}
TH19_FORBIDDEN = {
    "spiky-tower",
    "firecracker-tower",
    "guardian-shrine",
    "squad-barracks",
}
WALL_SPRITE_RE = re.compile(
    r"(removebg|train_wall_|Wall\d+(?:-removebg|_webp))",
    re.IGNORECASE,
)
VILLAGE_HINT_RE = re.compile(
    r"(Screenshot|th\d+_(?:farm|war|defence|defense)|base-finder|cocsegmentation|"
    r"IMG_\d+|frame_\d+)",
    re.IGNORECASE,
)


def area_frac(box: dict, width: int, height: int) -> float:
    x1, y1, x2, y2 = (float(box[k]) for k in ("x1", "y1", "x2", "y2"))
    return max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, width * height)


def is_crop_candidate(session: dict, image_name: str) -> tuple[bool, str]:
    meta = session.get("metadata") or {}
    focus = meta.get("class_focus")
    if focus in TH19_FORBIDDEN:
        return False, "classe TH19 ignorée"
    if focus not in ALLOWED_FOCUS:
        return False, "focus hors périmètre TH10–18"
    if focus == "town-hall-guardian" and meta.get("source") == "pedro-coc-model":
        return False, "guardian UI Pedro non promu"
    if VILLAGE_HINT_RE.search(image_name):
        return False, "indice village/plein cadre"

    image = session.get("image") or {}
    width = int(image.get("width") or 0)
    height = int(image.get("height") or 0)
    area = width * height
    boxes = [b for b in (session.get("boxes") or []) if b.get("status") != "rejected"]
    if not boxes:
        return False, "aucune boîte active"

    names = {b.get("class_name") for b in boxes}
    focus_boxes = [b for b in boxes if b.get("class_name") == focus]
    if not focus_boxes:
        return False, "pas de boîte focus"

    coverage = sum(area_frac(b, width, height) for b in boxes)
    lowered = image_name.lower()

    if "removebg" in lowered and focus == "wall" and names <= {"wall"}:
        return True, "removebg wall crop"

    if (
        focus == "wall"
        and names == {"wall"}
        and len(boxes) == 1
        and WALL_SPRITE_RE.search(image_name)
        and 0.08 <= coverage <= 0.55
        and area <= int(640 * 640 * 1.2)
    ):
        return True, "wall sprite crop"

    return False, f"non-crop (cov={coverage:.2%} area={area} focus={focus})"


def demote_false_promotions(path: Path, dry_run: bool) -> dict | None:
    """Remet draft les sessions crop_promoted qui ne passent plus les critères."""
    session = json.loads(path.read_text(encoding="utf-8"))
    meta = session.get("metadata") or {}
    if not meta.get("crop_promoted"):
        return None
    image_name = str((session.get("image") or {}).get("name") or path.name)
    ok, reason = is_crop_candidate(session, image_name)
    if ok:
        return None
    meta["exhaustive"] = False
    meta["draft"] = True
    meta["crop_promoted"] = False
    meta["crop_demoted_reason"] = reason
    meta["crop_demote_date"] = date.today().isoformat()
    session["metadata"] = meta
    if not dry_run:
        path.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "demoted": True,
        "reason": reason,
    }


def promote_session(path: Path, dry_run: bool) -> dict:
    session = json.loads(path.read_text(encoding="utf-8"))
    meta = session.get("metadata") or {}
    image = session.get("image") or {}
    width = int(image.get("width") or 1)
    height = int(image.get("height") or 1)
    image_name = str(image.get("name") or path.name)
    ok, reason = is_crop_candidate(session, image_name)
    stats: Counter = Counter()
    if not ok:
        return {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "promoted": False,
            "reason": reason,
        }

    focus = meta.get("class_focus")
    for box in session.get("boxes") or []:
        name = box.get("class_name")
        status = box.get("status")
        if status == "rejected":
            stats["already_rejected"] += 1
            continue
        frac = area_frac(box, width, height)
        # crops mur : plafond large
        lim = 0.55 if name == "wall" else 0.85
        if frac > lim:
            box["status"] = "rejected"
            box["review_note"] = f"crop promote: aire {frac:.2%} > {lim:.0%}"
            stats["rejected_oversized"] += 1
            continue
        if status != "accepted":
            box["status"] = "accepted"
            box["review_note"] = f"crop isolé accepté ({reason})"
            stats["accepted"] += 1
        else:
            stats["already_accepted"] += 1

    active = [b for b in (session.get("boxes") or []) if b.get("status") == "accepted"]
    focus_ok = [b for b in active if b.get("class_name") == focus]
    if not focus_ok:
        return {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "promoted": False,
            "reason": "plus de boîte focus acceptée après hygiène",
            **stats,
        }

    pending = [b for b in (session.get("boxes") or []) if b.get("status") == "pending"]
    if pending:
        return {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "promoted": False,
            "reason": f"{len(pending)} pending restants",
            **stats,
        }

    meta["exhaustive"] = True
    meta["draft"] = False
    meta["crop_promoted"] = True
    meta["crop_promote_reason"] = reason
    meta["crop_promote_date"] = date.today().isoformat()
    meta.pop("crop_demoted_reason", None)
    session["metadata"] = meta
    if not dry_run:
        path.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "promoted": True,
        "reason": reason,
        "focus": focus,
        "accepted_boxes": len(active),
        **stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-report",
        type=Path,
        default=TRAINING / "reports" / f"crop-promote-{date.today().isoformat().replace('-', '')}.json",
    )
    args = parser.parse_args()

    demoted = []
    results = []
    totals: Counter = Counter()
    for inbox in INBOXES:
        if not inbox.exists():
            continue
        for session_path in sorted(inbox.rglob("*.session.json")):
            if any(part in TH19_FORBIDDEN for part in session_path.parts):
                totals["skipped_th19_path"] += 1
                continue
            drow = demote_false_promotions(session_path, args.dry_run)
            if drow:
                demoted.append(drow)
                totals["demoted"] += 1
            row = promote_session(session_path, args.dry_run)
            results.append(row)
            totals["seen"] += 1
            if row.get("promoted"):
                totals["promoted"] += 1
                totals[f"focus_{row.get('focus')}"] += 1
            else:
                totals["skipped"] += 1

    payload = {
        "updated": date.today().isoformat(),
        "dry_run": args.dry_run,
        "policy": {
            "th19_classes_ignored": sorted(TH19_FORBIDDEN),
            "allowed_focus": sorted(ALLOWED_FOCUS),
            "strict_crops_only": True,
            "import_detector": False,
            "finetune": False,
            "note": (
                "Exhaustive = removebg/sprites mur seulement. "
                "Villages plein cadre restent draft même si focus boxes accepted."
            ),
        },
        "totals": dict(totals),
        "demoted_sessions": demoted,
        "promoted_sessions": [r for r in results if r.get("promoted")],
        "skipped_sample": [r for r in results if not r.get("promoted")][:40],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"totals": dict(totals), "demoted": len(demoted), "report": str(args.output_report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
