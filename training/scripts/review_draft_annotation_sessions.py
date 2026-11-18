"""Revue hygiène des brouillons C1–C2 + file privée (sans Jimmy).

- Rejette boîtes géométriquement absurdes (ex. piège couvrant >35 % de l'image).
- Conserve guardian déjà rejected (Pedro UI).
- N'accepte PAS une session village comme exhaustive : import reste bloqué sans
  revue visuelle complète de tous les bâtiments visibles.
- Option --accept-focus-boxes : marque accepted uniquement les boîtes focus
  géométriquement plausibles ; exhaustive reste false / draft true.
- Option --accept-plausible-boxes : accepte toute boîte pending géométriquement OK
  (focus + siblings mappés) ; exhaustive reste false.
- Option --include-private : parcourt aussi training/inbox/private.
- Ignore les dossiers classes TH19 (50–53) même s'ils existent.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_INBOX = ROOT / "training" / "inbox" / "public-reference"
PRIVATE_INBOX = ROOT / "training" / "inbox" / "private"
TH19_SKIP = {
    "spiky-tower",
    "firecracker-tower",
    "guardian-shrine",
    "squad-barracks",
}
SMALL_CLASSES = {
    "tornado-trap",
    "giga-bomb",
    "bomb",
    "spring-trap",
    "air-bomb",
    "giant-bomb",
    "seeking-air-mine",
    "skeleton-trap",
    "wall",
}
MAX_AREA_FRAC = {
    "tornado-trap": 0.20,
    "giga-bomb": 0.25,
    "wall": 0.08,
    "default_small": 0.30,
}


def area_frac(box: dict, width: int, height: int) -> float:
    x1, y1, x2, y2 = (float(box[k]) for k in ("x1", "y1", "x2", "y2"))
    return max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, width * height)


def review_session(
    path: Path,
    accept_focus: bool,
    accept_plausible: bool,
) -> dict:
    session = json.loads(path.read_text(encoding="utf-8"))
    meta = session.get("metadata") or {}
    image = session.get("image") or {}
    width = int(image.get("width") or 1)
    height = int(image.get("height") or 1)
    focus = meta.get("class_focus")
    changed = False
    stats = Counter()

    if focus in TH19_SKIP:
        stats["skipped_th19"] += 1
        return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "changed": False, **stats}

    # Contre-exemples guardian UI déjà connus
    if focus == "town-hall-guardian" and meta.get("source") == "pedro-coc-model":
        for box in session.get("boxes") or []:
            if box.get("status") != "rejected":
                box["status"] = "rejected"
                box["review_note"] = "UI tray troupes / pas guardian village"
                changed = True
                stats["rejected_guardian_ui"] += 1
        meta["draft"] = True
        meta["exhaustive"] = False
        session["metadata"] = meta
        if changed:
            path.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"path": path.as_posix(), "changed": changed, **stats}

    # Crops removebg : plafond mur plus large (objet remplit le cadre)
    wall_limit = 0.35 if "removebg" in str(image.get("name") or "").lower() else MAX_AREA_FRAC["wall"]

    for box in session.get("boxes") or []:
        name = box.get("class_name")
        status = box.get("status")
        if status == "rejected":
            stats["already_rejected"] += 1
            continue
        frac = area_frac(box, width, height)
        if name == "wall":
            limit = wall_limit
        else:
            limit = MAX_AREA_FRAC.get(
                name, MAX_AREA_FRAC["default_small"] if name in SMALL_CLASSES else 0.85
            )
        if frac > limit:
            box["status"] = "rejected"
            box["review_note"] = f"hygiène auto: aire {frac:.2%} > plafond {limit:.0%}"
            changed = True
            stats["rejected_oversized"] += 1
            continue
        if status == "pending" and accept_plausible:
            box["status"] = "accepted"
            box["review_note"] = "géométrie OK — session toujours draft/non exhaustive"
            changed = True
            stats["accepted_plausible"] += 1
        elif accept_focus and name == focus and status == "pending":
            box["status"] = "accepted"
            box["review_note"] = "focus géométrie OK — session toujours draft/non exhaustive"
            changed = True
            stats["accepted_focus"] += 1
        else:
            stats["left_pending"] += 1

    # Jamais d'exhaustive automatique ici (voir promote_isolated_crop_sessions.py)
    if meta.get("exhaustive") is True and meta.get("draft") is not False and not meta.get("crop_promoted"):
        meta["exhaustive"] = False
        changed = True
        stats["forced_non_exhaustive"] += 1
    if not meta.get("crop_promoted"):
        meta["draft"] = True
    session["metadata"] = meta
    if changed:
        path.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "changed": changed, **stats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--accept-focus-boxes",
        action="store_true",
        help="Accepter les boîtes focus géométriquement OK (exhaustive reste false)",
    )
    parser.add_argument(
        "--accept-plausible-boxes",
        action="store_true",
        help="Accepter toute boîte pending géométriquement OK (exhaustive reste false)",
    )
    parser.add_argument(
        "--include-private",
        action="store_true",
        help="Inclure training/inbox/private (images hors git)",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=ROOT / "training" / "reports" / "draft-review-hygiene.json",
    )
    args = parser.parse_args()

    roots = [PUBLIC_INBOX]
    if args.include_private:
        roots.append(PRIVATE_INBOX)

    results = []
    totals = Counter()
    for inbox in roots:
        if not inbox.exists():
            continue
        for session_path in sorted(inbox.rglob("*.session.json")):
            if any(part in TH19_SKIP for part in session_path.parts):
                totals["skipped_th19_path"] += 1
                continue
            row = review_session(
                session_path,
                args.accept_focus_boxes,
                args.accept_plausible_boxes,
            )
            results.append(row)
            for key, value in row.items():
                if key in {"path", "changed"}:
                    continue
                totals[key] += int(value)
            if row.get("changed"):
                totals["sessions_changed"] += 1
            totals["sessions_seen"] += 1

    report = {
        "inboxes": [str(r.relative_to(ROOT)).replace("\\", "/") for r in roots],
        "accept_focus_boxes": args.accept_focus_boxes,
        "accept_plausible_boxes": args.accept_plausible_boxes,
        "include_private": args.include_private,
        "exhaustive_auto": False,
        "import_ready": False,
        "th19_ignored": True,
        "note": (
            "Sans annotation exhaustive village (tous bâtiments visibles), "
            "import_annotation_session refuse toujours. Crops isolés : "
            "promote_isolated_crop_sessions.py. Pas de fine-tune ici."
        ),
        "totals": dict(totals),
        "sessions": results,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"totals": report["totals"], "note": report["note"]}, ensure_ascii=False, indent=2))
    print(f"Rapport: {args.output_report}")


if __name__ == "__main__":
    main()
