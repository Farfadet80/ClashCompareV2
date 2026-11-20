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
SPRITE_MARKERS = ("removebg", "webp", "sprite", "catalog", "wall1-", "wall2-", "mauer")
VILLAGE_SOURCES = {
    "evans-home-village-building-detector",
    "base-finder-th18",
    "cocsegmentation-villages",
    "glenn-th18-gigabomb",
    "coc-all-traps",
    "pedro-coc-model",
    "nothing-clash-of-clans",
    "clashia-detectwall-v7",
    "kiet-walls-v6",
}


def score_session(session: dict, prefer_mode_photo: bool = False) -> tuple[float, dict]:
    meta = session.get("metadata") or {}
    focus = meta.get("class_focus")
    image = session.get("image") or {}
    width = int(image.get("width") or 1)
    height = int(image.get("height") or 1)
    boxes = session.get("boxes") or []
    accepted = [b for b in boxes if b.get("status") == "accepted"]
    # File annotator: accepter aussi pending focus (imports privés draft)
    focus_boxes = [
        b
        for b in boxes
        if b.get("class_name") == focus and b.get("status") in {"accepted", "pending"}
    ]
    focus_acc = [b for b in accepted if b.get("class_name") == focus]
    if not focus_boxes or meta.get("exhaustive") is True:
        return -1.0, {}
    scored_focus = focus_acc or focus_boxes
    cov = sum(
        max(0.0, (float(b["x2"]) - float(b["x1"])) * (float(b["y2"]) - float(b["y1"])))
        for b in scored_focus
    ) / max(1.0, width * height)
    pending = sum(1 for b in boxes if b.get("status") == "pending")
    rejected = sum(1 for b in boxes if b.get("status") == "rejected")
    n_classes = len({b.get("class_name") for b in boxes if b.get("class_name")})
    source = str(meta.get("source") or "")
    image_name = str(image.get("name") or "").lower()
    sprite_like = any(marker in image_name for marker in SPRITE_MARKERS) or (
        "removebg" in image_name
    )
    score = len(scored_focus) * 2.0 + cov * 5.0 - pending * 0.05 - rejected * 0.05
    if prefer_mode_photo:
        # Villages Mode photo : grande image + multi-classes ; pénaliser sprites mur
        if width >= 900 and height >= 700:
            score += 4.0
        if n_classes >= 8:
            score += 6.0
        elif n_classes >= 3:
            score += 2.0
        if source in VILLAGE_SOURCES:
            score += 3.0
        if sprite_like:
            score -= 12.0
        elif n_classes <= 1 and width < 800:
            score -= 8.0
        elif n_classes <= 1:
            # Village partiellement labellé (ex. murs seuls) = bon candidat assist
            score += 1.0
        if width * height <= 400 * 400:
            score -= 5.0
        # Garde-fou : village (même redimensionné 512) avec focus reste shortlistable
        if (
            scored_focus
            and not sprite_like
            and width >= 480
            and height >= 480
            and (source in VILLAGE_SOURCES or width >= 800)
        ):
            score = max(score, 0.5)
    else:
        if width * height <= 640 * 640 * 1.2:
            score += 1.5
    return score, {
        "focus": focus,
        "focus_accepted": len(focus_acc),
        "focus_boxes": len(focus_boxes),
        "pending": pending,
        "rejected": rejected,
        "coverage": round(cov, 4),
        "n_classes": n_classes,
        "wh": [width, height],
        "source": meta.get("source"),
        "private": bool(meta.get("private")),
        "sprite_like": sprite_like,
        "mode_photo_candidate": bool(
            prefer_mode_photo and not sprite_like and n_classes >= 2 and width >= 800
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=15)
    parser.add_argument(
        "--prefer-mode-photo",
        action="store_true",
        help="Priorise villages Mode photo (pénalise sprites mur / removebg)",
    )
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
            # p0 packs déjà assistés / hard-neg : ne pas re-shortlister
            if "p0-mode-photo" in path.parts or "p0-tesla-hardneg" in path.parts:
                continue
            if "p0-wall-tiles" in path.parts or "focus-crops" in path.parts:
                continue
            session = json.loads(path.read_text(encoding="utf-8"))
            focus = (session.get("metadata") or {}).get("class_focus")
            if focus not in PRIORITY:
                continue
            score, info = score_session(session, prefer_mode_photo=args.prefer_mode_photo)
            if score < 0:
                continue
            if args.prefer_mode_photo and info.get("sprite_like") and focus == "wall":
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
        "purpose": (
            "File Mode photo assist (exhaustive=false)"
            if args.prefer_mode_photo
            else "File annotator exhaustive avant fine-tune (pas de train ici)"
        ),
        "prefer_mode_photo": bool(args.prefer_mode_photo),
        "per_class": args.per_class,
        "annotator": "http://127.0.0.1:8765/training/annotator/",
        "shortlist": shortlist,
        "counts": {k: len(v) for k, v in shortlist.items()},
        "notes": [
            "Villages mur-only (peters): utiles classifieur niveaux, dangereux detector multi-classe.",
            "Classes 50-53 ignorees.",
            "Apres exhaustive: import_annotation_session.py puis audit_split_integrity.",
            "Ne jamais importer wall-only poison / tiles poison sans revue.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": payload["counts"], "report": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
