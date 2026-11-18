"""Prépare splits mur sprite (removebg / Wall*_webp) avant fine-tune.

- Répare les boîtes wall rejetées à tort sur crops isolés (plafond trop bas).
- Promeut exhaustive les sprites wall stricts.
- Importe vers train/val/test (ratio ~60/20/20) via import_annotation_session.
- Ignore villages plein cadre (peters frames, etc.) — labels mur-only = poison multi-classe.
- Ne lance PAS de fine-tune.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
PRIVATE_WALL = TRAINING / "inbox" / "private" / "wall"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
IMPORT = TRAINING / "scripts" / "import_annotation_session.py"

SPRITE_RE = re.compile(r"(removebg|Wall\d+(?:-removebg|_webp))", re.IGNORECASE)
VILLAGE_RE = re.compile(r"(frame_|Screenshot|th\d+_|peters)", re.IGNORECASE)
WALL_CROP_LIMIT = 0.60


def area_frac(box: dict, width: int, height: int) -> float:
    return max(0.0, (float(box["x2"]) - float(box["x1"])) * (float(box["y2"]) - float(box["y1"]))) / max(
        1.0, width * height
    )


def repair_and_promote(path: Path) -> dict:
    session = json.loads(path.read_text(encoding="utf-8"))
    meta = session.get("metadata") or {}
    image = session.get("image") or {}
    name = str(image.get("name") or path.name)
    if VILLAGE_RE.search(name) and not SPRITE_RE.search(name):
        return {"path": path.name, "action": "skip_village"}
    if not SPRITE_RE.search(name):
        return {"path": path.name, "action": "skip_not_sprite"}
    if meta.get("class_focus") != "wall":
        return {"path": path.name, "action": "skip_focus"}

    width = int(image.get("width") or 1)
    height = int(image.get("height") or 1)
    changed = False
    accepted = 0
    for box in session.get("boxes") or []:
        if box.get("class_name") != "wall":
            continue
        frac = area_frac(box, width, height)
        if frac > WALL_CROP_LIMIT or frac <= 0:
            if box.get("status") != "rejected":
                box["status"] = "rejected"
                box["review_note"] = f"sprite prep: aire {frac:.2%} hors [0,{WALL_CROP_LIMIT:.0%}]"
                changed = True
            continue
        if box.get("status") != "accepted":
            box["status"] = "accepted"
            box["review_note"] = "sprite mur isolé réparé/accepté"
            changed = True
        accepted += 1

    if accepted == 0:
        if changed:
            path.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"path": path.name, "action": "no_accepted_wall"}

    pending = [b for b in (session.get("boxes") or []) if b.get("status") == "pending"]
    if pending:
        return {"path": path.name, "action": "still_pending", "pending": len(pending)}

    meta["exhaustive"] = True
    meta["draft"] = False
    meta["crop_promoted"] = True
    meta["crop_promote_reason"] = "wall sprite prep"
    meta["crop_promote_date"] = date.today().isoformat()
    session["metadata"] = meta
    path.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"path": path.name, "action": "promoted", "accepted": accepted, "image": name}


def assign_splits(promoted: list[dict]) -> list[tuple[dict, str]]:
    # Deterministic: sort by path, then 60/20/20
    ordered = sorted(promoted, key=lambda r: r["path"])
    out = []
    for index, row in enumerate(ordered):
        mod = index % 5
        split = "train" if mod in (0, 1, 2) else "val" if mod == 3 else "test"
        out.append((row, split))
    return out


def import_one(session_path: Path, image_path: Path, split: str, dry_run: bool) -> dict:
    cmd = [str(PY), str(IMPORT), str(session_path), str(image_path), "--split", split]
    if dry_run:
        cmd.append("--dry-run")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    try:
        body = json.loads(proc.stdout)
    except json.JSONDecodeError:
        body = {"ok": False, "errors": [proc.stdout or proc.stderr or f"exit {proc.returncode}"]}
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-report",
        type=Path,
        default=TRAINING / "reports" / f"wall-sprite-prep-{date.today().isoformat().replace('-', '')}.json",
    )
    args = parser.parse_args()

    repaired = []
    promoted = []
    for path in sorted(PRIVATE_WALL.glob("*.session.json")):
        row = repair_and_promote(path)
        repaired.append(row)
        if row.get("action") == "promoted":
            promoted.append(row | {"session": str(path)})

    imports = []
    totals = Counter()
    for row, split in assign_splits(promoted):
        session_path = Path(row["session"])
        image_path = session_path.with_name(row["image"])
        result = import_one(session_path, image_path, split, args.dry_run)
        imports.append({"session": row["path"], "split": split, **result})
        totals[f"attempt_{split}"] += 1
        totals["ok" if result.get("ok") else "fail"] += 1
        if not result.get("ok"):
            err = "; ".join(result.get("errors") or [])
            if "déjà présente" in err:
                totals["duplicate"] += 1

    payload = {
        "updated": date.today().isoformat(),
        "dry_run": args.dry_run,
        "finetune": False,
        "policy": {
            "sprites_only": True,
            "village_wall_only_not_imported": True,
            "note": (
                "Sprites mur OK pour val/test. Villages peters mur-only non importés "
                "(poison multi-classe). Pas de fine-tune."
            ),
        },
        "repair_totals": dict(Counter(r["action"] for r in repaired)),
        "promoted_count": len(promoted),
        "import_totals": dict(totals),
        "imports": imports,
        "repaired": repaired,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "repair_totals": payload["repair_totals"],
                "promoted_count": len(promoted),
                "import_totals": dict(totals),
                "report": str(args.output_report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
