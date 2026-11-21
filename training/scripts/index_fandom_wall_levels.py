"""Indexe les sprites murs Fandom (Wall/Home_Village) en inbox privée.

- Source: https://clashofclans.fandom.com/wiki/Wall/Home_Village (API MediaWiki)
- Destination: training/inbox/private/reference-only/wall-fandom/ (images gitignorées)
- Usage: référence visuelle locale seulement — PAS d'import detector, PAS de train
- Licence: contenu Fandom / assets jeu = copyright Supercell → reference-only

Exemple:
  .\\.venv\\Scripts\\python.exe training\\scripts\\index_fandom_wall_levels.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
DEFAULT_OUT = TRAINING / "inbox" / "private" / "reference-only" / "wall-fandom"
FANDOM_PAGE = "Wall/Home_Village"
FANDOM_PAGE_URL = "https://clashofclans.fandom.com/wiki/Wall/Home_Village"
API = "https://clashofclans.fandom.com/api.php"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
WALL_FILE_RE = re.compile(r"^File:Wall(\d+)\.png$", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def api_get(params: dict) -> dict:
    query = urllib.parse.urlencode({**params, "format": "json"})
    request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def list_wall_files() -> list[dict]:
    payload = api_get(
        {
            "action": "query",
            "generator": "images",
            "titles": FANDOM_PAGE,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|sha1|timestamp",
            "gimlimit": "100",
        }
    )
    pages = (payload.get("query") or {}).get("pages") or {}
    rows: list[dict] = []
    for page in pages.values():
        title = str(page.get("title") or "")
        match = WALL_FILE_RE.match(title)
        if not match:
            continue
        info = (page.get("imageinfo") or [{}])[0]
        rows.append(
            {
                "level": int(match.group(1)),
                "title": title,
                "url": info.get("url"),
                "width": info.get("width"),
                "height": info.get("height"),
                "bytes": info.get("size"),
                "sha1": info.get("sha1"),
                "mime": info.get("mime"),
                "timestamp": info.get("timestamp"),
            }
        )
    rows.sort(key=lambda row: row["level"])
    return rows


def download(url: str, dest: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            "Referer": "https://clashofclans.fandom.com/",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        dest.write_bytes(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-level", type=int, default=20, help="Inclure Wall1..WallN")
    parser.add_argument("--min-level", type=int, default=1)
    parser.add_argument(
        "--report",
        type=Path,
        default=TRAINING
        / "reports"
        / f"wall-fandom-index-{date.today().isoformat().replace('-', '')}.json",
    )
    args = parser.parse_args()

    out_dir = args.output if args.output.is_absolute() else ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".gitkeep").touch(exist_ok=True)

    listed = list_wall_files()
    selected = [
        row
        for row in listed
        if args.min_level <= int(row["level"]) <= args.max_level and row.get("url")
    ]
    if not selected:
        raise SystemExit("Aucune image WallN.png trouvée via l'API Fandom")

    levels: list[dict] = []
    downloaded = 0
    skipped = 0
    for row in selected:
        level = int(row["level"])
        filename = f"Wall{level}.png"
        dest = out_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
        else:
            download(str(row["url"]), dest)
            downloaded += 1
        digest = sha256_file(dest)
        provenance = {
            "source_page": FANDOM_PAGE_URL,
            "source_file": row["title"],
            "source_url": row["url"],
            "level": level,
            "license": "Supercell copyright — Fandom fair-use display; reference-only local copy",
            "usage": "reference_only_visual",
            "publish": False,
            "train": False,
            "detector_import": False,
            "sha256": digest,
            "bytes": dest.stat().st_size,
            "width": row.get("width"),
            "height": row.get("height"),
            "notes": [
                "Ne pas republier / ne pas committer l'image.",
                "Référence visuelle pour revue auto-annotator / annotator humain.",
                "Hors dataset/detector.",
            ],
        }
        (out_dir / f"{filename}.provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        levels.append(
            {
                "level": level,
                "file": filename,
                "path": str(dest.relative_to(ROOT)).replace("\\", "/"),
                "sha256": digest,
                "width": row.get("width"),
                "height": row.get("height"),
                "source_url": row["url"],
                "th19_adjacent": level >= 19,
            }
        )

    index = {
        "updated": date.today().isoformat(),
        "source_page": FANDOM_PAGE_URL,
        "purpose": "Référence visuelle niveaux de mur (Home Village) — hors train",
        "publish_images": False,
        "train_now": False,
        "detector_import": False,
        "output": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "levels": levels,
        "level_count": len(levels),
        "downloaded": downloaded,
        "skipped_existing": skipped,
        "auto_annotate_hook": {
            "env_or_flag": "--wall-fandom-index",
            "default_index": str((out_dir / "index.json").relative_to(ROOT)).replace(
                "\\", "/"
            ),
        },
    }
    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Rapport versionnable (sans binaires) : chemins + métadonnées
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "index": str((out_dir / "index.json").relative_to(ROOT)).replace("\\", "/"),
                "report": str(report_path),
                "level_count": len(levels),
                "downloaded": downloaded,
                "skipped_existing": skipped,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
