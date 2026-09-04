from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
catalog = json.loads((ROOT / "data" / "buildings.json").read_text(encoding="utf-8"))
out = ROOT / "training" / "dataset" / "levels"

for b in catalog["buildings"]:
    levels = b.get("levels", [])
    for level in levels:
        d = out / b["id"] / f"level-{level}"
        d.mkdir(parents=True, exist_ok=True)
        (d / ".gitkeep").touch()

print("Structure des niveaux créée dans", out)
