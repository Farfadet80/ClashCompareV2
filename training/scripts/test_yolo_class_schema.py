"""Vérifie l'alignement dataset.yaml ↔ classes.json et les réserves TH19."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
RESERVED = (
    "spiky-tower",
    "firecracker-tower",
    "guardian-shrine",
    "squad-barracks",
)


def yaml_names() -> list[str]:
    names: list[str] = []
    in_names = False
    for raw in (TRAINING / "dataset.yaml").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("names:"):
            in_names = True
            continue
        if not in_names:
            continue
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            break
        idx_text, name = line.split(":", 1)
        if not idx_text.isdigit():
            break
        assert int(idx_text) == len(names), f"Index YOLO non contigu: {idx_text}"
        names.append(name.strip())
    return names


def main() -> None:
    names = yaml_names()
    payload = json.loads((TRAINING / "classes.json").read_text(encoding="utf-8"))
    ids = [row["id"] for row in payload["detector_classes"]]
    assert ids == names, "classes.json et dataset.yaml divergent"
    assert len(names) >= 54, f"Attendu ≥54 classes (50 + 4 TH19), obtenu {len(names)}"
    for class_id in RESERVED:
        assert class_id in ids, f"Classe réservée absente: {class_id}"
        assert ids.index(class_id) >= 50, f"{class_id} doit être append-only (≥50)"

    mapping = json.loads((ROOT / "data" / "coc-export-mapping.json").read_text(encoding="utf-8"))
    pending = {row["id"]: row for row in mapping.get("pendingNewBuildings", [])}
    for class_id in RESERVED:
        assert class_id in pending, f"{class_id} doit rester dans pendingNewBuildings"
        assert pending[class_id].get("dataId") is None, "Ne pas inventer de dataId"

    manifest = json.loads(
        (TRAINING / "sources" / "visual-collection-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["policy"]["jimmy_captures"] == "not_required"
    assert manifest["policy"]["train_now"] is False
    assert set(manifest["target_classes"]["reserved_th19_yolo"]) == set(RESERVED)

    print(f"OK schéma YOLO: {len(names)} classes, réserves TH19 append-only, dataIds non inventés")


if __name__ == "__main__":
    main()
