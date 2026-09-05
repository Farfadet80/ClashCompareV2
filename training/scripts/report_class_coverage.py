"""Rapport de couverture YOLO vs catalogue ClashCompare (pas de TEST réservé)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"


def names_list(value: list[str] | dict[int | str, str]) -> list[str]:
    if isinstance(value, list):
        return value
    return [value[key] for key in sorted(value, key=lambda item: int(item))]


def count_split(split: str, n_classes: int) -> Counter:
    counts: Counter = Counter()
    label_dir = TRAINING / "dataset" / "detector" / "labels" / split
    for label in label_dir.glob("*.txt"):
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            class_id = int(parts[0])
            if 0 <= class_id < n_classes:
                counts[class_id] += 1
    return counts


def main() -> None:
    names = names_list(yaml.safe_load((TRAINING / "dataset.yaml").read_text(encoding="utf-8"))["names"])
    train = count_split("train", len(names))
    val = count_split("val", len(names))
    test = count_split("test", len(names))
    rows = []
    for class_id, name in enumerate(names):
        t, v, te = train[class_id], val[class_id], test[class_id]
        if t == 0:
            status = "unlearnable"
        elif t < 50:
            status = "very_weak"
        elif t < 200:
            status = "weak"
        else:
            status = "ok"
        if v == 0 and te == 0:
            status = "no_eval_split" if status == "ok" else status
        rows.append(
            {
                "id": name,
                "class_id": class_id,
                "train": t,
                "val": v,
                "test": te,
                "status": status,
            }
        )
    payload = {
        "classes": len(names),
        "unlearnable": [row["id"] for row in rows if row["status"] == "unlearnable"],
        "very_weak": [row["id"] for row in rows if row["status"] == "very_weak"],
        "weak": [row["id"] for row in rows if row["status"] == "weak"],
        "no_eval_split": [row["id"] for row in rows if row["val"] == 0 and row["test"] == 0],
        "rows": rows,
        "notes": [
            "town-hall-guardian : 0 annotation — curation visuelle seulement pour l'instant.",
            "wall : quasi absent des datasets publics (12 boîtes) — export JSON prioritaire.",
            "find-this-base importé (37 images train) ; autres sources locales absentes du disque (déjà fusionnées ou à re-télécharger).",
            "Source mur vérifiée : coc-wall-detection, 50 images CC BY 4.0 ; téléchargement Roboflow avec connexion requis.",
            "Bake-off VAL 2026-09-05 : politique dual-conf rejetée → conf 0.25 conservée.",
        ],
    }
    out = TRAINING / "reports" / "class-coverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("classes", "unlearnable", "very_weak", "weak", "no_eval_split")}, ensure_ascii=False, indent=2))
    print(f"Rapport: {out}")


if __name__ == "__main__":
    main()
