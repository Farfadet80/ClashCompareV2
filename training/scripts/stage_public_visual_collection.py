"""Prépare l'inbox autonome de collecte visuelle (sans captures Jimmy).

Ne télécharge pas d'assets copyrightés dans le train. Crée l'arborescence,
écrit les sidecars de provenance, et résume le manifeste public.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
MANIFEST = TRAINING / "sources" / "visual-collection-manifest.json"
INBOX = TRAINING / "inbox" / "public-reference"
REFERENCE = TRAINING / "reference-only" / "th19-visual"
CLASSES = TRAINING / "classes.json"
DATASET_YAML = TRAINING / "dataset.yaml"


def load_yaml_names() -> list[str]:
    names: list[str] = []
    in_names = False
    for raw in DATASET_YAML.read_text(encoding="utf-8").splitlines():
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
        names.append(name.strip())
    return names


def ensure_class_dir(base: Path, created: list[str]) -> None:
    marker = base / ".gitkeep"
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        created.append(str(base.relative_to(ROOT)))
    if not marker.exists():
        marker.write_text("", encoding="utf-8")
        created.append(str(marker.relative_to(ROOT)))


def ensure_dirs(inbox_ids: list[str], reference_ids: list[str]) -> list[str]:
    created: list[str] = []
    for class_id in inbox_ids:
        ensure_class_dir(INBOX / class_id, created)
    for class_id in reference_ids:
        ensure_class_dir(REFERENCE / class_id, created)
    readme = INBOX / "README.md"
    if not readme.exists():
        raise SystemExit(f"README manquant: {readme}")
    return created


def write_source_index(manifest: dict) -> Path:
    rows = []
    for source in manifest.get("sources", []):
        rows.append(
            {
                "id": source.get("id"),
                "url": source.get("url"),
                "license": source.get("license"),
                "usage": source.get("usage"),
                "status": source.get("status"),
                "targets": source.get("targets", []),
            }
        )
    out = INBOX / "SOURCES.index.json"
    payload = {
        "updated": date.today().isoformat(),
        "policy": manifest.get("policy", {}),
        "sources": rows,
        "note": (
            "Les médias reference_only restent hors dataset/detector. "
            "Les imports CC BY passent par audit_public_dataset.py puis revue exhaustive."
        ),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def write_class_sidecar_template(class_id: str) -> Path:
    path = INBOX / class_id / "_provenance.template.json"
    payload = {
        "village_group": f"public-{class_id}-TODO",
        "class_focus": class_id,
        "town_hall": None,
        "capture_mode": "public-gameplay-or-reference",
        "license": "TODO — CC BY 4.0 / CC BY-SA / consentement auteur",
        "source_url": "TODO",
        "author": "TODO",
        "usage": "train_candidate_after_audit",
        "levels_verified": False,
        "consent": False,
        "notes": [
            "Remplir avant annotation. Ne jamais inventer de niveaux.",
            "Annotation exhaustive via training/annotator/ obligatoire avant import.",
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Vérifie le schéma classes sans écrire l'inbox",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    class_payload = json.loads(CLASSES.read_text(encoding="utf-8"))
    detector_ids = [row["id"] for row in class_payload["detector_classes"]]
    yaml_names = load_yaml_names()
    if detector_ids != yaml_names:
        raise SystemExit(
            "Désync classes.json / dataset.yaml : "
            f"{len(detector_ids)} vs {len(yaml_names)}"
        )

    reserved = manifest["target_classes"]["reserved_th19_yolo"]
    missing = [name for name in reserved if name not in detector_ids]
    if missing:
        raise SystemExit(f"Classes TH19 absentes du schéma YOLO : {missing}")

    targets = sorted(
        {
            *manifest["target_classes"]["reserved_th19_yolo"],
            *manifest["target_classes"]["unlearnable_or_empty"],
            *manifest["target_classes"]["very_weak"],
            *manifest["target_classes"]["weak"],
        }
    )

    summary = {
        "detector_class_count": len(detector_ids),
        "reserved_th19": reserved,
        "inbox_targets": targets,
        "sources": len(manifest.get("sources", [])),
        "train_now": bool(manifest.get("policy", {}).get("train_now")),
        "jimmy_captures": manifest.get("policy", {}).get("jimmy_captures"),
    }

    if args.check_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    created = ensure_dirs(targets, reserved)
    # Nettoyage : anciennes cibles faibles créées par erreur sous th19-visual.
    for path in REFERENCE.glob("*"):
        if path.is_dir() and path.name not in reserved:
            keep = path / ".gitkeep"
            if keep.exists() and keep.stat().st_size == 0 and not any(
                child for child in path.iterdir() if child.name != ".gitkeep"
            ):
                keep.unlink(missing_ok=True)
                path.rmdir()
                created.append(f"removed:{path.relative_to(ROOT)}")
    index_path = write_source_index(manifest)
    templates = [str(write_class_sidecar_template(cid).relative_to(ROOT)) for cid in targets]
    summary["created_or_touched"] = created
    summary["source_index"] = str(index_path.relative_to(ROOT))
    summary["provenance_templates"] = templates
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
