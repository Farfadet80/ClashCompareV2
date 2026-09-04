"""Lance le détecteur après validation du dataset."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from ultralytics import YOLO


TRAINING = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--name", default="building-detector")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", default="0")
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--cls-pw", type=float, default=0.0)
    parser.add_argument("--distill-model", type=Path, help="Modèle professeur pour la distillation")
    parser.add_argument("--allow-small-dataset", action="store_true")
    parser.add_argument(
        "--allow-split-overlap",
        action="store_true",
        help=(
            "Utilise le train brut même s'il chevauche val/test. "
            "Dangereux : réservé à un diagnostic explicite."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Valide et matérialise la config assainie sans lancer d'entraînement.",
    )
    parser.add_argument("--resume", type=Path, help="Checkpoint last.pt à reprendre")
    parser.add_argument("--reuse-run-name", action="store_true", help="Autorise explicitement un dossier de run existant")
    args = parser.parse_args()

    if args.resume:
        checkpoint = args.resume.resolve()
        if not checkpoint.exists():
            raise SystemExit(f"Checkpoint absent: {checkpoint}")
        YOLO(str(checkpoint)).train(resume=True)
        return

    validation = [sys.executable, str(TRAINING / "scripts" / "validate_dataset.py")]
    if args.allow_small_dataset:
        validation += ["--min-train", "1", "--min-val", "1"]
    subprocess.run(validation, check=True)

    # Ultralytics resolves relative dataset paths from its global datasets_dir,
    # not from the YAML file. Materialize a local runtime config so the project
    # keeps working regardless of where the repository is installed.
    source_config = TRAINING / "dataset.yaml"
    runtime_config = TRAINING / "runs" / "dataset.resolved.yaml"
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    config["path"] = str((TRAINING / "dataset" / "detector").resolve())
    if not args.allow_split_overlap:
        clean_train_list = TRAINING / "dataset" / "detector" / "train-clean.txt"
        audit = [
            sys.executable,
            str(TRAINING / "scripts" / "audit_split_integrity.py"),
            "--yaml",
            str(source_config),
            "--clean-train-list",
            str(clean_train_list),
        ]
        subprocess.run(audit, check=True)
        config["train"] = str(clean_train_list.resolve())
        print(f"Train assaini utilisé: {clean_train_list}")
    else:
        print("ATTENTION: chevauchements train/val/test explicitement autorisés.")
    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if args.prepare_only:
        print(f"Configuration prête: {runtime_config}")
        return

    run_dir = TRAINING / "runs" / args.name
    if run_dir.exists() and not args.reuse_run_name:
        raise SystemExit(
            f"Le run existe déjà: {run_dir}. Choisir --name différent ou utiliser "
            "--reuse-run-name en connaissance de cause."
        )

    model = YOLO(args.model)
    teacher = str(args.distill_model.resolve()) if args.distill_model else None
    if args.distill_model and not args.distill_model.exists():
        raise SystemExit(f"Modèle professeur absent: {args.distill_model}")
    model.train(
        data=str(runtime_config),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(TRAINING / "runs"),
        name=args.name,
        exist_ok=args.reuse_run_name,
        patience=args.patience,
        optimizer=args.optimizer,
        cos_lr=args.cos_lr,
        save_period=args.save_period,
        cls_pw=args.cls_pw,
        distill_model=teacher,
    )


if __name__ == "__main__":
    main()
