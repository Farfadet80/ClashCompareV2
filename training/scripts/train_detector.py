"""Lance le détecteur après validation du dataset."""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from ultralytics import YOLO


TRAINING = Path(__file__).resolve().parents[1]


def materialize_student_weights(checkpoint: Path) -> Path:
    """Extrait l'élève d'un checkpoint complet de distillation Ultralytics.

    Les checkpoints périodiques V5 stockent ``ema`` comme DistillationModel :
    leurs clés commencent par ``student_model.`` et ne peuvent donc pas être
    chargées directement par ``YOLO(...).train()``. L'original reste intact ;
    seule une copie standard, sans optimizer, est créée pour le fine-tune.
    """

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ema = payload.get("ema") if isinstance(payload, dict) else None
    student = getattr(ema, "student_model", None)
    if student is None:
        return checkpoint

    output_dir = TRAINING / "runs" / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{checkpoint.parent.parent.name}-{checkpoint.stem}-student.pt"

    model = copy.deepcopy(student).half()
    for parameter in model.parameters():
        parameter.requires_grad = False
    normalized = {
        key: value
        for key, value in payload.items()
        if key not in {"model", "ema", "optimizer", "scaler"}
    }
    normalized.update(
        {
            "epoch": -1,
            "source_epoch": payload.get("epoch"),
            "model": model,
            "ema": None,
            "optimizer": None,
            "scaler": None,
        }
    )
    torch.save(normalized, output)
    print(
        f"Checkpoint distillé détecté (epoch {payload.get('epoch')}) : "
        f"poids élève matérialisés dans {output}"
    )
    return output


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
    parser.add_argument("--lr0", type=float, help="Taux d'apprentissage initial explicite")
    parser.add_argument("--lrf", type=float, help="Facteur LR final (lr0 × lrf)")
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-epochs", type=float, help="Durée du warmup en epochs")
    parser.add_argument("--warmup-bias-lr", type=float)
    parser.add_argument("--freeze", type=int, help="Nombre de premières couches à geler")
    parser.add_argument("--mosaic", type=float, help="Probabilité mosaic (0 désactive)")
    parser.add_argument("--close-mosaic", type=int, help="Désactive mosaic N epochs avant la fin")
    parser.add_argument("--scale", type=float, help="Amplitude de redimensionnement aléatoire")
    parser.add_argument("--erasing", type=float, help="Probabilité d'effacement aléatoire")
    parser.add_argument("--translate", type=float)
    parser.add_argument("--hsv-h", type=float)
    parser.add_argument("--hsv-s", type=float)
    parser.add_argument("--hsv-v", type=float)
    parser.add_argument("--fliplr", type=float)
    parser.add_argument("--flipud", type=float)
    parser.add_argument("--mixup", type=float)
    parser.add_argument("--cutmix", type=float)
    parser.add_argument("--copy-paste", type=float)
    parser.add_argument("--seed", type=int)
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

    model_path = Path(args.model)
    if model_path.suffix.lower() == ".pt" and model_path.exists():
        model_path = materialize_student_weights(model_path.resolve())
    model = YOLO(str(model_path))
    teacher = str(args.distill_model.resolve()) if args.distill_model else None
    if args.distill_model and not args.distill_model.exists():
        raise SystemExit(f"Modèle professeur absent: {args.distill_model}")
    if args.optimizer.lower() == "auto" and args.lr0 is not None:
        raise SystemExit(
            "--optimizer auto ignore --lr0. Choisir explicitement AdamW, Adam ou SGD."
        )

    train_options = {
        "data": str(runtime_config),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": 0,
        "project": str(TRAINING / "runs"),
        "name": args.name,
        "exist_ok": args.reuse_run_name,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "cos_lr": args.cos_lr,
        "save_period": args.save_period,
        "cls_pw": args.cls_pw,
        "distill_model": teacher,
    }
    optional_options = {
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "warmup_bias_lr": args.warmup_bias_lr,
        "freeze": args.freeze,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "scale": args.scale,
        "erasing": args.erasing,
        "translate": args.translate,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "fliplr": args.fliplr,
        "flipud": args.flipud,
        "mixup": args.mixup,
        "cutmix": args.cutmix,
        "copy_paste": args.copy_paste,
        "seed": args.seed,
    }
    train_options.update(
        {key: value for key, value in optional_options.items() if value is not None}
    )
    print(
        "Configuration fine-tune contrôlée: "
        + ", ".join(
            f"{key}={train_options[key]}"
            for key in optional_options
            if key in train_options
        )
    )
    model.train(**train_options)


if __name__ == "__main__":
    main()
