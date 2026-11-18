"""Vérifie Python, PyTorch, CUDA, la release active et une inférence YOLO sur le GPU."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

ACTIVE_RELEASE = "building-detector-v5s-infer800"
EXPECTED_IMGSZ = 800
EXPECTED_PT_SHA256 = "866595fad39a5b7dfdf87076332faadc40a88bc55eae1b02f093d996362fb93d"
EXPERIMENTAL_LEVEL_CLASSIFIERS = {
    "air-defense": Path("models/level-air-defense.pt"),
    "wall": Path("models/experimental/level-wall.pt"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_release() -> None:
    active = ROOT / "models" / "ACTIVE.json"
    if not active.exists():
        raise SystemExit(f"ERREUR: marqueur absent: {active}")
    meta = json.loads(active.read_text(encoding="utf-8"))
    if meta.get("release") != ACTIVE_RELEASE:
        raise SystemExit(f"ERREUR: release active inattendue: {meta.get('release')}")
    if int(meta.get("imgsz", 0)) != EXPECTED_IMGSZ:
        raise SystemExit(f"ERREUR: imgsz active inattendu: {meta.get('imgsz')}")
    if meta.get("tta") is not True:
        raise SystemExit(f"ERREUR: TTA active attendue (true), reçu: {meta.get('tta')}")

    release_dir = ROOT / "models" / "releases" / ACTIVE_RELEASE
    pt_alias = ROOT / "models" / "building-detector.pt"
    onnx_alias = ROOT / "models" / "building-detector.onnx"
    pt_release = release_dir / f"{ACTIVE_RELEASE}.pt"
    onnx_release = release_dir / f"{ACTIVE_RELEASE}.onnx"
    for path in (pt_alias, onnx_alias, pt_release, onnx_release):
        if not path.exists():
            raise SystemExit(f"ERREUR: fichier manquant: {path}")

    pt_hash = sha256(pt_alias)
    if pt_hash != EXPECTED_PT_SHA256:
        raise SystemExit(f"ERREUR: SHA-256 PT alias inattendu: {pt_hash}")
    if pt_hash != sha256(pt_release):
        raise SystemExit("ERREUR: alias PT ≠ release PT")
    if sha256(onnx_alias) != sha256(onnx_release):
        raise SystemExit("ERREUR: alias ONNX ≠ release ONNX")

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_alias), providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    if list(shape) != [1, 3, EXPECTED_IMGSZ, EXPECTED_IMGSZ]:
        raise SystemExit(f"ERREUR: forme ONNX {shape}, attendu 1x3x{EXPECTED_IMGSZ}x{EXPECTED_IMGSZ}")

    for building, relative_path in EXPERIMENTAL_LEVEL_CLASSIFIERS.items():
        weights = ROOT / relative_path
        if not weights.exists():
            raise SystemExit(f"ERREUR: classifieur expérimental absent: {weights}")

    production_levels = meta.get("level_classifiers", [])
    level_releases = meta.get("level_classifier_releases", {})
    for building in production_levels:
        release_meta = level_releases.get(building)
        if not release_meta:
            raise SystemExit(f"ERREUR: métadonnées de release absentes pour {building}")
        level_pt = ROOT / "models" / f"level-{building}.pt"
        level_onnx = ROOT / "models" / f"level-{building}.onnx"
        level_release_dir = ROOT / "models" / "releases" / release_meta["release"]
        release_pt = level_release_dir / level_pt.name
        release_onnx = level_release_dir / level_onnx.name
        for path in (level_pt, level_onnx, release_pt, release_onnx):
            if not path.exists():
                raise SystemExit(f"ERREUR: classifieur de niveau absent: {path}")
        if sha256(level_pt) != release_meta["pt_sha256"]:
            raise SystemExit(f"ERREUR: SHA-256 PT inattendu pour {building}")
        if sha256(level_onnx) != release_meta["onnx_sha256"]:
            raise SystemExit(f"ERREUR: SHA-256 ONNX inattendu pour {building}")
        if sha256(level_pt) != sha256(release_pt) or sha256(level_onnx) != sha256(release_onnx):
            raise SystemExit(f"ERREUR: alias de niveau ≠ release pour {building}")
        level_session = ort.InferenceSession(
            str(level_onnx), providers=["CPUExecutionProvider"]
        )
        level_shape = level_session.get_inputs()[0].shape
        if list(level_shape) != [1, 3, 224, 224]:
            raise SystemExit(
                f"ERREUR: forme ONNX de {building}: {level_shape}, attendu 1x3x224x224"
            )
    if (ROOT / "models" / "level-cannon.pt").exists():
        raise SystemExit(
            "ERREUR: level-cannon.pt ne doit pas être en production "
            "(voir models/experimental/)"
        )

    print(f"Release active: {ACTIVE_RELEASE} (imgsz {EXPECTED_IMGSZ}, TTA on)")
    print(f"PT SHA-256: {pt_hash}")
    print(f"ONNX input: {shape}")
    print(
        "Classifieurs production: "
        + (", ".join(production_levels) if production_levels else "aucun")
    )
    print(f"Classifieurs expérimentaux: {', '.join(EXPERIMENTAL_LEVEL_CLASSIFIERS)}")


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA PyTorch: {torch.version.cuda}")
    print(f"Ultralytics: {ultralytics.__version__}")

    if not torch.cuda.is_available():
        raise SystemExit("ERREUR: CUDA n'est pas disponible dans PyTorch.")

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    architectures = torch.cuda.get_arch_list()
    expected_arch = f"sm_{capability[0]}{capability[1]}"
    print(f"GPU: {name}")
    print(f"Compute capability: {capability}")
    print(f"Architectures intégrées: {architectures}")
    if expected_arch not in architectures:
        raise SystemExit(f"ERREUR: PyTorch ne contient pas {expected_arch}.")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        a = torch.randn((512, 512), device="cuda")
        _ = a @ a
        torch.cuda.synchronize()

    check_release()

    model = YOLO("yolo11n.yaml")
    result = model.predict(
        source=np.zeros((320, 320, 3), dtype=np.uint8),
        imgsz=320,
        device=0,
        verbose=False,
    )[0]
    torch.cuda.synchronize()
    print(f"Inférence YOLO CUDA: OK ({result.orig_shape[0]}x{result.orig_shape[1]})")
    print(f"Mémoire GPU maximale: {torch.cuda.max_memory_allocated() / 1024**2:.1f} Mo")


if __name__ == "__main__":
    main()
