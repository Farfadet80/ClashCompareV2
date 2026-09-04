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
PRODUCTION_LEVEL_CLASSIFIERS = ("air-defense", "town-hall")


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

    for building in PRODUCTION_LEVEL_CLASSIFIERS:
        weights = ROOT / "models" / f"level-{building}.pt"
        if not weights.exists():
            raise SystemExit(f"ERREUR: classifieur production absent: {weights}")
    if (ROOT / "models" / "level-cannon.pt").exists():
        raise SystemExit(
            "ERREUR: level-cannon.pt ne doit pas être en production "
            "(voir models/experimental/)"
        )

    print(f"Release active: {ACTIVE_RELEASE} (imgsz {EXPECTED_IMGSZ})")
    print(f"PT SHA-256: {pt_hash}")
    print(f"ONNX input: {shape}")
    print(f"Classifieurs production: {', '.join(PRODUCTION_LEVEL_CLASSIFIERS)}")


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
