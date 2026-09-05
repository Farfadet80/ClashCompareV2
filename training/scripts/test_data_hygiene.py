"""Tests sans dépendance externe des garde-fous d'import et d'annotation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "training" / "scripts"


def run(*args: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != expected:
        raise AssertionError(
            f"Code {result.returncode}, attendu {expected}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def make_source(root: Path) -> Path:
    source = root / "source"
    (source / "train" / "images").mkdir(parents=True)
    (source / "train" / "labels").mkdir(parents=True)
    (source / "data.yaml").write_text("names: [AD]\n", encoding="utf-8")
    (source / "train" / "images" / "sample.jpg").write_bytes(b"not-a-real-jpeg")
    (source / "train" / "labels" / "sample.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    return source


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="clashcompare-hygiene-") as temp:
        root = Path(temp)
        source = make_source(root)
        report = root / "audit.json"

        run(
            str(SCRIPTS / "audit_public_dataset.py"),
            str(source),
            "find-this-base",
            "--output",
            str(report),
        )
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["status"] == "manual_exhaustiveness_review_required"
        assert not payload["blockers"]
        assert payload["class_boxes"] == {"AD": 1}

        refused = run(
            str(SCRIPTS / "import_public_dataset.py"),
            str(source),
            "find-this-base",
            "--as-train",
            expected=1,
        )
        assert "--audit-report" in (refused.stdout + refused.stderr)

        run(
            str(SCRIPTS / "import_public_dataset.py"),
            str(source),
            "find-this-base",
            "--dry-run",
            "--as-train",
        )

        image = root / "owner.png"
        Image.new("RGB", (100, 80), "green").save(image)
        session = {
            "version": 1,
            "image": {"name": "owner.png", "width": 100, "height": 80},
            "metadata": {
                "village_group": "test-owner-th16",
                "source": "capture test",
                "license": "consentement local test",
                "exhaustive": True,
            },
            "expected_counts": {"town-hall": 1, "bob-hut": 1},
            "boxes": [
                {
                    "class_id": 0,
                    "class_name": "town-hall",
                    "x1": 10,
                    "y1": 10,
                    "x2": 50,
                    "y2": 60,
                    "status": "accepted",
                }
            ],
        }
        session_path = root / "session.json"
        session_path.write_text(json.dumps(session), encoding="utf-8")
        valid = run(
            str(SCRIPTS / "import_annotation_session.py"),
            str(session_path),
            str(image),
            "--dry-run",
        )
        imported = json.loads(valid.stdout)
        assert imported["ok"] is True
        assert imported["box_count"] == 1

        session["boxes"][0]["status"] = "pending"
        session_path.write_text(json.dumps(session), encoding="utf-8")
        refused_session = run(
            str(SCRIPTS / "import_annotation_session.py"),
            str(session_path),
            str(image),
            "--dry-run",
            expected=2,
        )
        assert "encore en attente" in refused_session.stdout

    print("OK garde-fous datasets publics et sessions humaines")


if __name__ == "__main__":
    main()
