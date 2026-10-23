"""Package a verified complete study without altering original prediction bytes."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "results/walkforward/full"
    verification = json.loads((root / "verification.json").read_text())
    if verification["validated_folds"] != verification["planned_folds"]:
        raise ValueError("verify the complete study before packaging")
    with (root / "predictions.csv").open("rb") as source:
        with (root / "predictions.csv.gz").open("wb") as dest:
            with gzip.GzipFile(fileobj=dest, mode="wb", filename="", mtime=0) as compressed:
                shutil.copyfileobj(source, compressed)
    with (root / "fold_predictions.tar.gz").open("wb") as dest:
        with gzip.GzipFile(fileobj=dest, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                for path in sorted(root.glob("folds/*/*/*.csv")):
                    info = archive.gettarinfo(str(path), arcname=str(path.relative_to(root)))
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as source:
                        archive.addfile(info, source)
    with tarfile.open(root / "fold_predictions.tar.gz", "r:gz") as archive:
        for member in archive.getmembers():
            source = archive.extractfile(member)
            if source is None or source.read() != (root / member.name).read_bytes():
                raise AssertionError(f"archive does not preserve {member.name}")
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "artifact_checksums.json"
        and not (
            path.suffix == ".csv" and ("folds" in path.parts or path.name == "predictions.csv")
        )
    ]
    checksums = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }
    (root / "artifact_checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    print(f"Packaged exact fold bytes and checksummed {len(checksums)} artifacts")


if __name__ == "__main__":
    main()
