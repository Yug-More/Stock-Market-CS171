"""Compile the manuscript in a temporary directory; record successful outputs only.

Requires Tectonic on PATH (or --compiler) and pypdf. Visually inspect each new PDF;
this script records compilation, not automatic approval of page layout.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--compiler", default="tectonic")
args = parser.parse_args()
version = subprocess.check_output([args.compiler, "--version"], text=True).strip()
with tempfile.TemporaryDirectory(prefix="marketpulse-tex-") as temporary:
    command = [
        args.compiler,
        "--keep-logs",
        "--keep-intermediates",
        "--outdir",
        temporary,
        "MarketPulse_Research_Paper.tex",
    ]
    result = subprocess.run(
        command, cwd=root / "paper", text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(
            f"PDF compile failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
        )
    pdf = Path(temporary) / "MarketPulse_Research_Paper.pdf"
    pages = len(PdfReader(pdf).pages)
    shutil.copyfile(pdf, root / "paper/MarketPulse_Research_Paper.pdf")
    (root / "paper/evidence/tectonic_console.txt").write_text(result.stdout + result.stderr)
paths = [
    "paper/MarketPulse_Research_Paper.tex",
    "paper/references.bib",
    "paper/MarketPulse_Research_Paper.pdf",
    "paper/evidence/tectonic_console.txt",
]
paths += [str(p.relative_to(root)) for p in sorted((root / "paper/figures").glob("*"))]
record = {
    "compiler": version,
    "exit_code": result.returncode,
    "pages": pages,
    "command": (
        "tectonic --keep-logs --keep-intermediates --outdir TEMP MarketPulse_Research_Paper.tex"
    ),
    "files": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
    "visual_review": "Required separately for this PDF; see phase review report",
    "scope": "Successful compilation and source hashes, not external scientific review",
}
(root / "paper/evidence/pdf_build.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"status": "compiled", "pages": pages, "visual_review": "separate"}))
