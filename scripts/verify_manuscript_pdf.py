"""Check recorded PDF/source hashes; visual review is separately documented."""

import json
from pathlib import Path

from marketpulse.experiments.recovery import file_hash

root = Path(__file__).resolve().parents[1]
record = json.loads((root / "paper/evidence/pdf_build.json").read_text())
for path, expected in record["files"].items():
    if file_hash(root / path) != expected:
        raise ValueError(f"PDF build input/output changed: {path}")
assert (root / "paper/MarketPulse_Research_Paper.pdf").read_bytes().startswith(b"%PDF-")
print(json.dumps({"status": "passed", "check": "PDF build hashes", "pages": record["pages"]}))
