"""Check declared primary comparison Cartesian products, not only family row counts."""

import argparse
import hashlib
import itertools
import json
import tomllib
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
if args.output and args.output.exists():
    raise ValueError("preserve previous family report")
paths = [
    "configs/robustness.toml",
    "configs/phase10.json",
    "configs/phase11.json",
    "results/statistical_reliability/comparisons.csv",
    "results/phase10/comparisons.csv",
    "results/phase11/score_comparisons.csv",
    "scripts/verify_comparison_families.py",
]
cells = list(itertools.product(["NASDAQ-100", "S&P 500"], [1, 5, 20]))
original = tomllib.loads((root / paths[0]).read_text())
expanded = json.loads((root / paths[1]).read_text())
adaptive = json.loads((root / paths[2]).read_text())
for config, source, flag in [
    (original, paths[3], "primary_family"),
    (expanded, paths[4], "primary"),
]:
    table = pd.read_csv(root / source)
    primary = table[table[flag]]
    expected = {(d, h, a, b) for d, h in cells for a, b in config["primary_pairs"]}
    actual = list(
        primary[["dataset", "horizon", "first", "second"]].itertuples(index=False, name=None)
    )
    assert set(actual) == expected and len(actual) == len(expected)
    assert primary.block_length.eq(20).all()
    assert (
        primary["population" if flag == "primary_family" else "slice"]
        .eq("native" if flag == "primary_family" else "overall")
        .all()
    )
primary = pd.read_csv(root / paths[5])
primary = primary[primary.primary]
expected = {
    (d, h, m, level)
    for d, h in cells
    for m in ["zero_return", "historical_mean", "linear_regression", "random_forest", "lstm"]
    for level in adaptive["levels"]
}
actual = list(
    primary[["dataset", "horizon", "model", "nominal"]].itertuples(index=False, name=None)
)
assert set(actual) == expected and len(actual) == len(expected) == 60
assert primary.variant.eq(adaptive["primary_variant"]).all() and primary.block_length.eq(20).all()
report = {
    "status": "passed",
    "families": [18, 30, 60],
    "scope": "declared Cartesian products, unique contrasts and primary block lengths",
    "inputs": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
}
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "inputs"}))
