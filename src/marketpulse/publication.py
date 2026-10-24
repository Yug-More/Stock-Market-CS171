"""Generate a manuscript from explicit evidence bindings; never mutate experiments."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd

from marketpulse.data.catalog import load_catalog
from marketpulse.data.loader import load_market_csv
from marketpulse.experiments.benchmark import environment_record
from marketpulse.experiments.final_synthesis import verify_claim
from marketpulse.experiments.recovery import file_hash
from marketpulse.experiments.robustness_data import write_json

MODELS = ["zero_return", "historical_mean", "linear_regression", "random_forest", "lstm"]
LABELS = dict(zip(MODELS, ["Zero", "Mean", "Linear", "RF", "LSTM"], strict=True))


def table(number: int, caption: str, headers: list[str], rows: list[list]) -> str:
    def row(values):
        return "| " + " | ".join(map(str, values)) + " |"

    return "\n".join(
        [
            f"**Table {number}. {caption}**",
            "",
            row(headers),
            row(["---"] * len(headers)),
            *map(row, rows),
        ]
    )


def sources(root: Path) -> dict[str, list[str]]:
    return {
        "history": ["docs/REPOSITORY_AUDIT.md", "docs/RECOVERY.md", "data/README.md"],
        "data": ["configs/datasets.toml", "data/nasdaq100.csv", "data/SPX.csv", "data/README.md"],
        "protocols": [
            f"configs/{p}"
            for p in [
                "benchmark.toml",
                "forecasting.toml",
                "robustness.toml",
                "phase10.json",
                "phase11.json",
                "independent_validation.json",
            ]
        ],
        "loader": ["src/marketpulse/data/loader.py", "src/marketpulse/data/validation.py"],
        "features": [
            "src/marketpulse/features.py",
            "src/marketpulse/targets.py",
            "src/marketpulse/sequences.py",
        ],
        "models": [
            "src/marketpulse/experiments/benchmark.py",
            "src/marketpulse/models/baselines.py",
            "src/marketpulse/models/classical.py",
            "src/marketpulse/models/lstm.py",
        ],
        "metrics": [
            "src/marketpulse/evaluation/metrics.py",
            "src/marketpulse/experiments/summary.py",
        ],
        "folds": [
            "src/marketpulse/evaluation/partitions.py",
            "results/walkforward/full/fold_plan.json",
        ],
        "regimes": [
            "src/marketpulse/evaluation/regimes.py",
            "results/walkforward/full/stress_metrics.csv",
            "results/walkforward/full/regime_metrics.csv",
        ],
        "statistics": [
            "src/marketpulse/evaluation/reliability.py",
            "docs/STATISTICAL_RELIABILITY.md",
        ],
        "calibration": [
            "src/marketpulse/evaluation/calibration.py",
            "src/marketpulse/experiments/calibration_experiment.py",
        ],
        "adaptive": [
            "src/marketpulse/evaluation/adaptive_calibration.py",
            "src/marketpulse/experiments/adaptive_study.py",
        ],
        "diagnostics": ["scripts/phase11_diagnostics.py", "results/phase11/diagnostics.csv"],
        "prospective": [
            "src/marketpulse/experiments/prospective_intake.py",
            "docs/PHASE9_INDEPENDENT_VALIDATION.md",
        ],
        "tests": [str(p.relative_to(root)) for p in sorted((root / "tests").glob("test_*.py"))],
        "manuscript": [
            "paper/manuscript_source.md",
            "scripts/manuscript.py",
            "src/marketpulse/publication.py",
        ],
    }


def render(root: Path) -> tuple[str, str, dict]:
    claims = json.loads((root / "results/phase12/claims.json").read_text())
    for claim in claims:
        verify_claim(root, claim)
    values = {c["id"]: c["value"] for c in claims if not isinstance(c["value"], list)}
    native = pd.read_csv(root / "results/walkforward/full/pooled_metrics.csv")
    primary = pd.read_csv(root / "results/statistical_reliability/comparisons.csv")
    primary = primary[primary.primary_family]
    assert len(native) == 30 and len(primary) == 18
    assert native[native.model == "lstm"]["rank"].eq(5).all()
    assert len(native[native["rank"].eq(1) & native.model.eq("random_forest")]) == 3
    rows = []
    for (dataset, horizon), frame in native.groupby(["dataset", "horizon"]):
        rows.append(
            [
                dataset,
                horizon,
                int(frame.n.iloc[0]),
                *[f"{frame[frame.model == model].rmse.iloc[0]:.6f}" for model in MODELS],
            ]
        )
    values["POINT_TABLE"] = table(
        1,
        "Native pooled RMSE (log-return units); all procedures.",
        ["Index", "h", "n", "Zero", "Mean", "Linear", "RF", "LSTM"],
        rows,
    )
    rows = []
    for r in primary.itertuples():
        rows.append(
            [
                r.dataset,
                r.horizon,
                f"{LABELS[r.first]}/{LABELS[r.second]}",
                f"{r.mean_loss_difference:.4g}",
                f"[{r.ci_lower:.4g}, {r.ci_upper:.4g}]",
                f"{r.p_value:.4g}",
                f"{r.holm_p_value:.4g}",
            ]
        )
    values["INFERENCE_TABLE"] = table(
        2,
        "Original primary MSE effects: first minus second; "
        "marginal block CIs and separate p-values.",
        ["Index", "h", "Pair", "Delta MSE", "95% block CI", "p", "Holm p"],
        rows,
    )
    for prefix, dataset, horizon, second in [
        ("RF_NDX1", "NASDAQ-100", 1, "zero_return"),
        ("RF_SP20", "S&P 500", 20, "historical_mean"),
    ]:
        r = primary[
            (primary.dataset == dataset) & (primary.horizon == horizon) & (primary.second == second)
        ].iloc[0]
        values.update(
            {
                prefix + "_EFFECT": f"{r.mean_loss_difference:.8g}",
                prefix + "_CI": f"[{r.ci_lower:.8g}, {r.ci_upper:.8g}]",
                prefix + "_P": f"{r.p_value:.6g}",
                prefix + "_ADJ": f"{r.holm_p_value:.6g}",
            }
        )
    seeds = pd.read_csv(root / "results/phase10/seed_metrics.csv")
    rows = []
    for (dataset, horizon), frame in seeds.groupby(["dataset", "horizon"]):
        for model in ["random_forest", "lstm"]:
            part = frame[frame.model == model]
            assert part.seed.nunique() == 5 and part.n.eq(1260).all()
            rows.append(
                [
                    dataset,
                    horizon,
                    LABELS[model],
                    f"{part.rmse.min():.6f}",
                    f"{part.rmse.max():.6f}",
                    f"{int(part['rank'].min())}--{int(part['rank'].max())}",
                ]
            )
    values["SEED_TABLE"] = table(
        3,
        "Expanded selected-fold stochastic sensitivity: "
        "min/max across fixed seeds, not confidence intervals.",
        ["Index", "h", "Model", "RMSE min", "RMSE max", "Rank range"],
        rows,
    )
    metrics = pd.read_csv(root / "results/phase11/all_methods_metrics.csv")
    selected = metrics[
        metrics.model.eq("zero_return") & metrics.nominal.eq(0.9) & metrics.slice.eq("overall")
    ]
    rows = []
    for (dataset, horizon), frame in selected.groupby(["dataset", "horizon"]):
        for method, label in [
            ("trailing", "Trailing"),
            ("aci_w504_g001", "Adaptive"),
            ("volatility_scaled", "Vol-scaled"),
        ]:
            r = frame[frame.method == method].iloc[0]
            rows.append(
                [
                    dataset,
                    horizon,
                    label,
                    int(r.n),
                    f"{r.coverage:.6f}",
                    f"{r.mean_width:.6f}",
                    f"{r.interval_score:.6f}",
                ]
            )
    values["CALIBRATION_TABLE"] = table(
        4,
        "Zero-return 90% intervals: same eligible origins, "
        "coverage fraction and log-return width/score.",
        ["Index", "h", "Method", "n", "Coverage", "Width", "Score"],
        rows,
    )
    for method, key in [("aci_w504_g001", "SP20_COVERAGE"), ("trailing", "SP20_TRAILING_COVERAGE")]:
        cell = selected[
            selected.dataset.eq("S&P 500") & selected.horizon.eq(20) & selected.method.eq(method)
        ]
        values[key] = f"{cell.coverage.iloc[0]:.6f}"
    diagnostics = pd.read_csv(root / "results/phase11/diagnostics.csv")
    rows = []
    for variant, part in diagnostics.groupby("variant"):
        window = 504 if "504" in variant else 252
        rate = ".01" if "g001" in variant else ".05"
        rows.append(
            [
                window,
                rate,
                int(part.eligible.sum()),
                int(part.nesting_violations.sum()),
                int(part.clipped_80.sum()),
                int(part.clipped_90.sum()),
            ]
        )
    values["NESTING_TABLE"] = table(
        5,
        "Adaptive failure diagnostics; repeated method-origin pairs and clipped update counts.",
        ["Window", "Rate", "Eligible", "Crossings", "Clipped 80", "Clipped 90"],
        rows,
    )
    comparison = pd.read_csv(root / "results/phase11/score_comparisons.csv")
    r = comparison[
        comparison.primary
        & comparison.dataset.eq("NASDAQ-100")
        & comparison.horizon.eq(1)
        & comparison.model.eq("historical_mean")
        & comparison.nominal.eq(0.8)
    ].iloc[0]
    values["ACI_NOMINAL_P"], values["ACI_ADJUSTED_P"] = f"{r.p_value:.6g}", f"{r.holm_p:.6g}"
    rows = []
    for entry in load_catalog(root):
        frame = load_market_csv(root / entry.path, dataset_name=entry.name)
        rows.append(
            [
                entry.name,
                entry.index_id,
                len(frame),
                str(frame.date.iloc[0].date()),
                str(frame.date.iloc[-1].date()),
                entry.path,
            ]
        )
    values["DATASETS"] = "\n".join(
        [
            "| Index | Identifier | Rows | First date | Last date | File |",
            "| --- | --- | --- | --- | --- | --- |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )
    references = json.loads((root / "paper/evidence/reference_verification.json").read_text())
    values["REFERENCES"] = "\n\n".join(
        f"- **{r['key']}**: {r['authors']} ({r['year']}). {r['title']}. "
        f"{r['publication']}. {r['verification_url']}"
        for r in references
    )
    text = (root / "paper/manuscript_source.md").read_text()
    for key, value in values.items():
        if isinstance(value, (int, float)):
            value = str(int(value)) if int(value) == value else str(value)
        text = text.replace("{{" + key + "}}", str(value))
    if re.search(r"\{\{[^}]+\}\}", text):
        raise ValueError("unresolved manuscript evidence binding")
    evidence = sources(root)
    bindings = re.findall(r"<!-- evidence: (.*?) -->", text)
    if not bindings:
        raise ValueError("manuscript has no machine-readable evidence bindings")
    bound_claims: set[str] = set()
    bound_groups: set[str] = set()
    for binding in bindings:
        for item in binding.split(";"):
            item = item.strip()
            if item.startswith("E:"):
                bound_groups.add(item[2:])
            elif re.fullmatch(r"C\d{2}(?:,C\d{2})*", item):
                bound_claims.update(item.split(","))
            else:
                raise ValueError(f"invalid evidence binding: {item}")
    claim_ids = {claim["id"] for claim in claims}
    if not bound_claims <= claim_ids:
        raise ValueError(f"unknown bound claim IDs: {sorted(bound_claims - claim_ids)}")
    if not bound_groups <= evidence.keys():
        raise ValueError(f"unknown evidence groups: {sorted(bound_groups - evidence.keys())}")
    # Bindings stay in the authoritative source and manifest, not reader-facing prose.
    text = re.sub(r"\s*<!-- evidence: .*? -->", "", text)
    cited = set(re.findall(r"\[@(\w+)\]", text))
    assert cited == {r["key"] for r in references}
    bib = (root / "paper/references.bib").read_text()
    entries = {key: body for key, body in re.findall(r"(?ms)^@\w+\{(\w+),(.*?)(?=^@|\Z)", bib)}
    assert set(entries) == cited
    for reference in references:
        body = entries[reference["key"]]
        assert re.search(r"year\s*=\s*\{" + str(reference["year"]) + r"\}", body)
        assert not reference.get("doi") or reference["doi"] in body
        assert reference["title"].lower() in body.lower()
    inputs = {c["artifact"]: c["sha256"] for c in claims}
    for paths in evidence.values():
        inputs.update({p: file_hash(root / p) for p in paths})
    for p in [
        "paper/references.bib",
        "paper/evidence/reference_verification.json",
        "results/phase12/claims.json",
        "results/statistical_reliability/comparisons.csv",
        "results/phase11/score_comparisons.csv",
    ]:
        inputs[p] = file_hash(root / p)
    for path in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
        target = (root / "paper" / path).resolve()
        inputs[str(target.relative_to(root))] = file_hash(target)
        if path.startswith("figures/") and path.endswith(".png"):
            vector = target.with_suffix(".pdf")
            inputs[str(vector.relative_to(root))] = file_hash(vector)
    return (
        text,
        latex(text),
        {
            "inputs": inputs,
            "evidence_groups": evidence,
            "claim_ids": sorted(bound_claims),
            "bound_evidence_groups": sorted(bound_groups),
            "citation_keys": sorted(cited),
            "precision": "tables are display-rounded; exact saved CSVs are authoritative",
            "scope": (
                "executable traceability and arithmetic bindings; "
                "not automated semantic proof or external peer review"
            ),
        },
    )


def inline(text: str) -> str:
    patterns = r"(\$[^$]+\$|`[^`]+`|\[@\w+\]|\*\*[^*]+\*\*)"

    def escape(value):
        for old, new in [
            ("\\", r"\textbackslash{}"),
            ("&", r"\&"),
            ("%", r"\%"),
            ("_", r"\_"),
            ("#", r"\#"),
            ("{", r"\{"),
            ("}", r"\}"),
            ("^", r"\textasciicircum{}"),
            ("~", r"\textasciitilde{}"),
        ]:
            value = value.replace(old, new)
        return value

    out = []
    for token in re.split(patterns, text):
        if token.startswith("$") and token.endswith("$"):
            out.append(token)
        elif token.startswith("`"):
            out.append(r"\path{" + token[1:-1] + "}")
        elif token.startswith("[@"):
            out.append(r"\citep{" + token[2:-1] + "}")
        elif token.startswith("**"):
            out.append(r"\textbf{" + escape(token[2:-2]) + "}")
        else:
            out.append(escape(html.unescape(token)))
    return "".join(out)


def latex(text: str) -> str:
    lines = text.splitlines()
    title = lines[0].removeprefix("# ")
    output = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb,graphicx,longtable,booktabs,natbib,url,needspace}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\setlength{\emergencystretch}{3em}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\title{" + inline(title) + "}",
        r"\author{Yug More\\\texttt{yugmore20@gmail.com}}",
        r"\date{October 2026\\Unpublished research draft}",
        r"\begin{document}",
        r"\maketitle",
    ]
    i, caption = 2, None
    if lines[2].startswith("Yug More"):
        i = 3
    while i < len(lines):
        line = lines[i]
        if line.strip() == "$$":
            output.append(r"\[")
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                output.append(lines[i])
                i += 1
            if i == len(lines):
                raise ValueError("unclosed display-math block")
            output.append(r"\]")
            i += 1
            continue
        elif line.startswith("## References"):
            output += [r"\bibliographystyle{plainnat}", r"\bibliography{references}"]
            i += 1
            while i < len(lines) and not lines[i].startswith("## "):
                i += 1
            continue
        if line.startswith("```"):
            output.append(r"\begin{verbatim}")
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                output.append(lines[i])
                i += 1
            output.append(r"\end{verbatim}")
        elif line.startswith("**Table "):
            caption = re.sub(r"^\*\*Table \d+\. |\*\*$", "", line)
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", c) for c in cells):
                    rows.append(cells)
                i += 1
            output.append(r"\Needspace{" + str(35 + 12 * len(rows)) + "pt}")
            if caption is None:
                output += [
                    r"\begin{center}{\footnotesize\begin{tabular}{" + "l" * len(rows[0]) + "}",
                    r"\toprule",
                    " & ".join(inline(c) for c in rows[0]) + r"\\",
                    r"\midrule",
                    *[" & ".join(inline(c) for c in row) + r"\\" for row in rows[1:]],
                    r"\bottomrule\end{tabular}}\end{center}",
                ]
                continue
            output += [r"{\footnotesize", r"\begin{longtable}{" + "l" * len(rows[0]) + "}"]
            if caption:
                output.append(r"\caption{" + inline(caption) + r"}\\")
                caption = None
            header = " & ".join(inline(c) for c in rows[0]) + r"\\"
            output += [
                r"\toprule",
                header,
                r"\midrule\endfirsthead",
                r"\toprule",
                header,
                r"\midrule\endhead",
            ]
            output += [" & ".join(inline(c) for c in row) + r"\\" for row in rows[1:]]
            output += [r"\bottomrule\end{longtable}}"]
            continue
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[(.*)\]\((.*)\)", line)
            assert match
            output += [
                r"\begin{figure}[htbp]\centering",
                r"\includegraphics[width=\textwidth]{"
                + (match[2][:-4] + ".pdf" if match[2].startswith("figures/") else match[2])
                + "}",
                r"\caption{" + inline(match[1]) + "}",
                r"\end{figure}",
            ]
        elif line.startswith("### "):
            output.append(r"\subsection*{" + inline(line[4:]) + "}")
        elif line.startswith("## "):
            output.append(r"\section*{" + inline(line[3:]) + "}")
        else:
            output.append(inline(line))
        i += 1
    return "\n".join([*output, r"\end{document}", ""])


def build(root: Path) -> None:
    markdown, tex, metadata = render(root)
    (root / "paper/MarketPulse_Research_Paper.md").write_text(markdown)
    (root / "paper/MarketPulse_Research_Paper.tex").write_text(tex)
    metadata["environment"] = environment_record(root, "manuscript_authoring")
    metadata["outputs"] = {
        p: file_hash(root / p)
        for p in ["paper/MarketPulse_Research_Paper.md", "paper/MarketPulse_Research_Paper.tex"]
    }
    write_json(root / "paper/evidence/manuscript_manifest.json", metadata)
    verify(root)


def verify(root: Path) -> None:
    metadata = json.loads((root / "paper/evidence/manuscript_manifest.json").read_text())
    for path, sha in {**metadata["inputs"], **metadata["outputs"]}.items():
        if file_hash(root / path) != sha:
            raise ValueError(f"manuscript input/output changed: {path}")
    markdown, tex, _ = render(root)
    assert markdown == (root / "paper/MarketPulse_Research_Paper.md").read_text()
    assert tex == (root / "paper/MarketPulse_Research_Paper.tex").read_text()
    for generated in (markdown, tex):
        if re.search(r"\bC\d{2}\b|E:[a-z]+", generated):
            raise ValueError("internal evidence identifiers leaked into reader-facing manuscript")
    print(
        json.dumps(
            {
                "status": "passed",
                "cited_claims": len(metadata["claim_ids"]),
                "verified_references": len(metadata["citation_keys"]),
                "independent_dataset_validation": False,
            }
        )
    )
