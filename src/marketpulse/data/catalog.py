"""Repository dataset catalog."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from marketpulse import __version__
from marketpulse.data.exceptions import MarketDataFormatError
from marketpulse.data.loader import load_market_csv
from marketpulse.data.validation import DatasetSummary, validate_market_frame

_DATASETS_FILE = Path("configs") / "datasets.toml"


@dataclass(frozen=True)
class CatalogEntry:
    """One raw dataset declared in the repository catalog."""

    name: str
    path: str
    index_id: str
    documented_source: str
    documented_limitations: tuple[str, ...]


@dataclass(frozen=True)
class CatalogResult:
    """A catalog entry paired with the measurement of its file."""

    entry: CatalogEntry
    summary: DatasetSummary


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root by walking parents for the dataset catalog."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / _DATASETS_FILE).is_file():
            return candidate
    raise MarketDataFormatError(
        "Could not locate configs/datasets.toml. Run this command from the repository root."
    )


def load_catalog(root: Path) -> tuple[CatalogEntry, ...]:
    """Read the raw-dataset catalog. Paths stay relative to ``root``."""

    catalog_path = root / _DATASETS_FILE
    if not catalog_path.is_file():
        raise MarketDataFormatError(f"Dataset catalog does not exist: {catalog_path}")
    with catalog_path.open("rb") as handle:
        payload = tomllib.load(handle)
    entries = payload.get("dataset")
    if not isinstance(entries, list) or not entries:
        raise MarketDataFormatError(f"{catalog_path} does not declare any datasets")

    parsed: list[CatalogEntry] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise MarketDataFormatError(f"Invalid dataset entry in {catalog_path}")
        limitations = entry.get("documented_limitations", [])
        if not isinstance(limitations, list) or not all(isinstance(item, str) for item in limitations):
            raise MarketDataFormatError(
                f"documented_limitations for {entry.get('name', '<unknown>')} must be a list of strings"
            )
        parsed.append(
            CatalogEntry(
                name=_required_text(entry, "name"),
                path=_required_text(entry, "path"),
                index_id=_required_text(entry, "index_id"),
                documented_source=_required_text(entry, "documented_source"),
                documented_limitations=tuple(limitations),
            )
        )
    return tuple(parsed)


def validate_catalog(root: Path | None = None) -> tuple[CatalogResult, ...]:
    """Load and validate every dataset declared in the catalog."""

    project_root = find_project_root(root)
    results: list[CatalogResult] = []
    for entry in load_catalog(project_root):
        frame = load_market_csv(project_root / entry.path, dataset_name=entry.name)
        summary = validate_market_frame(frame, entry.name)
        results.append(CatalogResult(entry, summary))
    return tuple(results)


def build_catalog_report(root: Path | None = None) -> dict[str, object]:
    """Validate every catalog dataset and return a JSON-ready report.

    Documented source text is copied from the catalog. Measured fields come
    only from the files themselves.
    """

    datasets: list[dict[str, object]] = []
    for result in validate_catalog(root):
        entry = result.entry
        datasets.append(
            {
                "name": entry.name,
                "index_id": entry.index_id,
                "path": entry.path,
                "documented_source": entry.documented_source,
                "documented_limitations": list(entry.documented_limitations),
                "summary": result.summary.to_dict(),
            }
        )
    return {"package_version": __version__, "datasets": datasets}


def _required_text(entry: dict[str, object], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MarketDataFormatError(f"Dataset catalog entry is missing a non-empty '{key}'")
    return value
