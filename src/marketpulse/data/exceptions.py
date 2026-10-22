"""Exceptions raised while reading or validating market data."""


class MarketDataError(Exception):
    """Base error for market-data loading and validation."""


class MarketDataFormatError(MarketDataError):
    """The CSV cannot be parsed into a market table."""


class MissingColumnError(MarketDataError):
    """One or more required market columns are absent."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = list(missing)
        listed = ", ".join(self.missing)
        super().__init__(f"Missing required columns: {listed}")


class TimestampParseError(MarketDataError):
    """One or more date values are not ISO calendar dates."""

    def __init__(self, invalid_count: int, examples: list[str]) -> None:
        self.invalid_count = invalid_count
        self.examples = list(examples)
        preview = ", ".join(self.examples)
        super().__init__(
            f"Could not parse {invalid_count} timestamp(s) as YYYY-MM-DD. Examples: {preview}"
        )
