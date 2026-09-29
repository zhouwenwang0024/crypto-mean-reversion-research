"""Download the same disjoint sample at the original one-minute frequency."""
try:
    from . import download_5m as _d
except ImportError:
    import download_5m as _d


if __name__ == "__main__":
    _d.OUT = _d.ROOT / "data" / "extended_1m"
    _d.INTERVAL = "1m"
    _d.STEP_MS = 60_000
    _d.main()
