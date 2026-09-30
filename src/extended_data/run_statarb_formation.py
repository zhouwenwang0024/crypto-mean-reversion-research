"""Run the frozen causal monthly formation for the 5-minute archive."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from statarb_models import MODEL_NAMES, build_models  # noqa: E402
from validate_5m_mean_reversion import SYMBOLS, load_prices  # noqa: E402


def main() -> None:
    index, _, close, _ = load_prices()
    selected, diagnostics = build_models(
        close, index=index, symbols=SYMBOLS, fit_days=60, cal_days=28,
        formation_start="2024-06-01", formation_end="2026-09-01", fdr_q=0.10,
        max_selected=3,
    )
    results = ROOT / "results"
    selected.to_csv(results / "statarb_formation_selected.csv", index=False)
    diagnostics.to_csv(results / "statarb_formation_diagnostics.csv", index=False)
    summary = (diagnostics.groupby(["model_end_time", "model"], as_index=False)
               .agg(pairs=("a", "size"), eligible=("eligible", "sum"),
                    selected=("selected", "sum"),
                    median_adf_p=("adf_pvalue", "median"),
                    median_half_life_hours=("half_life_hours", "median")))
    summary.to_csv(results / "statarb_formation_summary.csv", index=False)
    manifest = {
        "source": "data/combined_5m_2024-02_to_2026-08.zip",
        "fit_days": 60, "calibration_days": 28, "hourly_formation": True,
        "formation_period": "2024-06-01/2026-09-01",
        "models": list(MODEL_NAMES), "fdr_q": 0.10,
        "max_selected_per_model_month": 3,
        "selection": "ADF Benjamini-Hochberg plus beta stability, AR half-life 6-168h, and split calibration recovery; no return selection",
        "selected_rows": int(len(selected)), "diagnostic_rows": int(len(diagnostics)),
    }
    (results / "statarb_formation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(summary.groupby("model", as_index=False)[["eligible", "selected"]].sum().to_string(index=False))


if __name__ == "__main__":
    main()
