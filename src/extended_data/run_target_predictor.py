"""Build causal monthly peer-price prediction diagnostics on the 5m archive."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from target_predictor import build_target_models  # noqa: E402
from validate_5m_mean_reversion import SYMBOLS, load_prices  # noqa: E402


def main() -> None:
    index, _, close, _ = load_prices()
    selected, diagnostics = build_target_models(
        close, index=index, symbols=SYMBOLS, fit_days=60, cal_days=28,
        formation_start="2024-06-01", formation_end="2026-09-01",
        ridge_alpha=10.0, estimators=("ridge", "ols"), min_r2=0.0,
        adf_p=0.10, max_scale_ratio=2.5,
    )
    results = ROOT / "results"
    selected.to_csv(results / "target_predictor_selected.csv", index=False)
    diagnostics.to_csv(results / "target_predictor_diagnostics.csv", index=False)
    summary = (diagnostics.groupby("model", as_index=False)
               .agg(target_months=("target", "size"), eligible=("eligible", "sum"),
                    median_r2=("cal_r2", "median"), median_adf=("resid_adf_pvalue", "median"),
                    median_half_life=("resid_half_life_hours", "median"),
                    median_error_pct=("median_abs_error_pct", "median")))
    summary.to_csv(results / "target_predictor_summary.csv", index=False)
    target_summary = (diagnostics.groupby(["model", "target_symbol"], as_index=False)
                      .agg(target_months=("eligible", "size"), eligible=("eligible", "sum"),
                           median_r2=("cal_r2", "median"), median_adf=("resid_adf_pvalue", "median"),
                           median_half_life=("resid_half_life_hours", "median"),
                           median_error_pct=("median_abs_error_pct", "median")))
    target_summary.to_csv(results / "target_predictor_target_summary.csv", index=False)
    manifest = {
        "source": "data/combined_5m_2024-02_to_2026-08.zip",
        "fit_days": 60, "calibration_days": 28, "hourly_formation": True,
        "formation_period": "2024-06-01/2026-09-01", "estimators": ["ridge", "ols"],
        "ridge_alpha": 10.0, "selection": "OOS calibration only: R2>=0, raw ADF p<=0.10, within-month-model BH q<=0.10, residual half-life 6-168h, scale ratio<=2.5",
        "selected_rows": int(len(selected)), "diagnostic_rows": int(len(diagnostics)),
    }
    (results / "target_predictor_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
