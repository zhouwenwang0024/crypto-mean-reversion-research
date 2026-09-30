"""Run the predeclared neutral peer-price formation study."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neutral_peer_models import MODEL_NAMES, build_neutral_models  # noqa: E402
from validate_5m_mean_reversion import SYMBOLS, load_prices  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def main() -> None:
    index, _, close, _ = load_prices()
    selected, diagnostics = build_neutral_models(
        close, index=index, symbols=SYMBOLS, fit_days=60, cal_days=28,
        formation_start="2024-06-01", formation_end="2026-09-01",
        models=MODEL_NAMES, alpha=.1, hourly=True,
    )
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    selected_path = results / "neutral_peer_selected.csv"
    diagnostics_path = results / "neutral_peer_diagnostics.csv.gz"
    selected.to_csv(selected_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False, compression="gzip")
    summary = (diagnostics.groupby("model", as_index=False)
               .agg(target_months=("target", "size"), eligible=("eligible", "sum"),
                    optimization_failures=("optimization_success", lambda x: int((~x).sum())),
                    median_r2=("cal_r2", "median"), median_adf=("resid_adf_pvalue", "median"),
                    median_half_life=("resid_half_life_hours", "median"),
                    median_sensitivity=("sensitivity_pred_rmse_log", "median")))
    summary_path = results / "neutral_peer_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "source": "data/combined_5m_2024-02_to_2026-08.zip",
        "fit_days": 60, "calibration_days": 28, "hourly_formation": True,
        "formation_period": "2024-06-01/2026-09-01", "models": list(MODEL_NAMES),
        "alpha": .1, "selection": "OOS calibration: R2>=0, ADF lag1 p<=.10, monthly BH q<=.10 across 100 target-model hypotheses, half-life 6-168h, scale ratio<=2.5",
        "diagnostic_rows": int(len(diagnostics)), "selected_rows": int(len(selected)),
        "optimization_failures": int((~diagnostics.optimization_success).sum()) if len(diagnostics) else 0,
        "files": {p.name: _sha256(p) for p in (selected_path, diagnostics_path, summary_path)},
    }
    manifest_path = results / "neutral_peer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))
    if manifest["optimization_failures"]:
        print("optimization failures retained in diagnostics:", manifest["optimization_failures"])


if __name__ == "__main__":
    main()
