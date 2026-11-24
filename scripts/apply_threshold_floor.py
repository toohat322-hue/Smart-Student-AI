"""Phase 4c — apply the threshold floor to the saved models WITHOUT retraining.

cut-off = max(THRESHOLD_FLOOR, raw tuned cut-off). The forests are not touched; only the
FixedThresholdClassifier.threshold changes. Metrics are updated exactly:
  * a floored model's metrics at the final cut-off are its stored metrics at 0.5,
  * permutation importance (depends on the cut-off) is recomputed from the saved model,
  * every copied number for the saved models is re-checked against a live evaluation.
Idempotent: running it twice gives the same result. Future full training applies the same
rule in src/supervised.tune_threshold, so scripts/train_supervised.py reproduces this state.

Usage:
    python scripts/apply_threshold_floor.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import train_supervised as ts  # noqa: E402  (figure functions + rule text)
from src.data import TARGET_COL  # noqa: E402
from src.metrics_io import load_metrics, update_metrics  # noqa: E402
from src.preprocessing import SCENARIOS, feature_columns  # noqa: E402
from src.supervised import (  # noqa: E402
    DEFAULT_THRESHOLD, THRESHOLD_FLOOR, evaluate, fairness_check, model_path, permutation_top,
    prepare_subject, split_indices,
)

COMPARE_KEYS = ("accuracy", "f1_macro", "at_risk_recall", "at_risk_precision", "false_alerts", "roc_auc")


def no_curves(t: dict) -> dict:
    return {k: v for k, v in t.items() if k != "roc_curve"}


def migrate_model_entry(m: dict) -> tuple[float, float]:
    """Rewrite threshold_tuning + test_tuned of one model entry. Returns (raw, final)."""
    tt = m["threshold_tuning"]
    raw = tt.get("threshold_raw", tt["threshold"])
    final = max(THRESHOLD_FLOOR, raw)
    raw_recall = tt.get("oof_at_risk_recall_raw", tt["oof_at_risk_recall"])
    m["threshold_tuning"] = {
        "threshold": final,
        "threshold_raw": raw,
        "threshold_floor": THRESHOLD_FLOOR,
        "target_at_risk_recall": tt["target_at_risk_recall"],
        "oof_at_risk_recall": tt["oof_at_risk_recall_at_default"] if final != raw else raw_recall,
        "oof_at_risk_recall_raw": raw_recall,
        "oof_at_risk_precision_raw": tt.get("oof_at_risk_precision_raw", tt.get("oof_at_risk_precision")),
        "oof_f1_macro_raw": tt.get("oof_f1_macro_raw", tt.get("oof_f1_macro")),
        "oof_at_risk_recall_at_default": tt["oof_at_risk_recall_at_default"],
    }
    if final != raw:
        m["test_tuned"] = no_curves(copy.deepcopy(m["test_default"]))
    return raw, final


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    metrics = load_metrics()
    sup = metrics["supervised"]
    sup["config"]["threshold_rule"] = ts.THRESHOLD_RULE
    sup["config"]["threshold_floor"] = THRESHOLD_FLOOR

    for s in ts.SUBJECT_ORDER:
        df = prepare_subject(s)
        _, test_idx = split_indices(df)
        for sc in SCENARIOS:
            e = sup["subjects"][s][sc]
            raw, final = migrate_model_entry(e["models"]["rf"])
            migrate_model_entry(e["models"]["rf_unconstrained"])
            migrate_model_entry(e["sensitivity_excl_g3_zero"]["rf_excl_g3_zero"])
            if final != raw:
                sens_main = e["sensitivity_excl_g3_zero"]["rf_main_on_test_excl_g3_zero"]
                sens_main["test_tuned"] = copy.deepcopy(sens_main["test_default"])
                e["fairness"]["rf_tuned"] = copy.deepcopy(e["fairness"]["rf_default"])
            e["threshold"] = final

            # Saved model: change only the cut-off, then verify against a live evaluation.
            path = model_path(s, sc)
            model = joblib.load(path)
            model.set_params(threshold=final)
            joblib.dump(model, path)

            X = df[feature_columns(sc)]
            X_te, y_te = X.iloc[test_idx], df[TARGET_COL].iloc[test_idx]
            live = evaluate(model, X_te, y_te, final, include_curves=False)
            stored = e["models"]["rf"]["test_tuned"]
            mismatch = {k: (stored[k], live[k]) for k in COMPARE_KEYS if stored[k] != live[k]}
            assert not mismatch, f"{s}/{sc}: stored vs live mismatch {mismatch}"
            live_fair = fairness_check(model, X_te, y_te, df["sex"].iloc[test_idx], final)
            assert live_fair == e["fairness"]["rf_tuned"], f"{s}/{sc}: fairness mismatch"
            assert model.predict(X_te).tolist() == [int(p >= final) for p in model.predict_proba(X_te)[:, 1]]

            e["permutation_importance"] = permutation_top(model, X_te, y_te)
            print(f"{s}/{sc:8s} cut-off raw {raw:.3f} -> final {final:.3f}  "
                  f"{'(floored to default)' if final != raw else '(unchanged)'}  "
                  f"AR-recall {live['at_risk_recall']:.3f}  false alerts {live['false_alerts']}  [verified]")

    sup["figures"] = [ts.fig_confusion(sup), ts.fig_roc(sup), ts.fig_importance(sup), ts.fig_early_vs_midterm(sup),
                      ts.fig_absences_pd(sup), ts.fig_threshold_effect(sup)]
    update_metrics("supervised", sup)
    print(f"Default cut-off {DEFAULT_THRESHOLD}, floor {THRESHOLD_FLOOR}. Updated models/, metrics.json, figures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
