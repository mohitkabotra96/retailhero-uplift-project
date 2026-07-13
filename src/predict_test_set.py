from __future__ import annotations

from pathlib import Path

import joblib
import json
import pandas as pd

from train_uplift_models import MODELS_DIR, PROCESSED_DIR, prepare_X

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_PATH = PROCESSED_DIR / "test_modeling.parquet"
OUTPUT_PATH = REPORTS_DIR / "test_set_predictions.csv"


def main() -> None:
    test_df = pd.read_parquet(TEST_PATH)
    print(f"test_modeling: {test_df.shape}")

    model_treatment = joblib.load(MODELS_DIR / "uplift_random_forest_treatment.joblib")
    model_control = joblib.load(MODELS_DIR / "uplift_random_forest_control.joblib")

    with open(MODELS_DIR / "feature_columns.json") as f:
        meta = json.load(f)

    X_test = prepare_X(test_df, meta["gender_categories"])

    print("Predizione in corso...")
    p_treatment = model_treatment.predict_proba(X_test)[:, 1]
    p_control = model_control.predict_proba(X_test)[:, 1]
    uplift_score = p_treatment - p_control

    output = test_df[["client_id", "ltv_proxy", "ltv_cluster"]].copy()
    output["p_treatment"] = p_treatment
    output["p_control"] = p_control
    output["uplift_score"] = uplift_score
    output["raccomandazione"] = output["uplift_score"].apply(
        lambda s: "Includi" if s > 0 else "Escludi"
    )

    n_include = (output["raccomandazione"] == "Includi").sum()
    n_exclude = (output["raccomandazione"] == "Escludi").sum()
    print(f"\nRaccomandazione 'Includi': {n_include:,} ({n_include/len(output)*100:.1f}%)")
    print(f"Raccomandazione 'Escludi': {n_exclude:,} ({n_exclude/len(output)*100:.1f}%)")

    print("\nDistribuzione uplift_score:")
    print(output["uplift_score"].describe())

    print("\nDistribuzione cluster LTV tra i clienti raccomandati 'Includi':")
    print(output[output["raccomandazione"] == "Includi"]["ltv_cluster"].value_counts())

    output = output.sort_values("uplift_score", ascending=False)
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSalvato: {OUTPUT_PATH}")

    print("\nTop 10 clienti per uplift_score:")
    print(output.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
