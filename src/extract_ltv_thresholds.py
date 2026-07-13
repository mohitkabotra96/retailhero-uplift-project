
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_LTV_PATH = PROCESSED_DIR / "customer_features_ltv.parquet"
OUTPUT_PATH = MODELS_DIR / "ltv_thresholds.json"

LTV_LEVELS = ["Basso", "Medio-basso", "Medio-alto", "Alto"]


def main() -> None:
    df = pd.read_parquet(FEATURES_LTV_PATH)

    q25, q50, q75 = df["ltv_proxy"].quantile([0.25, 0.5, 0.75]).values

    thresholds = {
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "levels": LTV_LEVELS,
        "note": (
            "Per classificare un nuovo cliente: ltv <= q25 -> Basso; "
            "<= q50 -> Medio-basso; <= q75 -> Medio-alto; altrimenti Alto."
        ),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"Soglie LTV: q25={q25:.2f}  q50={q50:.2f}  q75={q75:.2f}")
    def classify(v: float) -> str:
        if v <= q25:
            return "Basso"
        if v <= q50:
            return "Medio-basso"
        if v <= q75:
            return "Medio-alto"
        return "Alto"

    check = df["ltv_proxy"].apply(classify)
    print("\nSanity check (deve essere vicino a 100040 per gruppo):")
    print(check.value_counts())

    print(f"\nSalvato: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
