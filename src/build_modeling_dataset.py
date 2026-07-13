from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = PROCESSED_DIR / "customer_features.parquet"
UPLIFT_TRAIN_PATH = RAW_DIR / "uplift_train.parquet"
UPLIFT_TEST_PATH = RAW_DIR / "uplift_test.parquet"

OUT_FEATURES_LTV = PROCESSED_DIR / "customer_features_ltv.parquet"
OUT_TRAIN = PROCESSED_DIR / "train_modeling.parquet"
OUT_TEST = PROCESSED_DIR / "test_modeling.parquet"


MAX_TRANSACTION_DT = pd.Timestamp("2019-03-18 23:40:03")
CUTOFF_DAYS = 14
CUTOFF_DATE = MAX_TRANSACTION_DT - pd.Timedelta(days=CUTOFF_DAYS)

ZERO_FILL_COLS = [
    "n_transactions_pre",
    "total_revenue_pre",
    "avg_basket_value_pre",
    "total_regular_points_received_pre",
    "total_express_points_received_pre",
    "total_regular_points_spent_pre",
    "total_express_points_spent_pre",
    "n_product_lines",
    "n_distinct_products",
    "total_quantity",
    "total_iss_sum",
    "share_alcohol_lines",
    "share_own_trademark_lines",
]

LTV_LEVELS = ["Basso", "Medio-basso", "Medio-alto", "Alto"]


def add_ltv_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge ltv_cluster (4 livelli) calcolato su ltv_proxy via quartili."""
    df = df.copy()
    try:
        df["ltv_cluster"] = pd.qcut(
            df["ltv_proxy"], q=4, labels=LTV_LEVELS, duplicates="drop"
        )
        n_bins = df["ltv_cluster"].nunique()
        if n_bins < 4:
            print(
                f"ATTENZIONE: qcut ha prodotto solo {n_bins} bin invece di 4 "
                "(troppi valori duplicati ai bordi, es. molti clienti con "
                "ltv_proxy=0). Uso fallback su rank percentile."
            )
            raise ValueError("fallback")
    except ValueError:
        ranks = df["ltv_proxy"].rank(method="first", pct=True)
        df["ltv_cluster"] = pd.cut(
            ranks, bins=[0, 0.25, 0.5, 0.75, 1.0], labels=LTV_LEVELS, include_lowest=True
        )
    return df


def main() -> None:
    print(f"Cutoff usato (deve coincidere con build_features.py): {CUTOFF_DATE}")

    features = pd.read_parquet(FEATURES_PATH)
    uplift_train = pd.read_parquet(UPLIFT_TRAIN_PATH)
    uplift_test = pd.read_parquet(UPLIFT_TEST_PATH)

    print(f"customer_features: {features.shape}")
    print(f"uplift_train: {uplift_train.shape}")
    print(f"uplift_test: {uplift_test.shape}")

   
    features["first_issue_date"] = pd.to_datetime(features["first_issue_date"], errors="coerce")


    features["tenure_days"] = (CUTOFF_DATE - features["first_issue_date"]).dt.days


    features["is_new_customer"] = features["n_transactions_pre"].isna().astype(int)
    print(f"Clienti senza transazioni pre-cutoff: {features['is_new_customer'].sum():,}")


    for col in ZERO_FILL_COLS:
        features[col] = features[col].fillna(0)


    recency_sentinel = features["recency_days_pre"].max() + 30
    features["recency_days_pre"] = features["recency_days_pre"].fillna(recency_sentinel)
    print(f"Sentinella recency per nuovi clienti: {recency_sentinel:.0f} giorni")

  
    features = add_ltv_cluster(features)
    print("\nDistribuzione cluster LTV (su tutti i 400.162 clienti):")
    print(features["ltv_cluster"].value_counts().sort_index())

    features.to_parquet(OUT_FEATURES_LTV, index=False)
    print(f"\nSalvato: {OUT_FEATURES_LTV}")


    train_df = uplift_train.merge(features, on="client_id", how="left")
    missing_in_features = train_df["age"].isna().sum()
    print(
        f"\ntrain_modeling: {train_df.shape[0]:,} righe "
        f"({missing_in_features} senza match in customer_features, atteso 0)"
    )
    print("Distribuzione treatment_flg / target nel train:")
    print(train_df.groupby(["treatment_flg", "target"]).size())

    train_df.to_parquet(OUT_TRAIN, index=False)
    print(f"Salvato: {OUT_TRAIN}")

    test_df = uplift_test.merge(features, on="client_id", how="left")
    missing_in_features_test = test_df["age"].isna().sum()
    print(
        f"\ntest_modeling: {test_df.shape[0]:,} righe "
        f"({missing_in_features_test} senza match in customer_features, atteso 0)"
    )
    test_df.to_parquet(OUT_TEST, index=False)
    print(f"Salvato: {OUT_TEST}")

    print("\nColonne disponibili in train_modeling.parquet:")
    print(list(train_df.columns))


if __name__ == "__main__":
    main()
