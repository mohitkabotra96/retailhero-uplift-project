from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CLIENTS_PATH = RAW_DIR / "clients.parquet"
PRODUCTS_PATH = RAW_DIR / "products.parquet"
PURCHASES_PATH = RAW_DIR / "purchases.parquet"

OUTPUT_PATH = PROCESSED_DIR / "customer_features.parquet"

# Cutoff = max(transaction_datetime) - CUTOFF_DAYS: esclude dalle
# feature di MODELING le transazioni successive, per evitare leakage.
CUTOFF_DAYS = 14


def main() -> None:
    con = duckdb.connect(database=":memory:")

    print("Registrazione parquet in DuckDB...")
    con.execute(f"CREATE VIEW clients AS SELECT * FROM read_parquet('{CLIENTS_PATH.as_posix()}')")
    con.execute(f"CREATE VIEW products AS SELECT * FROM read_parquet('{PRODUCTS_PATH.as_posix()}')")
    con.execute(f"CREATE VIEW purchases AS SELECT * FROM read_parquet('{PURCHASES_PATH.as_posix()}')")


    date_range = con.execute(
        """
        SELECT
            MIN(transaction_datetime) AS min_dt,
            MAX(transaction_datetime) AS max_dt,
            COUNT(DISTINCT transaction_id) AS n_transactions
        FROM purchases
        """
    ).fetchdf()
    print("\n=== Diagnostica temporale purchases ===")
    print(date_range.to_string(index=False))

    max_dt = pd.to_datetime(date_range["max_dt"].iloc[0])
    cutoff_date = max_dt - pd.Timedelta(days=CUTOFF_DAYS)
    print(f"\nCutoff anti-leakage: max_dt - {CUTOFF_DAYS}gg = {cutoff_date}")
    print("Feature di MODELING calcolate solo su transazioni <= cutoff.")
    print("LTV proxy calcolato sull'intero storico (full window).")
    print("==============================================================\n")


    con.execute(
        """
        CREATE VIEW transactions_dedup AS
        SELECT DISTINCT
            client_id,
            transaction_id,
            transaction_datetime,
            purchase_sum,
            regular_points_received,
            express_points_received,
            regular_points_spent,
            express_points_spent
        FROM purchases
        """
    )

    sanity = con.execute(
        "SELECT COUNT(*) AS n_dedup_rows FROM transactions_dedup"
    ).fetchdf()
    n_distinct_tx = date_range["n_transactions"].iloc[0]
    print(
        f"Sanity check dedup: {sanity['n_dedup_rows'].iloc[0]:,} righe dopo dedup "
        f"vs {n_distinct_tx:,} transaction_id distinti attesi "
        f"(devono coincidere se purchase_sum/punti sono davvero costanti per scontrino)."
    )


    con.execute(
        f"""
        CREATE VIEW client_transaction_agg_modeling AS
        SELECT
            client_id,
            COUNT(*)                              AS n_transactions_pre,
            SUM(purchase_sum)                      AS total_revenue_pre,
            AVG(purchase_sum)                      AS avg_basket_value_pre,
            MIN(transaction_datetime)               AS first_purchase_dt_pre,
            MAX(transaction_datetime)               AS last_purchase_dt_pre,
            SUM(regular_points_received)            AS total_regular_points_received_pre,
            SUM(express_points_received)            AS total_express_points_received_pre,
            SUM(regular_points_spent)               AS total_regular_points_spent_pre,
            SUM(express_points_spent)               AS total_express_points_spent_pre
        FROM transactions_dedup
        WHERE transaction_datetime <= '{cutoff_date}'
        GROUP BY client_id
        """
    )


    con.execute(
        """
        CREATE VIEW client_transaction_agg_full AS
        SELECT
            client_id,
            COUNT(*)         AS n_transactions_full,
            SUM(purchase_sum) AS total_revenue_full,
            MIN(transaction_datetime) AS first_purchase_dt_full,
            MAX(transaction_datetime) AS last_purchase_dt_full
        FROM transactions_dedup
        GROUP BY client_id
        """
    )

    con.execute(
        f"""
        CREATE VIEW client_product_agg AS
        SELECT
            p.client_id,
            COUNT(*)                                       AS n_product_lines,
            COUNT(DISTINCT p.product_id)                    AS n_distinct_products,
            SUM(p.product_quantity)                          AS total_quantity,
            SUM(p.trn_sum_from_iss)                           AS total_iss_sum,
            AVG(CAST(pr.is_alcohol AS DOUBLE))                 AS share_alcohol_lines,
            AVG(CAST(pr.is_own_trademark AS DOUBLE))            AS share_own_trademark_lines
        FROM purchases p
        LEFT JOIN products pr ON p.product_id = pr.product_id
        WHERE p.transaction_datetime <= '{cutoff_date}'
        GROUP BY p.client_id
        """
    )


    features = con.execute(
        """
        SELECT
            c.client_id,
            c.age,
            c.gender,
            c.first_issue_date,
            c.first_redeem_date,

            -- feature di MODELING (pre-cutoff, no leakage)
            tm.n_transactions_pre,
            tm.total_revenue_pre,
            tm.avg_basket_value_pre,
            tm.first_purchase_dt_pre,
            tm.last_purchase_dt_pre,
            tm.total_regular_points_received_pre,
            tm.total_express_points_received_pre,
            tm.total_regular_points_spent_pre,
            tm.total_express_points_spent_pre,
            pa.n_product_lines,
            pa.n_distinct_products,
            pa.total_quantity,
            pa.total_iss_sum,
            pa.share_alcohol_lines,
            pa.share_own_trademark_lines,

            -- aggregati FULL WINDOW (solo per LTV, non per modeling)
            tf.n_transactions_full,
            tf.total_revenue_full,
            tf.first_purchase_dt_full,
            tf.last_purchase_dt_full

        FROM clients c
        LEFT JOIN client_transaction_agg_modeling tm ON c.client_id = tm.client_id
        LEFT JOIN client_product_agg pa ON c.client_id = pa.client_id
        LEFT JOIN client_transaction_agg_full tf ON c.client_id = tf.client_id
        """
    ).fetchdf()

    print(f"\nFeature table costruita: {features.shape[0]:,} righe x {features.shape[1]} colonne")

 
    for col in [
        "first_issue_date", "first_redeem_date",
        "first_purchase_dt_pre", "last_purchase_dt_pre",
        "first_purchase_dt_full", "last_purchase_dt_full",
    ]:
        features[col] = pd.to_datetime(features[col])


    features["tenure_days"] = (
        features["last_purchase_dt_pre"] - features["first_issue_date"]
    ).dt.days

   
    features["recency_days_pre"] = (
        cutoff_date - features["last_purchase_dt_pre"]
    ).dt.days


    features["has_redeemed"] = features["first_redeem_date"].notna().astype(int)


    features["ltv_proxy"] = features["total_revenue_full"].fillna(0)

    n_no_purchases_pre = features["n_transactions_pre"].isna().sum()
    n_no_purchases_full = features["n_transactions_full"].isna().sum()
    print(f"Clienti senza transazioni PRE-cutoff (rischio per modeling): {n_no_purchases_pre:,}")
    print(f"Clienti senza transazioni in tutto il periodo (per LTV): {n_no_purchases_full:,}")

    features.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSalvato: {OUTPUT_PATH}")
    print(features.dtypes)
    print("\nPrime righe:")
    print(features.head(5).to_string())


if __name__ == "__main__":
    main()
