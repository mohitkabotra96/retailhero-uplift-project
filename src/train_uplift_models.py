from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from uplift_metrics import qini_coefficient

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = PROCESSED_DIR / "train_modeling.parquet"
COMPARISON_REPORT = Path(__file__).resolve().parent.parent / "models_comparison.md"
VAL_PRED_PATH = PROCESSED_DIR / "val_predictions.parquet"

RANDOM_STATE = 42

FEATURE_COLS = [
    "age",
    "gender",  # categorica, codificata sotto
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
    "tenure_days",
    "recency_days_pre",
    "has_redeemed",
    "is_new_customer",
]


def prepare_X(df: pd.DataFrame, gender_categories: list[str]) -> pd.DataFrame:
    """One-hot encoding di gender, allineato sulle categorie viste in train."""
    X = df[FEATURE_COLS].copy()
    gender_dummies = pd.get_dummies(X["gender"], prefix="gender")
    # Allinea le colonne anche se in questo split manca una categoria
    for cat in gender_categories:
        col = f"gender_{cat}"
        if col not in gender_dummies.columns:
            gender_dummies[col] = 0
    gender_dummies = gender_dummies[[f"gender_{c}" for c in gender_categories]]

    X = X.drop(columns=["gender"]).join(gender_dummies)
    return X.astype(float)


def get_model(algo: str, kind: str):
    """kind: 'treatment' o 'control', solo per eventuale tuning differenziato futuro."""
    if algo == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    if algo == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=50,
            n_jobs=-1, random_state=RANDOM_STATE,
        )
    if algo == "xgboost":
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE,
        )
    raise ValueError(f"Algoritmo sconosciuto: {algo}")


def main() -> None:
    df = pd.read_parquet(TRAIN_PATH)
    print(f"train_modeling: {df.shape}")

    gender_categories = sorted(df["gender"].unique().tolist())
    print(f"Categorie gender: {gender_categories}")


    df["_strata"] = df["treatment_flg"].astype(str) + "_" + df["target"].astype(str)
    train_df, val_df = train_test_split(
        df, test_size=0.25, random_state=RANDOM_STATE, stratify=df["_strata"]
    )
    print(f"Train: {train_df.shape[0]:,} | Validation: {val_df.shape[0]:,}")

    X_train = prepare_X(train_df, gender_categories)
    X_val = prepare_X(val_df, gender_categories)

    feature_columns_final = X_train.columns.tolist()
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(
            {"feature_columns": feature_columns_final, "gender_categories": gender_categories},
            f, indent=2,
        )

    results = []
    best_qini = -np.inf
    best_algo = None

    for algo in ["logistic_regression", "random_forest", "xgboost"]:
        print(f"\n=== {algo} ===")


        train_t = train_df[train_df["treatment_flg"] == 1]
        X_train_t = X_train.loc[train_t.index]
        y_train_t = train_t["target"]

        model_treatment = get_model(algo, "treatment")
        model_treatment.fit(X_train_t, y_train_t)

        train_c = train_df[train_df["treatment_flg"] == 0]
        X_train_c = X_train.loc[train_c.index]
        y_train_c = train_c["target"]

        model_control = get_model(algo, "control")
        model_control.fit(X_train_c, y_train_c)


        val_t_mask = val_df["treatment_flg"] == 1
        val_c_mask = val_df["treatment_flg"] == 0

        auc_treatment = roc_auc_score(
            val_df.loc[val_t_mask, "target"],
            model_treatment.predict_proba(X_val.loc[val_t_mask])[:, 1],
        )
        auc_control = roc_auc_score(
            val_df.loc[val_c_mask, "target"],
            model_control.predict_proba(X_val.loc[val_c_mask])[:, 1],
        )


        p_treat_all = model_treatment.predict_proba(X_val)[:, 1]
        p_ctrl_all = model_control.predict_proba(X_val)[:, 1]
        uplift_score = p_treat_all - p_ctrl_all

        qini = qini_coefficient(
            uplift_score=uplift_score,
            target=val_df["target"].values,
            treatment=val_df["treatment_flg"].values,
        )

        print(f"AUC model_treatment: {auc_treatment:.4f}")
        print(f"AUC model_control:   {auc_control:.4f}")
        print(f"Qini coefficient:    {qini:.4f}")
        print(f"Uplift score medio:  {uplift_score.mean():.4f}")

        results.append({
            "algo": algo,
            "auc_treatment": auc_treatment,
            "auc_control": auc_control,
            "qini_coefficient": qini,
            "mean_uplift": uplift_score.mean(),
        })

        joblib.dump(model_treatment, MODELS_DIR / f"uplift_{algo}_treatment.joblib")
        joblib.dump(model_control, MODELS_DIR / f"uplift_{algo}_control.joblib")

        if qini > best_qini:
            best_qini = qini
            best_algo = algo

            val_predictions = val_df[["client_id", "treatment_flg", "target"]].copy()
            val_predictions["uplift_score"] = uplift_score
            val_predictions["p_treatment"] = p_treat_all
            val_predictions["p_control"] = p_ctrl_all

    val_predictions.to_parquet(VAL_PRED_PATH, index=False)
    print(f"\nSalvato: {VAL_PRED_PATH}")

    results_df = pd.DataFrame(results).sort_values("qini_coefficient", ascending=False)
    print("\n=== Confronto algoritmi (ordinati per Qini) ===")
    print(results_df.to_string(index=False))

    print(f"\nModello migliore: {best_algo} (Qini = {best_qini:.4f})")

    table_header = "| algo | auc_treatment | auc_control | qini_coefficient | mean_uplift |"
    table_sep = "|---|---|---|---|---|"
    table_rows = [
        f"| {r['algo']} | {r['auc_treatment']:.4f} | {r['auc_control']:.4f} "
        f"| {r['qini_coefficient']:.4f} | {r['mean_uplift']:.4f} |"
        for r in results_df.to_dict(orient="records")
    ]

    lines = [
        "# Confronto algoritmi uplift modeling (T-Learner)",
        "",
        f"Modello migliore: **{best_algo}** (Qini coefficient = {best_qini:.4f})",
        "",
        table_header,
        table_sep,
        *table_rows,
        "",
        "Note:",
        "- AUC valutata separatamente per model_treatment e model_control",
        "  (qualità predittiva dei due sotto-modelli presi singolarmente).",
        "- Qini coefficient valuta la qualità del ranking di uplift combinato",
        "  (metrica principale per la selezione del modello: vedi src/uplift_metrics.py).",
        "- Feature usate: solo pre-cutoff, nessuna colonna *_full o ltv_*",
        "  (vedi nota anti-leakage nel docstring dello script).",
    ]
    COMPARISON_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report salvato: {COMPARISON_REPORT}")


if __name__ == "__main__":
    main()
