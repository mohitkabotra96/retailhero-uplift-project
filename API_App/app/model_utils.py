from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd
import shap

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

FEATURE_COLS = [
    "age",
    "gender",
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

TOP_K_FEATURES = 5


@lru_cache(maxsize=1)
def load_artifacts():
    """Carica modelli e metadati una sola volta (cache di processo)."""
    required_files = [
        "uplift_random_forest_treatment.joblib",
        "uplift_random_forest_control.joblib",
        "feature_columns.json",
        "ltv_thresholds.json",
    ]
    missing = [f for f in required_files if not (MODELS_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"File mancanti in {MODELS_DIR}: {missing}. "
            "Verificare che models/ contenga gli artefatti prodotti da src/train_uplift_models.py."
        )

    model_treatment = joblib.load(MODELS_DIR / "uplift_random_forest_treatment.joblib")
    model_control = joblib.load(MODELS_DIR / "uplift_random_forest_control.joblib")

    with open(MODELS_DIR / "feature_columns.json", encoding="utf-8") as f:
        feature_meta = json.load(f)

    with open(MODELS_DIR / "ltv_thresholds.json", encoding="utf-8") as f:
        ltv_thresholds = json.load(f)

    explainer_treatment = shap.TreeExplainer(model_treatment)
    explainer_control = shap.TreeExplainer(model_control)

    return {
        "model_treatment": model_treatment,
        "model_control": model_control,
        "feature_meta": feature_meta,
        "ltv_thresholds": ltv_thresholds,
        "explainer_treatment": explainer_treatment,
        "explainer_control": explainer_control,
    }


def prepare_X(client_dict: dict, gender_categories: list[str]) -> pd.DataFrame:
    """Converte l'input ricevuto dall'API in un DataFrame a una riga,
    nello stesso formato (one-hot gender) usato in training."""
    row = {col: client_dict[col] for col in FEATURE_COLS}
    X = pd.DataFrame([row])

    gender_dummies = pd.get_dummies(X["gender"], prefix="gender")
    for cat in gender_categories:
        col = f"gender_{cat}"
        if col not in gender_dummies.columns:
            gender_dummies[col] = 0
    gender_dummies = gender_dummies[[f"gender_{c}" for c in gender_categories]]

    X = X.drop(columns=["gender"]).join(gender_dummies)
    return X.astype(float)


def classify_ltv(ltv_value: float, thresholds: dict) -> str:
    if ltv_value <= thresholds["q25"]:
        return "Low"
    if ltv_value <= thresholds["q50"]:
        return "Medium-low"
    if ltv_value <= thresholds["q75"]:
        return "Medium-high"
    return "High"


def explain_client(X_row: pd.DataFrame, explainer_treatment, explainer_control, top_k: int = TOP_K_FEATURES):
    """Spiegazione SHAP per un singolo cliente (vedi src/explainability.py
    per la nota metodologica sull'approssimazione 'differenza di SHAP')."""
    sv_t = explainer_treatment.shap_values(X_row)
    sv_c = explainer_control.shap_values(X_row)

    sv_t = sv_t[1] if isinstance(sv_t, list) else sv_t[:, :, 1]
    sv_c = sv_c[1] if isinstance(sv_c, list) else sv_c[:, :, 1]

    sv_uplift = (sv_t - sv_c)[0]

    contributions = pd.Series(sv_uplift, index=X_row.columns)
    top = contributions.abs().sort_values(ascending=False).head(top_k)

    return [
        {
            "feature": feature,
            "shap_uplift_contribution": float(contributions[feature]),
            "direction": "increases uplift" if contributions[feature] > 0 else "decreases uplift",
        }
        for feature in top.index
    ]


def predict_for_client(client_dict: dict) -> dict:
    """Pipeline completa: input cliente -> uplift score, raccomandazione,
    LTV/cluster, top feature. Usata dall'endpoint POST /predict."""
    artifacts = load_artifacts()

    X_row = prepare_X(client_dict, artifacts["feature_meta"]["gender_categories"])

    p_treatment = float(artifacts["model_treatment"].predict_proba(X_row)[:, 1][0])
    p_control = float(artifacts["model_control"].predict_proba(X_row)[:, 1][0])
    uplift_score = p_treatment - p_control

    raccomandazione = "Include" if uplift_score > 0 else "Exclude"

    ltv_proxy = client_dict["total_revenue_pre"]
    ltv_cluster = classify_ltv(ltv_proxy, artifacts["ltv_thresholds"])

    top_features = explain_client(
        X_row, artifacts["explainer_treatment"], artifacts["explainer_control"]
    )

    return {
        "uplift_score": uplift_score,
        "p_treatment": p_treatment,
        "p_control": p_control,
        "raccomandazione": raccomandazione,
        "ltv_proxy": ltv_proxy,
        "ltv_cluster": ltv_cluster,
        "top_features": top_features,
    }
