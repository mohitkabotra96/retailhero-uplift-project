
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from train_uplift_models import MODELS_DIR, TRAIN_PATH, RANDOM_STATE, prepare_X

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SHAP_SAMPLE_SIZE = 5000
TOP_K_FEATURES_PER_CLIENT = 5


def load_artifacts():
    """Carica modelli Random Forest selezionati + lista feature/colonne."""
    model_treatment = joblib.load(MODELS_DIR / "uplift_random_forest_treatment.joblib")
    model_control = joblib.load(MODELS_DIR / "uplift_random_forest_control.joblib")

    with open(MODELS_DIR / "feature_columns.json") as f:
        meta = json.load(f)

    return model_treatment, model_control, meta


def build_validation_sample(meta: dict, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ricostruisce lo stesso validation set di train_uplift_models.py
    (stesso seed/stratify) e ne campiona n righe per SHAP globale."""
    df = pd.read_parquet(TRAIN_PATH)
    df["_strata"] = df["treatment_flg"].astype(str) + "_" + df["target"].astype(str)
    _, val_df = train_test_split(
        df, test_size=0.25, random_state=RANDOM_STATE, stratify=df["_strata"]
    )
    X_val = prepare_X(val_df, meta["gender_categories"])

    if len(val_df) > n:
        sample_idx = val_df.sample(n=n, random_state=RANDOM_STATE).index
        val_df = val_df.loc[sample_idx]
        X_val = X_val.loc[sample_idx]

    return val_df.reset_index(drop=True), X_val.reset_index(drop=True)


def compute_global_importance(model_treatment, model_control, X_sample: pd.DataFrame) -> pd.DataFrame:
    """Calcola |SHAP| medio per feature sui due modelli e sulla loro
    differenza (approssimazione del contributo all'uplift)."""
    explainer_t = shap.TreeExplainer(model_treatment)
    explainer_c = shap.TreeExplainer(model_control)
    sv_t = explainer_t.shap_values(X_sample)
    sv_c = explainer_c.shap_values(X_sample)

    sv_t = sv_t[1] if isinstance(sv_t, list) else sv_t[:, :, 1]
    sv_c = sv_c[1] if isinstance(sv_c, list) else sv_c[:, :, 1]

    sv_uplift = sv_t - sv_c  

    importance = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap_treatment": np.abs(sv_t).mean(axis=0),
        "mean_abs_shap_control": np.abs(sv_c).mean(axis=0),
        "mean_abs_shap_uplift": np.abs(sv_uplift).mean(axis=0),
    }).sort_values("mean_abs_shap_uplift", ascending=False)

    return importance, sv_t, sv_c, sv_uplift


def explain_client(
    client_features: pd.DataFrame,
    model_treatment,
    model_control,
    top_k: int = TOP_K_FEATURES_PER_CLIENT,
) -> list[dict]:
    """
    Spiega la predizione di uplift per un singolo cliente (client_features:
    DataFrame a una riga, formato prodotto da prepare_X()).

    Ritorna le top_k feature per |contributo|: feature, shap_uplift,
    direction. Importata identica nel backend FastAPI per l'endpoint
    /predict.
    """
    explainer_t = shap.TreeExplainer(model_treatment)
    explainer_c = shap.TreeExplainer(model_control)

    sv_t = explainer_t.shap_values(client_features)
    sv_c = explainer_c.shap_values(client_features)

    sv_t = sv_t[1] if isinstance(sv_t, list) else sv_t[:, :, 1]
    sv_c = sv_c[1] if isinstance(sv_c, list) else sv_c[:, :, 1]

    sv_uplift = (sv_t - sv_c)[0]  
    contributions = pd.Series(sv_uplift, index=client_features.columns)
    top = contributions.abs().sort_values(ascending=False).head(top_k)

    result = []
    for feature in top.index:
        value = contributions[feature]
        result.append({
            "feature": feature,
            "shap_uplift_contribution": float(value),
            "direction": "aumenta l'uplift" if value > 0 else "riduce l'uplift",
        })
    return result


def main() -> None:
    model_treatment, model_control, meta = load_artifacts()
    val_df, X_val_sample = build_validation_sample(meta, SHAP_SAMPLE_SIZE)
    print(f"Calcolo SHAP su un campione di {X_val_sample.shape[0]:,} clienti...")

    importance, sv_t, sv_c, sv_uplift = compute_global_importance(
        model_treatment, model_control, X_val_sample
    )

    print("\n=== Importanza globale feature (ordinata per contributo all'uplift) ===")
    print(importance.to_string(index=False))

    importance.to_csv(REPORTS_DIR / "shap_global_importance.csv", index=False)
    print(f"\nSalvato: {REPORTS_DIR / 'shap_global_importance.csv'}")

    top15 = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top15["feature"], top15["mean_abs_shap_uplift"], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP| contribution to uplift score")
    ax.set_title("Importanza feature — contributo all'uplift (Random Forest)")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "shap_global_importance.png", dpi=150)
    print(f"Salvato: {REPORTS_DIR / 'shap_global_importance.png'}")
    print("\n=== Esempio: spiegazione primi 3 clienti del campione ===")
    for i in range(3):
        client_row = X_val_sample.iloc[[i]]
        client_id = val_df.loc[i, "client_id"]
        explanation = explain_client(client_row, model_treatment, model_control)
        print(f"\nCliente {client_id}:")
        for item in explanation:
            print(f"  {item['feature']}: {item['shap_uplift_contribution']:+.4f} ({item['direction']})")


if __name__ == "__main__":
    main()
