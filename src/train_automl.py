from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from flaml import AutoML
from sklearn.model_selection import train_test_split

from train_uplift_models import (
    MODELS_DIR,
    TRAIN_PATH,
    RANDOM_STATE,
    prepare_X,
)
from uplift_metrics import qini_coefficient

COMPARISON_REPORT = Path(__file__).resolve().parent.parent / "models_comparison.md"

AUTOML_ESTIMATOR_LIST = ["lrl1", "rf", "xgboost", "extra_tree"]


TIME_BUDGET_SECONDS = 120


def main() -> None:
    df = pd.read_parquet(TRAIN_PATH)
    gender_categories = sorted(df["gender"].unique().tolist())

    df["_strata"] = df["treatment_flg"].astype(str) + "_" + df["target"].astype(str)
    train_df, val_df = train_test_split(
        df, test_size=0.25, random_state=RANDOM_STATE, stratify=df["_strata"]
    )
    print(f"Train: {train_df.shape[0]:,} | Validation: {val_df.shape[0]:,}")

    X_train = prepare_X(train_df, gender_categories)
    X_val = prepare_X(val_df, gender_categories)

    train_t = train_df[train_df["treatment_flg"] == 1]
    X_train_t = X_train.loc[train_t.index]
    y_train_t = train_t["target"]

    print(f"\n=== AutoML: model_treatment (budget {TIME_BUDGET_SECONDS}s) ===")
    automl_treatment = AutoML()
    automl_treatment.fit(
        X_train=X_train_t, y_train=y_train_t,
        task="classification", metric="roc_auc",
        time_budget=TIME_BUDGET_SECONDS, seed=RANDOM_STATE, verbose=0,
        estimator_list=AUTOML_ESTIMATOR_LIST,
    )
    print(f"Miglior estimator (treatment): {automl_treatment.best_estimator}")
    print(f"Best config: {automl_treatment.best_config}")

    train_c = train_df[train_df["treatment_flg"] == 0]
    X_train_c = X_train.loc[train_c.index]
    y_train_c = train_c["target"]

    print(f"\n=== AutoML: model_control (budget {TIME_BUDGET_SECONDS}s) ===")
    automl_control = AutoML()
    automl_control.fit(
        X_train=X_train_c, y_train=y_train_c,
        task="classification", metric="roc_auc",
        time_budget=TIME_BUDGET_SECONDS, seed=RANDOM_STATE, verbose=0,
        estimator_list=AUTOML_ESTIMATOR_LIST,
    )
    print(f"Miglior estimator (control): {automl_control.best_estimator}")
    print(f"Best config: {automl_control.best_config}")


    from sklearn.metrics import roc_auc_score

    val_t_mask = val_df["treatment_flg"] == 1
    val_c_mask = val_df["treatment_flg"] == 0

    auc_treatment = roc_auc_score(
        val_df.loc[val_t_mask, "target"],
        automl_treatment.predict_proba(X_val.loc[val_t_mask])[:, 1],
    )
    auc_control = roc_auc_score(
        val_df.loc[val_c_mask, "target"],
        automl_control.predict_proba(X_val.loc[val_c_mask])[:, 1],
    )

    p_treat_all = automl_treatment.predict_proba(X_val)[:, 1]
    p_ctrl_all = automl_control.predict_proba(X_val)[:, 1]
    uplift_score = p_treat_all - p_ctrl_all

    qini = qini_coefficient(
        uplift_score=uplift_score,
        target=val_df["target"].values,
        treatment=val_df["treatment_flg"].values,
    )

    print(f"\nAUC model_treatment: {auc_treatment:.4f}")
    print(f"AUC model_control:   {auc_control:.4f}")
    print(f"Qini coefficient:    {qini:.4f}")
    print(f"Uplift score medio:  {uplift_score.mean():.4f}")

    joblib.dump(automl_treatment, MODELS_DIR / "uplift_automl_treatment.joblib")
    joblib.dump(automl_control, MODELS_DIR / "uplift_automl_control.joblib")
    print(f"\nModelli salvati in {MODELS_DIR}")


    new_row = (
        f"| automl | {auc_treatment:.4f} | {auc_control:.4f} "
        f"| {qini:.4f} | {uplift_score.mean():.4f} |"
    )
    detail_header = "## Dettaglio AutoML (FLAML)"

    if COMPARISON_REPORT.exists():
        content = COMPARISON_REPORT.read_text(encoding="utf-8")
        lines = content.split("\n")

        automl_row_idx = next(
            (i for i, line in enumerate(lines) if line.startswith("| automl |")), None
        )
        if automl_row_idx is not None:
            lines[automl_row_idx] = new_row
        else:
            insert_idx = None
            for i, line in enumerate(lines):
                if line.startswith("|") and not line.startswith("|---"):
                    insert_idx = i
            if insert_idx is not None:
                lines.insert(insert_idx + 1, new_row)
            else:
                lines.append(new_row)

        if detail_header in lines:
            lines = lines[: lines.index(detail_header)]
            while lines and lines[-1] == "":
                lines.pop()

        lines += [
            "",
            detail_header,
            "",
            f"- Best estimator (treatment): `{automl_treatment.best_estimator}`",
            f"- Best estimator (control): `{automl_control.best_estimator}`",
            f"- Time budget per modello: {TIME_BUDGET_SECONDS}s",
        ]
        COMPARISON_REPORT.write_text("\n".join(lines), encoding="utf-8")
        print(f"Report aggiornato: {COMPARISON_REPORT}")
    else:
        print("models_comparison.md non trovato: riga AutoML non aggiunta.")


if __name__ == "__main__":
    main()
