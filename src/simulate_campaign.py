from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

VAL_PRED_PATH = PROCESSED_DIR / "val_predictions.parquet"

BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]
N_RANDOM_REPETITIONS = 20
RANDOM_STATE = 42


def incremental_gain(selected: pd.DataFrame) -> float:
    """Stima empirica del guadagno incrementale su una selezione
    di clienti (vedi formula nel docstring del modulo)."""
    treat = selected[selected["treatment_flg"] == 1]
    ctrl = selected[selected["treatment_flg"] == 0]

    n1, n0 = len(treat), len(ctrl)
    if n0 == 0:
        return np.nan 

    y1 = treat["target"].sum()
    y0 = ctrl["target"].sum()

    return y1 - y0 * (n1 / n0)


def main() -> None:
    df = pd.read_parquet(VAL_PRED_PATH)
    n_total = len(df)
    print(f"Validation set: {n_total:,} clienti")
    print(f"Guadagno incrementale stimabile sull'intera popolazione: {incremental_gain(df):.1f}")

    rng = np.random.default_rng(RANDOM_STATE)
    results = []

    for budget in BUDGETS:
        n_select = int(budget * n_total)

        uplift_selected = df.nlargest(n_select, "uplift_score")
        gain_uplift = incremental_gain(uplift_selected)

        random_gains = []
        for _ in range(N_RANDOM_REPETITIONS):
            idx = rng.choice(df.index, size=n_select, replace=False)
            random_selected = df.loc[idx]
            random_gains.append(incremental_gain(random_selected))
        gain_random_mean = float(np.nanmean(random_gains))
        gain_random_std = float(np.nanstd(random_gains))

        improvement_pct = (
            (gain_uplift - gain_random_mean) / abs(gain_random_mean) * 100
            if gain_random_mean != 0 else np.nan
        )

        results.append({
            "budget_pct": budget * 100,
            "n_selected": n_select,
            "incremental_gain_uplift": gain_uplift,
            "incremental_gain_random_mean": gain_random_mean,
            "incremental_gain_random_std": gain_random_std,
            "improvement_pct": improvement_pct,
        })

        print(
            f"Budget {budget*100:5.0f}% ({n_select:6,} clienti) | "
            f"uplift-based: {gain_uplift:7.1f} | "
            f"random: {gain_random_mean:7.1f} ± {gain_random_std:5.1f} | "
            f"miglioramento: {improvement_pct:+6.1f}%"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORTS_DIR / "campaign_simulation.csv", index=False)
    print(f"\nSalvato: {REPORTS_DIR / 'campaign_simulation.csv'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results_df["budget_pct"], results_df["incremental_gain_uplift"],
            marker="o", label="Selezione uplift-based", color="#4C72B0")
    ax.errorbar(results_df["budget_pct"], results_df["incremental_gain_random_mean"],
                yerr=results_df["incremental_gain_random_std"],
                marker="s", label="Selezione random (media ± std)", color="#DD8452")
    ax.set_xlabel("% clienti raggiunti dalla campagna (budget)")
    ax.set_ylabel("Guadagno incrementale stimato (conversioni in più)")
    ax.set_title("Simulazione campagna: uplift-based vs random")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "campaign_simulation.png", dpi=150)
    print(f"Salvato: {REPORTS_DIR / 'campaign_simulation.png'}")

    print("\n=== Tabella riassuntiva ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
