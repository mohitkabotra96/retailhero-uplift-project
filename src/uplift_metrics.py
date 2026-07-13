from __future__ import annotations

import numpy as np
import pandas as pd


def qini_curve(
    uplift_score: np.ndarray,
    target: np.ndarray,
    treatment: np.ndarray,
    n_bins: int = 20,
) -> pd.DataFrame:
    """
    Ordina i clienti per uplift_score decrescente e calcola, per ogni
    frazione cumulata k della popolazione:

        gain(k) = Y1(k) - Y0(k) * N1(k) / N0(k)

    Ritorna un DataFrame con fraction, gain, gain_random (baseline casuale).
    """
    uplift_score = np.asarray(uplift_score, dtype=float)

    if len(uplift_score) == 0:
        raise ValueError("qini_curve richiede almeno un cliente in input.")
    if np.isnan(uplift_score).any():
        raise ValueError("uplift_score contiene NaN: verificare le feature del cliente.")

    df = pd.DataFrame(
        {"uplift_score": uplift_score, "target": target, "treatment": treatment}
    ).sort_values("uplift_score", ascending=False).reset_index(drop=True)

    n = len(df)
    step = max(n // n_bins, 1)
    cut_points = list(range(step, n, step)) + [n]

    rows = []
    for k in cut_points:
        top_k = df.iloc[:k]
        treat_mask = top_k["treatment"] == 1
        ctrl_mask = top_k["treatment"] == 0

        n1 = treat_mask.sum()
        n0 = ctrl_mask.sum()
        y1 = top_k.loc[treat_mask, "target"].sum()
        y0 = top_k.loc[ctrl_mask, "target"].sum()

        gain = y1 - y0 * (n1 / n0) if n0 > 0 else 0.0
        rows.append({"fraction": k / n, "gain": gain})

    curve = pd.DataFrame(rows)
    total_gain = curve["gain"].iloc[-1]
    curve["gain_random"] = curve["fraction"] * total_gain

    return curve


def qini_coefficient(
    uplift_score: np.ndarray,
    target: np.ndarray,
    treatment: np.ndarray,
    n_bins: int = 20,
) -> float:
    """Area tra curva del modello e baseline random. Più alto = miglior
    ranking per uplift stimato. Confronto relativo tra modelli, non
    normalizzato rispetto a un modello ideale."""
    curve = qini_curve(uplift_score, target, treatment, n_bins=n_bins)
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area_model = _trapz(curve["gain"], curve["fraction"])
    area_random = _trapz(curve["gain_random"], curve["fraction"])
    return float(area_model - area_random)
