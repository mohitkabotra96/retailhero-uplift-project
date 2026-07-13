from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClientInput(BaseModel):
    age: int = Field(..., ge=14, le=120, description="Età del cliente")
    gender: Literal["M", "F", "U"] = Field(..., description="Genere: M, F, U (sconosciuto)")

    n_transactions_pre: int = Field(..., ge=0, description="Numero di transazioni storiche")
    total_revenue_pre: float = Field(..., ge=0, description="Spesa totale storica")
    avg_basket_value_pre: float = Field(..., ge=0, description="Valore medio scontrino")

    total_regular_points_received_pre: float = Field(0, description="Punti regolari ricevuti")
    total_express_points_received_pre: float = Field(0, description="Punti express ricevuti")
    total_regular_points_spent_pre: float = Field(0, description="Punti regolari spesi (negativo o 0)")
    total_express_points_spent_pre: float = Field(0, description="Punti express spesi (negativo o 0)")

    n_product_lines: int = Field(..., ge=0, description="Numero totale righe prodotto acquistate")
    n_distinct_products: int = Field(..., ge=0, description="Numero di prodotti distinti acquistati")
    total_quantity: float = Field(..., ge=0, description="Quantità totale di prodotti acquistati")
    total_iss_sum: float = Field(..., ge=0, description="Somma trn_sum_from_iss")

    share_alcohol_lines: float = Field(0, ge=0, le=1, description="Quota di righe alcolici")
    share_own_trademark_lines: float = Field(0, ge=0, le=1, description="Quota righe marca propria")

    tenure_days: int = Field(..., ge=0, description="Giorni di fedeltà (da iscrizione)")
    recency_days_pre: float = Field(..., ge=0, description="Giorni dall'ultimo acquisto")
    has_redeemed: int = Field(..., ge=0, le=1, description="Ha mai riscattato il coupon di benvenuto (0/1)")
    is_new_customer: int = Field(0, ge=0, le=1, description="Cliente nuovo, senza storico pre-cutoff (0/1)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 45,
                "gender": "F",
                "n_transactions_pre": 20,
                "total_revenue_pre": 8500.0,
                "avg_basket_value_pre": 425.0,
                "total_regular_points_received_pre": 80.0,
                "total_express_points_received_pre": 0.0,
                "total_regular_points_spent_pre": -50.0,
                "total_express_points_spent_pre": 0.0,
                "n_product_lines": 150,
                "n_distinct_products": 95,
                "total_quantity": 210.0,
                "total_iss_sum": 8100.0,
                "share_alcohol_lines": 0.01,
                "share_own_trademark_lines": 0.2,
                "tenure_days": 400,
                "recency_days_pre": 5,
                "has_redeemed": 1,
                "is_new_customer": 0,
            }
        }


class FeatureContribution(BaseModel):
    feature: str
    shap_uplift_contribution: float
    direction: str


class PredictionOutput(BaseModel):
    uplift_score: float = Field(..., description="p_treatment - p_control")
    p_treatment: float = Field(..., description="Probabilità di acquisto se trattato")
    p_control: float = Field(..., description="Probabilità di acquisto se non trattato")
    raccomandazione: Literal["Include", "Exclude"]
    ltv_proxy: float = Field(..., description="Proxy LTV basato sull'input fornito")
    ltv_cluster: Literal["Low", "Medium-low", "Medium-high", "High"]
    top_features: list[FeatureContribution]
