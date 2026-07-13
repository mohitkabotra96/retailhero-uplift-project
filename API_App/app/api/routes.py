from __future__ import annotations

from fastapi import APIRouter, HTTPException

from schemas import ClientInput, PredictionOutput
from model_utils import predict_for_client

router = APIRouter()


@router.post("/predict", response_model=PredictionOutput)
def predict(client: ClientInput) -> PredictionOutput:
    try:
        result = predict_for_client(client.model_dump())
    except FileNotFoundError as e:

        raise HTTPException(status_code=503, detail=f"Modelli non disponibili: {e}")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Errore nella predizione: {e}")

    return PredictionOutput(**result)
