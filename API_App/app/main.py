from __future__ import annotations

from fastapi import FastAPI

from api.routes import router as api_router
from web_routes import router as web_router

app = FastAPI(
    title="Explainable Campaign Targeting Simulator",
    description=(
        "API per stimare l'uplift di una campagna marketing su un "
        "cliente, con raccomandazione operativa, LTV e spiegazione "
        "delle variabili più rilevanti."
    ),
    version="1.0.0",
)

app.include_router(api_router)
app.include_router(web_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
