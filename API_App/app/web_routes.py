from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from model_utils import predict_for_client
from schemas import ClientInput

router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "result": None, "values": None, "error": None}
    )


@router.post("/web/predict", response_class=HTMLResponse)
def web_predict(
    request: Request,
    age: int = Form(...),
    gender: str = Form(...),
    n_transactions_pre: int = Form(...),
    total_revenue_pre: float = Form(...),
    avg_basket_value_pre: float = Form(...),
    total_regular_points_received_pre: float = Form(0),
    total_express_points_received_pre: float = Form(0),
    total_regular_points_spent_pre: float = Form(0),
    total_express_points_spent_pre: float = Form(0),
    n_product_lines: int = Form(...),
    n_distinct_products: int = Form(...),
    total_quantity: float = Form(...),
    total_iss_sum: float = Form(...),
    share_alcohol_lines: float = Form(0),
    share_own_trademark_lines: float = Form(0),
    tenure_days: int = Form(...),
    recency_days_pre: float = Form(...),
    has_redeemed: int = Form(0),
    is_new_customer: int = Form(0),
):
    values = {
        "age": age, "gender": gender,
        "n_transactions_pre": n_transactions_pre,
        "total_revenue_pre": total_revenue_pre,
        "avg_basket_value_pre": avg_basket_value_pre,
        "total_regular_points_received_pre": total_regular_points_received_pre,
        "total_express_points_received_pre": total_express_points_received_pre,
        "total_regular_points_spent_pre": total_regular_points_spent_pre,
        "total_express_points_spent_pre": total_express_points_spent_pre,
        "n_product_lines": n_product_lines,
        "n_distinct_products": n_distinct_products,
        "total_quantity": total_quantity,
        "total_iss_sum": total_iss_sum,
        "share_alcohol_lines": share_alcohol_lines,
        "share_own_trademark_lines": share_own_trademark_lines,
        "tenure_days": tenure_days,
        "recency_days_pre": recency_days_pre,
        "has_redeemed": has_redeemed,
        "is_new_customer": is_new_customer,
    }

    try:
     
        validated = ClientInput(**values)
        result = predict_for_client(validated.model_dump())
        error = None
    except ValidationError as e:
        result = None
        error = "; ".join(f"{err['loc'][0]}: {err['msg']}" for err in e.errors())
    except Exception as e:  
        result = None
        error = str(e)

    return templates.TemplateResponse(
        "index.html", {"request": request, "result": result, "values": values, "error": error}
    )
