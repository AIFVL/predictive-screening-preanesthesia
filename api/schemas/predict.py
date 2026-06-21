from __future__ import annotations

from pydantic import BaseModel


class PredictResponse(BaseModel):
    predicted_class: int
    probability: float
    threshold: float
    risk_level: str
    calibrated: bool
    prevalence_train: float | None
    warnings: list[str]


class ShapContributionSchema(BaseModel):
    feature: str
    value: float | None
    shap_value: float


class ExplainResponse(BaseModel):
    contributions: list[ShapContributionSchema]
    top_n: int
    algorithm: str
    model_id: str


__all__ = [
    "PredictResponse",
    "ExplainResponse",
    "ShapContributionSchema",
]
