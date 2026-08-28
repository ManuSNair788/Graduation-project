from typing import Literal, Optional

from pydantic import BaseModel, Field


class ExtractionSchema(BaseModel):
    barrier: Literal[
        "fit_size",
        "price_uncertainty",
        "styling_doubt",
        "occasion_fit",
        "quality_doubt",
        "choice_overload",
        "trust_returns",
        "stock_size_unavailable",
        "other",
    ]
    save_intent: Literal["purchase_intent", "bookmarking", "aspiration", "comparison", "unclear"]
    journey_stage: Literal["browse", "saved", "revisit", "cart", "checkout"]
    segment_signal: Optional[str] = None
    info_sought_outside_app: Optional[str] = None
    workaround: Optional[str] = None
    intensity: int = Field(ge=1, le=5)
    addressable_without_money: bool
    evidence: str


class RelevanceBatch(BaseModel):
    """Expected shape of a Stage 1 batch response: {"relevant": [true, false, ...]}"""

    relevant: list[bool]
