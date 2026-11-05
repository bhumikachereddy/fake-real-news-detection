"""
Request and response schemas for news prediction API.
"""

from pydantic import BaseModel, Field
from typing import List


class NewsRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        description="News article text"
    )


class BatchNewsRequest(BaseModel):

    texts: List[str] = Field(
        ...,
        min_length=1,
        description="List of news articles"
    )


class PredictionResponse(BaseModel):

    prediction: int
    result: str
    message: str


class BatchPredictionResponse(BaseModel):

    predictions: List[PredictionResponse]
    total: int