import uuid
from datetime import date

from pydantic import BaseModel


class CoordinatesSchema(BaseModel):
    longitude: float
    latitude: float


class ContourSchema(BaseModel):
    id: str | None = None
    name: str | None = None
    coordinates: list[CoordinatesSchema]


class InternalFieldRefSchema(BaseModel):
    """Ответ GET /api/internal/.../all-coordinates (MeteoResponse)."""

    longitude: float
    latitude: float
    id: uuid.UUID


class CropRotationSchema(BaseModel):
    id: str
    culture: str
    cultivar: str
    startDate: date
    endDate: date | None = None
    description: str | None = None
