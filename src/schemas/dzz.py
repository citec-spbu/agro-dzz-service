import uuid
from datetime import date, datetime

from pydantic import BaseModel, Field


class DzzSceneSchema(BaseModel):
    field_id: uuid.UUID
    season_id: uuid.UUID | None = None
    scene_date: date
    collection: str
    sensor: str
    scene_id: str
    cloud_cover: float | None = None


class DzzTimeseriesPointSchema(BaseModel):
    date: date
    sensor: str
    scene_id: str
    cloud_cover: float | None = None
    ndvi: float | None = None
    evi: float | None = None
    ndwi: float | None = None
    msavi: float | None = None


class DzzContourCultureSchema(BaseModel):
    contour_id: str
    contour_name: str | None = None
    culture: str
    cultivar: str
    start_date: date
    end_date: date | None = None
    description: str | None = None


class DzzCultureContextSchema(BaseModel):
    field_id: uuid.UUID
    season_id: uuid.UUID | None = None
    source: str
    primary_culture: str | None = None
    primary_cultivar: str | None = None
    total_contours: int
    contours_with_culture: int
    contours: list[DzzContourCultureSchema] = Field(default_factory=list)


class DzzRasterOverlaySchema(BaseModel):
    scene_id: str
    scene_date: date
    sensor: str
    collection: str
    index_name: str
    image_url: str
    bounds: list[list[float]]
    display_min: float
    display_max: float
    actual_min: float | None = None
    actual_max: float | None = None
    mean_value: float | None = None
    mode: str = "single"


class DzzIndexMapSchema(BaseModel):
    field_id: uuid.UUID
    season_id: uuid.UUID | None = None
    index_name: str
    overlay: DzzRasterOverlaySchema


class DzzCompareSummarySchema(BaseModel):
    mean_a: float | None = None
    mean_b: float | None = None
    mean_delta: float | None = None
    min_delta: float | None = None
    max_delta: float | None = None


class DzzCompareSchema(BaseModel):
    field_id: uuid.UUID
    season_id: uuid.UUID | None = None
    index_name: str
    scene_a: DzzSceneSchema
    scene_b: DzzSceneSchema
    overlay_a: DzzRasterOverlaySchema
    overlay_b: DzzRasterOverlaySchema
    diff_overlay: DzzRasterOverlaySchema
    summary: DzzCompareSummarySchema


class DzzSummarySchema(BaseModel):
    field_id: uuid.UUID
    season_id: uuid.UUID | None = None
    generated_at: datetime
    total_scenes: int
    total_best_daily_scenes: int
    latest_scene_date: date | None = None
    latest_sensor: str | None = None
    latest_cloud_cover: float | None = None
    latest_ndvi: float | None = None
    latest_evi: float | None = None
    latest_ndwi: float | None = None
    latest_msavi: float | None = None
    ndvi_delta: float | None = None
    health_status: str | None = None
    notes: list[str] = Field(default_factory=list)


class DzzDashboardSchema(BaseModel):
    summary: DzzSummarySchema
    culture_context: DzzCultureContextSchema | None = None
    catalog: list[DzzSceneSchema]
    timeseries: list[DzzTimeseriesPointSchema]
