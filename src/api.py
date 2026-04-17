import uuid
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query

from src.schemas.dzz import (
    DzzCompareSchema,
    DzzCultureContextSchema,
    DzzDashboardSchema,
    DzzIndexMapSchema,
    DzzSceneSchema,
    DzzSummarySchema,
    DzzTimeseriesPointSchema,
)
from src.service import DzzService

router = APIRouter(prefix="/api/dzz", tags=["dzz data"])


@router.get("/{field_id}", response_model=DzzDashboardSchema)
async def get_dashboard(
    field_id: uuid.UUID,
    service: Annotated[DzzService, Depends()],
    season_id: uuid.UUID | None = Query(default=None, alias="seasonId"),
    contour_id: str | None = Query(default=None, alias="contourId"),
    date_from: date | None = Query(default=None, alias="dateFrom"),
    date_to: date | None = Query(default=None, alias="dateTo"),
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
):
    return await service.get_dashboard(
        field_id,
        season_id,
        authorization,
        contour_id,
        date_from=date_from,
        date_to=date_to,
    )


@router.get("/{field_id}/summary", response_model=DzzSummarySchema)
async def get_summary(
    field_id: uuid.UUID,
    service: Annotated[DzzService, Depends()],
    season_id: uuid.UUID | None = Query(default=None, alias="seasonId"),
    contour_id: str | None = Query(default=None, alias="contourId"),
    date_from: date | None = Query(default=None, alias="dateFrom"),
    date_to: date | None = Query(default=None, alias="dateTo"),
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
):
    dashboard = await service.get_dashboard(
        field_id,
        season_id,
        authorization,
        contour_id,
        date_from=date_from,
        date_to=date_to,
    )
    return dashboard.summary


@router.get("/{field_id}/catalog", response_model=list[DzzSceneSchema])
async def get_catalog(
    field_id: uuid.UUID,
    service: Annotated[DzzService, Depends()],
    season_id: uuid.UUID | None = Query(default=None, alias="seasonId"),
    contour_id: str | None = Query(default=None, alias="contourId"),
    date_from: date | None = Query(default=None, alias="dateFrom"),
    date_to: date | None = Query(default=None, alias="dateTo"),
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
):
    dashboard = await service.get_dashboard(
        field_id,
        season_id,
        authorization,
        contour_id,
        date_from=date_from,
        date_to=date_to,
    )
    return dashboard.catalog


@router.get("/{field_id}/timeseries", response_model=list[DzzTimeseriesPointSchema])
async def get_timeseries(
    field_id: uuid.UUID,
    service: Annotated[DzzService, Depends()],
    season_id: uuid.UUID | None = Query(default=None, alias="seasonId"),
    contour_id: str | None = Query(default=None, alias="contourId"),
    date_from: date | None = Query(default=None, alias="dateFrom"),
    date_to: date | None = Query(default=None, alias="dateTo"),
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
):
    dashboard = await service.get_dashboard(
        field_id,
        season_id,
        authorization,
        contour_id,
        date_from=date_from,
        date_to=date_to,
    )
    return dashboard.timeseries


@router.get("/{field_id}/culture-context", response_model=DzzCultureContextSchema)
async def get_culture_context(
    field_id: uuid.UUID,
    service: Annotated[DzzService, Depends()],
    season_id: uuid.UUID | None = Query(default=None, alias="seasonId"),
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
):
    return await service.get_culture_context(field_id, season_id, authorization)


@router.get("/{field_id}/index-map", response_model=DzzIndexMapSchema)
async def get_index_map(
    field_id: uuid.UUID,
    service: Annotated[DzzService, Depends()],
    scene_id: str = Query(alias="sceneId"),
    index_name: str = Query(default="ndvi", alias="index"),
    season_id: uuid.UUID | None = Query(default=None, alias="seasonId"),
    contour_id: str | None = Query(default=None, alias="contourId"),
    date_from: date | None = Query(default=None, alias="dateFrom"),
    date_to: date | None = Query(default=None, alias="dateTo"),
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
):
    return await service.get_index_map(
        field_id,
        season_id,
        scene_id,
        index_name,
        authorization,
        contour_id,
        date_from=date_from,
        date_to=date_to,
    )


@router.get("/{field_id}/compare", response_model=DzzCompareSchema)
async def compare_scenes(
    field_id: uuid.UUID,
    service: Annotated[DzzService, Depends()],
    scene_id_a: str = Query(alias="sceneIdA"),
    scene_id_b: str = Query(alias="sceneIdB"),
    index_name: str = Query(default="ndvi", alias="index"),
    season_id: uuid.UUID | None = Query(default=None, alias="seasonId"),
    contour_id: str | None = Query(default=None, alias="contourId"),
    date_from: date | None = Query(default=None, alias="dateFrom"),
    date_to: date | None = Query(default=None, alias="dateTo"),
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
):
    return await service.compare_scenes(
        field_id,
        season_id,
        scene_id_a,
        scene_id_b,
        index_name,
        authorization,
        contour_id,
        date_from=date_from,
        date_to=date_to,
    )


@router.post("/{field_id}/refresh", response_model=DzzDashboardSchema)
async def refresh_dashboard(
    field_id: uuid.UUID,
    service: Annotated[DzzService, Depends()],
    season_id: uuid.UUID | None = Query(default=None, alias="seasonId"),
    contour_id: str | None = Query(default=None, alias="contourId"),
    date_from: date | None = Query(default=None, alias="dateFrom"),
    date_to: date | None = Query(default=None, alias="dateTo"),
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
):
    return await service.get_dashboard(
        field_id,
        season_id,
        authorization,
        contour_id,
        force_refresh=True,
        date_from=date_from,
        date_to=date_to,
    )
