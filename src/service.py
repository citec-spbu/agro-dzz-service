import base64
import io
import logging
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import planetary_computer
from fastapi import HTTPException, status
from odc.stac import stac_load
from PIL import Image
from pystac_client import Client
from shapely.geometry import Polygon
from shapely.ops import unary_union

from src.clients.fields import FieldContoursClient
from src.config import settings
from src.persistence import DzzSceneAnalyticsRecord, storage
from src.schemas.dzz import (
    DzzCompareSchema,
    DzzCompareSummarySchema,
    DzzContourCultureSchema,
    DzzCultureContextSchema,
    DzzDashboardSchema,
    DzzIndexMapSchema,
    DzzRasterOverlaySchema,
    DzzSceneSchema,
    DzzSummarySchema,
    DzzTimeseriesPointSchema,
)
from src.schemas.fields import ContourSchema, CropRotationSchema

logger = logging.getLogger(__name__)

VALID_INDICES = {"ndvi", "evi", "ndwi", "msavi"}
INDEX_PALETTE = [
    (126, 45, 17),
    (189, 99, 31),
    (235, 188, 67),
    (124, 179, 66),
    (35, 107, 53),
]
DIFF_PALETTE = [
    (179, 38, 30),
    (249, 224, 219),
    (38, 87, 168),
]


@dataclass
class CachedPayload:
    generated_at: datetime
    data: object


@dataclass
class FieldContext:
    contours: list[ContourSchema]
    geometry: object
    active_contour_id: str | None = None
    active_contour_name: str | None = None


class DzzService:
    _dashboard_cache: dict[str, CachedPayload] = {}
    _culture_cache: dict[str, CachedPayload] = {}
    _overlay_cache: dict[str, CachedPayload] = {}
    _compare_cache: dict[str, CachedPayload] = {}
    _cache_ttl = timedelta(hours=settings.CACHE_TTL_HOURS)

    def __init__(self):
        self._fields_client = FieldContoursClient()
        self._catalog = None

    async def get_dashboard(
        self,
        field_id: uuid.UUID,
        season_id: uuid.UUID | None,
        authorization: str | None,
        contour_id: str | None = None,
        force_refresh: bool = False,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> DzzDashboardSchema:
        self._ensure_authorization(authorization)
        resolved_date_from, resolved_date_to = self._resolve_date_range(date_from, date_to)
        use_persisted_only = not force_refresh and (
            date_from is not None or date_to is not None
        )
        cache_key = (
            f"{field_id}:{season_id or 'none'}:{contour_id or 'all'}:"
            f"{resolved_date_from.isoformat()}:{resolved_date_to.isoformat()}"
        )
        cached = self._get_cached(self._dashboard_cache, cache_key, force_refresh)
        if cached is not None:
            if self._should_refresh_cached_culture_context(cached.culture_context):
                field_context = await self._build_field_context(
                    field_id, authorization, contour_id
                )
                culture_context = await self._build_culture_context(
                    field_id, season_id, field_context.contours, authorization
                )
                if culture_context.contours_with_culture > 0:
                    refreshed_dashboard = cached.model_copy(
                        update={"culture_context": culture_context}
                    )
                    self._store_cached(
                        self._dashboard_cache, cache_key, refreshed_dashboard
                    )
                    return refreshed_dashboard
            return cached

        field_context = await self._build_field_context(field_id, authorization, contour_id)
        scene_catalog, scene_analytics, _items_by_id = self._get_or_build_scene_dataset(
            field_id,
            season_id,
            field_context,
            resolved_date_from,
            resolved_date_to,
            force_refresh=force_refresh,
            use_persisted_only=use_persisted_only,
        )
        timeseries = self._build_timeseries(scene_analytics)
        culture_context = await self._build_culture_context(
            field_id, season_id, field_context.contours, authorization
        )
        summary = self._build_summary(field_id, season_id, scene_catalog, timeseries)

        dashboard = DzzDashboardSchema(
            summary=summary,
            culture_context=culture_context,
            catalog=scene_catalog,
            timeseries=timeseries,
        )
        self._store_cached(self._dashboard_cache, cache_key, dashboard)
        return dashboard

    async def get_culture_context(
        self,
        field_id: uuid.UUID,
        season_id: uuid.UUID | None,
        authorization: str | None,
        force_refresh: bool = False,
    ) -> DzzCultureContextSchema:
        self._ensure_authorization(authorization)
        cache_key = f"{field_id}:{season_id or 'none'}"
        cached = self._get_cached(self._culture_cache, cache_key, force_refresh)
        if cached is not None:
            return cached

        field_context = await self._build_field_context(field_id, authorization)
        culture_context = await self._build_culture_context(
            field_id, season_id, field_context.contours, authorization
        )
        self._store_cached(self._culture_cache, cache_key, culture_context)
        return culture_context

    async def get_index_map(
        self,
        field_id: uuid.UUID,
        season_id: uuid.UUID | None,
        scene_id: str,
        index_name: str,
        authorization: str | None,
        contour_id: str | None = None,
        force_refresh: bool = False,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> DzzIndexMapSchema:
        self._ensure_authorization(authorization)
        normalized_index = self._validate_index_name(index_name)
        resolved_date_from, resolved_date_to = self._resolve_date_range(date_from, date_to)
        cache_key = (
            f"{field_id}:{season_id or 'none'}:{contour_id or 'all'}:"
            f"{resolved_date_from.isoformat()}:{resolved_date_to.isoformat()}:"
            f"{scene_id}:{normalized_index}"
        )
        cached = self._get_cached(self._overlay_cache, cache_key, force_refresh)
        if cached is not None:
            return cached

        field_context = await self._build_field_context(field_id, authorization, contour_id)
        scene_catalog = self._load_persisted_scene_catalog(
            field_id,
            season_id,
            field_context.active_contour_id,
            resolved_date_from,
            resolved_date_to,
        )
        items_by_id = {}
        if not scene_catalog:
            scene_catalog, items_by_id = self._search_scenes(
                field_id,
                season_id,
                field_context.geometry,
                resolved_date_from,
                resolved_date_to,
            )
        scene = self._find_scene(scene_catalog, scene_id)
        item = self._resolve_scene_item(
            field_id,
            season_id,
            field_context.geometry,
            scene,
            items_by_id,
        )
        ds = self._load_scene_cube(item, field_context.geometry)
        indices = self._calc_indices(ds, item.collection_id)
        overlay = self._build_overlay(
            indices[normalized_index],
            field_context.geometry,
            scene,
            normalized_index,
        )
        response = DzzIndexMapSchema(
            field_id=field_id,
            season_id=season_id,
            index_name=normalized_index,
            overlay=overlay,
        )
        self._store_cached(self._overlay_cache, cache_key, response)
        return response

    async def compare_scenes(
        self,
        field_id: uuid.UUID,
        season_id: uuid.UUID | None,
        scene_id_a: str,
        scene_id_b: str,
        index_name: str,
        authorization: str | None,
        contour_id: str | None = None,
        force_refresh: bool = False,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> DzzCompareSchema:
        self._ensure_authorization(authorization)
        normalized_index = self._validate_index_name(index_name)
        resolved_date_from, resolved_date_to = self._resolve_date_range(date_from, date_to)
        if scene_id_a == scene_id_b:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Choose two different scenes for comparison.",
            )

        cache_key = (
            f"{field_id}:{season_id or 'none'}:{contour_id or 'all'}:"
            f"{resolved_date_from.isoformat()}:{resolved_date_to.isoformat()}:"
            f"{scene_id_a}:{scene_id_b}:{normalized_index}"
        )
        cached = self._get_cached(self._compare_cache, cache_key, force_refresh)
        if cached is not None:
            return cached

        field_context = await self._build_field_context(field_id, authorization, contour_id)
        scene_catalog = self._load_persisted_scene_catalog(
            field_id,
            season_id,
            field_context.active_contour_id,
            resolved_date_from,
            resolved_date_to,
        )
        items_by_id = {}
        if not scene_catalog:
            scene_catalog, items_by_id = self._search_scenes(
                field_id,
                season_id,
                field_context.geometry,
                resolved_date_from,
                resolved_date_to,
            )
        scene_a = self._find_scene(scene_catalog, scene_id_a)
        scene_b = self._find_scene(scene_catalog, scene_id_b)
        self._validate_compare_pair(scene_a, scene_b)

        item_a = self._resolve_scene_item(
            field_id,
            season_id,
            field_context.geometry,
            scene_a,
            items_by_id,
        )
        item_b = self._resolve_scene_item(
            field_id,
            season_id,
            field_context.geometry,
            scene_b,
            items_by_id,
        )
        ds_a = self._load_scene_cube(item_a, field_context.geometry)
        ds_b = self._load_scene_cube(item_b, field_context.geometry)
        array_a = self._calc_indices(ds_a, item_a.collection_id)[normalized_index]
        array_b = self._calc_indices(ds_b, item_b.collection_id)[normalized_index]
        aligned_b = self._align_array(array_b, array_a)
        diff = (aligned_b - array_a).where(np.isfinite(aligned_b - array_a))

        overlay_a = self._build_overlay(
            array_a, field_context.geometry, scene_a, normalized_index
        )
        overlay_b = self._build_overlay(
            aligned_b, field_context.geometry, scene_b, normalized_index
        )
        diff_overlay = self._build_overlay(
            diff,
            field_context.geometry,
            scene_b,
            normalized_index,
            mode="diff",
        )
        summary = DzzCompareSummarySchema(
            mean_a=self._extract_mean(array_a),
            mean_b=self._extract_mean(aligned_b),
            mean_delta=self._extract_mean(diff),
            min_delta=self._extract_stat(diff, "min"),
            max_delta=self._extract_stat(diff, "max"),
        )
        response = DzzCompareSchema(
            field_id=field_id,
            season_id=season_id,
            index_name=normalized_index,
            scene_a=scene_a,
            scene_b=scene_b,
            overlay_a=overlay_a,
            overlay_b=overlay_b,
            diff_overlay=diff_overlay,
            summary=summary,
        )
        self._store_cached(self._compare_cache, cache_key, response)
        return response

    async def _build_field_context(
        self,
        field_id: uuid.UUID,
        authorization: str,
        contour_id: str | None = None,
    ) -> FieldContext:
        contours = await self._fields_client.get_field_contours(field_id, authorization)
        return self.build_field_context_from_contours(
            contours, contour_id, field_id=field_id
        )

    def build_field_context_from_contours(
        self,
        contours: list[ContourSchema],
        contour_id: str | None = None,
        *,
        field_id: uuid.UUID | None = None,
    ) -> FieldContext:
        polygons = []
        polygon_by_contour_id: dict[str, Polygon] = {}

        for contour in contours:
            points = [(point.longitude, point.latitude) for point in contour.coordinates]
            if len(points) < 3:
                continue
            if points[0] != points[-1]:
                points.append(points[0])
            polygon = Polygon(points)
            if not polygon.is_valid or polygon.is_empty:
                continue
            polygons.append(polygon)
            if contour.id:
                polygon_by_contour_id[contour.id] = polygon

        if not polygons:
            suffix = f" for field {field_id}." if field_id else "."
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Unable to build geometry{suffix}",
            )

        active_contour_name = None
        geometry = unary_union(polygons)

        if contour_id:
            geometry = polygon_by_contour_id.get(contour_id)
            if geometry is None:
                detail = (
                    f"Contour {contour_id} was not found for field {field_id}."
                    if field_id
                    else f"Contour {contour_id} was not found."
                )
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=detail,
                )
            active_contour_name = next(
                (contour.name for contour in contours if contour.id == contour_id),
                None,
            )

        return FieldContext(
            contours=contours,
            geometry=geometry,
            active_contour_id=contour_id,
            active_contour_name=active_contour_name,
        )

    async def run_daily_planetary_sync(self) -> None:
        logger.info("Planetary sync job: started.")
        try:
            refs = await self._fields_client.list_all_fields_internal()
        except Exception:
            logger.exception("Planetary sync: cannot fetch field list from fields-service.")
            return

        targets: set[tuple[uuid.UUID, uuid.UUID | None, str | None]] = {
            (ref.id, None, None) for ref in refs
        }
        targets.update(storage.fetch_distinct_refresh_targets())

        ordered = sorted(
            targets,
            key=lambda item: (str(item[0]), str(item[1] or ""), str(item[2] or "")),
        )
        for field_id, season_id, contour_id in ordered:
            try:
                contours = await self._fields_client.get_field_contours_internal(field_id)
                field_context = self.build_field_context_from_contours(
                    contours, contour_id, field_id=field_id
                )
            except HTTPException as exc:
                logger.warning(
                    "Planetary sync: skip field=%s season=%s contour=%s (%s)",
                    field_id,
                    season_id,
                    contour_id,
                    exc.detail,
                )
                continue
            except Exception:
                logger.exception(
                    "Planetary sync: skip field=%s season=%s contour=%s",
                    field_id,
                    season_id,
                    contour_id,
                )
                continue

            date_from, date_to = self._resolve_date_range(None, None)
            persisted_records = storage.fetch_scene_analytics(
                field_id,
                season_id,
                field_context.active_contour_id,
                date_from,
                date_to,
            )
            try:
                self._refresh_scene_dataset(
                    field_id,
                    season_id,
                    field_context,
                    date_from,
                    date_to,
                    persisted_records,
                )
            except Exception:
                logger.exception(
                    "Planetary sync: refresh failed field=%s season=%s contour=%s",
                    field_id,
                    season_id,
                    contour_id,
                )

        self.clear_response_caches()
        logger.info("Planetary sync job: finished.")

    async def _build_culture_context(
        self,
        field_id: uuid.UUID,
        season_id: uuid.UUID | None,
        contours: list[ContourSchema],
        authorization: str,
    ) -> DzzCultureContextSchema:
        contour_contexts: list[DzzContourCultureSchema] = []

        for contour in contours:
            if not contour.id:
                continue
            try:
                rotations = await self._fields_client.get_contour_crop_rotations(
                    contour.id, authorization
                )
            except RuntimeError as exc:
                logger.warning("Skip crop rotations for contour %s: %s", contour.id, exc)
                continue

            current_rotation = self._select_current_rotation(rotations)
            if current_rotation is None:
                continue

            contour_contexts.append(
                DzzContourCultureSchema(
                    contour_id=contour.id,
                    contour_name=contour.name,
                    culture=current_rotation.culture,
                    cultivar=current_rotation.cultivar,
                    start_date=current_rotation.startDate,
                    end_date=current_rotation.endDate,
                    description=current_rotation.description,
                )
            )

        primary_culture = None
        primary_cultivar = None
        if contour_contexts:
            dominant_pair, _ = Counter(
                (item.culture, item.cultivar) for item in contour_contexts
            ).most_common(1)[0]
            primary_culture, primary_cultivar = dominant_pair

        return DzzCultureContextSchema(
            field_id=field_id,
            season_id=season_id,
            source="fields-service crop rotations",
            primary_culture=primary_culture,
            primary_cultivar=primary_cultivar,
            total_contours=len(contours),
            contours_with_culture=len(contour_contexts),
            contours=sorted(contour_contexts, key=lambda item: item.contour_name or ""),
        )

    def _get_or_build_scene_dataset(
        self,
        field_id: uuid.UUID,
        season_id: uuid.UUID | None,
        field_context: FieldContext,
        date_from: date,
        date_to: date,
        force_refresh: bool = False,
        use_persisted_only: bool = False,
    ) -> tuple[list[DzzSceneSchema], list[DzzSceneAnalyticsRecord], dict]:
        scope_contour_id = field_context.active_contour_id
        persisted_records = storage.fetch_scene_analytics(
            field_id,
            season_id,
            scope_contour_id,
            date_from,
            date_to,
        )
        if persisted_records:
            if force_refresh:
                return self._refresh_scene_dataset(
                    field_id,
                    season_id,
                    field_context,
                    date_from,
                    date_to,
                    persisted_records,
                )
            return (
                self._build_scene_catalog_from_records(field_id, season_id, persisted_records),
                persisted_records,
                {},
            )

        if use_persisted_only:
            logger.info(
                "Skip STAC lookup for field=%s contour=%s period=%s..%s because persisted-only mode is enabled.",
                field_id,
                scope_contour_id or "all",
                date_from.isoformat(),
                date_to.isoformat(),
            )
            return [], [], {}

        if force_refresh:
            return self._refresh_scene_dataset(
                field_id,
                season_id,
                field_context,
                date_from,
                date_to,
                persisted_records,
            )

        scene_catalog, items_by_id = self._search_scenes(
            field_id,
            season_id,
            field_context.geometry,
            date_from,
            date_to,
        )
        if not scene_catalog:
            return [], [], {}

        persisted_by_scene_id = {
            record.scene_id: record for record in persisted_records
        }

        missing_scenes = [
            scene for scene in scene_catalog if scene.scene_id not in persisted_by_scene_id
        ]
        fresh_records = self._build_scene_analytics(
            missing_scenes,
            items_by_id,
            field_context.geometry,
            field_id,
            season_id,
            scope_contour_id,
        )

        if fresh_records:
            storage.upsert_scene_analytics(fresh_records)
            persisted_by_scene_id.update(
                {record.scene_id: record for record in fresh_records}
            )

        ordered_records = [
            persisted_by_scene_id[scene.scene_id]
            for scene in scene_catalog
            if scene.scene_id in persisted_by_scene_id
        ]
        return scene_catalog, ordered_records, items_by_id

    def _load_persisted_scene_catalog(
        self,
        field_id: uuid.UUID,
        season_id: uuid.UUID | None,
        contour_id: str | None,
        date_from: date,
        date_to: date,
    ) -> list[DzzSceneSchema]:
        persisted_records = storage.fetch_scene_analytics(
            field_id,
            season_id,
            contour_id,
            date_from,
            date_to,
        )
        if not persisted_records:
            return []
        return self._build_scene_catalog_from_records(field_id, season_id, persisted_records)

    def _refresh_scene_dataset(
        self,
        field_id: uuid.UUID,
        season_id: uuid.UUID | None,
        field_context: FieldContext,
        date_from: date,
        date_to: date,
        persisted_records: list[DzzSceneAnalyticsRecord],
    ) -> tuple[list[DzzSceneSchema], list[DzzSceneAnalyticsRecord], dict]:
        scope_contour_id = field_context.active_contour_id
        persisted_by_scene_id = {
            record.scene_id: record for record in persisted_records
        }
        items_by_id: dict = {}
        deduped_scenes: dict[str, DzzSceneSchema] = {}

        for chunk_from, chunk_to in self._iter_refresh_chunks(date_from, date_to):
            logger.info(
                "Refresh DZZ chunk field=%s contour=%s period=%s..%s",
                field_id,
                scope_contour_id or "all",
                chunk_from.isoformat(),
                chunk_to.isoformat(),
            )
            chunk_catalog, chunk_items = self._search_scenes(
                field_id,
                season_id,
                field_context.geometry,
                chunk_from,
                chunk_to,
            )
            items_by_id.update(chunk_items)
            for scene in chunk_catalog:
                deduped_scenes[scene.scene_id] = scene

            missing_chunk_scenes = [
                scene for scene in chunk_catalog if scene.scene_id not in persisted_by_scene_id
            ]
            if not missing_chunk_scenes:
                continue

            fresh_records = self._build_scene_analytics(
                missing_chunk_scenes,
                chunk_items,
                field_context.geometry,
                field_id,
                season_id,
                scope_contour_id,
            )
            if not fresh_records:
                continue

            storage.upsert_scene_analytics(fresh_records)
            persisted_by_scene_id.update(
                {record.scene_id: record for record in fresh_records}
            )

        scene_catalog = sorted(
            deduped_scenes.values(),
            key=lambda scene: (scene.scene_date, scene.cloud_cover or 999.0),
        )
        ordered_records = [
            persisted_by_scene_id[scene.scene_id]
            for scene in scene_catalog
            if scene.scene_id in persisted_by_scene_id
        ]
        return scene_catalog, ordered_records, items_by_id

    @staticmethod
    def _build_scene_catalog_from_records(
        field_id: uuid.UUID,
        season_id: uuid.UUID | None,
        records: list[DzzSceneAnalyticsRecord],
    ) -> list[DzzSceneSchema]:
        return [
            DzzSceneSchema(
                field_id=field_id,
                season_id=season_id,
                scene_date=record.scene_date,
                collection=record.collection,
                sensor=record.sensor,
                scene_id=record.scene_id,
                cloud_cover=record.cloud_cover,
            )
            for record in records
        ]

    def _search_scenes(
        self,
        field_id: uuid.UUID,
        season_id: uuid.UUID | None,
        aoi_geom,
        date_from: date,
        date_to: date,
    ):
        date_range = f"{date_from.isoformat()}/{date_to.isoformat()}"
        catalog = self._get_catalog_client()

        search = catalog.search(
            collections=["sentinel-2-l2a", "landsat-c2-l2"],
            intersects=aoi_geom.__geo_interface__,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": settings.MAX_CLOUD}},
        )
        items = list(search.items())
        items_by_id = {item.id: item for item in items}

        rows = []
        for item in items:
            if item.datetime is None:
                continue
            rows.append(
                DzzSceneSchema(
                    field_id=field_id,
                    season_id=season_id,
                    scene_date=item.datetime.date(),
                    collection=item.collection_id,
                    sensor=self._resolve_sensor(item.collection_id),
                    scene_id=item.id,
                    cloud_cover=self._to_optional_float(
                        item.properties.get("eo:cloud_cover")
                    ),
                )
            )

        rows.sort(key=lambda scene: (scene.scene_date, scene.cloud_cover or 999.0))
        return rows, items_by_id

    def _resolve_scene_item(
        self,
        field_id: uuid.UUID,
        season_id: uuid.UUID | None,
        aoi_geom,
        scene: DzzSceneSchema,
        cached_items_by_id: dict,
    ):
        item = cached_items_by_id.get(scene.scene_id)
        if item is not None:
            return item

        _, items_by_id = self._search_scenes(
            field_id,
            season_id,
            aoi_geom,
            scene.scene_date,
            scene.scene_date,
        )
        item = items_by_id.get(scene.scene_id)
        if item is not None:
            return item
        return self._find_item(cached_items_by_id, scene.scene_id)

    @staticmethod
    def _iter_refresh_chunks(date_from: date, date_to: date) -> list[tuple[date, date]]:
        chunk_days = max(settings.REFRESH_CHUNK_DAYS, 1)
        chunks: list[tuple[date, date]] = []
        chunk_start = date_from

        while chunk_start <= date_to:
            chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), date_to)
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end + timedelta(days=1)

        return chunks

    def _build_scene_analytics(
        self,
        scene_catalog: list[DzzSceneSchema],
        items_by_id: dict,
        aoi_geom,
        field_id: uuid.UUID,
        season_id: uuid.UUID | None,
        contour_id: str | None,
    ) -> list[DzzSceneAnalyticsRecord]:
        if not scene_catalog:
            return []

        analytics_rows: list[DzzSceneAnalyticsRecord] = []
        for scene in scene_catalog:
            item = items_by_id.get(scene.scene_id)
            if item is None:
                continue
            try:
                ds = self._load_scene_cube(item, aoi_geom)
                indices = self._calc_indices(ds, item.collection_id)
                ndvi = self._extract_mean(indices["ndvi"])
                evi = self._extract_mean(indices["evi"])
                ndwi = self._extract_mean(indices["ndwi"])
                msavi = self._extract_mean(indices["msavi"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skip scene %s: %s", scene.scene_id, exc)
                continue

            analytics_rows.append(
                DzzSceneAnalyticsRecord(
                    field_id=field_id,
                    season_id=season_id,
                    contour_id=contour_id,
                    scene_id=scene.scene_id,
                    scene_date=scene.scene_date,
                    collection=scene.collection,
                    sensor=scene.sensor,
                    cloud_cover=scene.cloud_cover,
                    ndvi=ndvi,
                    evi=evi,
                    ndwi=ndwi,
                    msavi=msavi,
                    updated_at=datetime.now(UTC),
                )
            )
        return analytics_rows

    def _build_timeseries(
        self, scene_analytics: list[DzzSceneAnalyticsRecord]
    ) -> list[DzzTimeseriesPointSchema]:
        if not scene_analytics:
            return []

        best_daily_scenes = (
            pd.DataFrame(
                [
                    {
                        "scene_date": row.scene_date,
                        "sensor": row.sensor,
                        "scene_id": row.scene_id,
                        "cloud_cover": row.cloud_cover,
                        "ndvi": row.ndvi,
                        "evi": row.evi,
                        "ndwi": row.ndwi,
                        "msavi": row.msavi,
                    }
                    for row in scene_analytics
                ]
            )
            .sort_values(["scene_date", "cloud_cover"])
            .groupby("scene_date", as_index=False)
            .first()
        )

        return [
            DzzTimeseriesPointSchema(
                date=pd.to_datetime(row["scene_date"]).date(),
                sensor=row["sensor"],
                scene_id=row["scene_id"],
                cloud_cover=self._to_optional_float(row["cloud_cover"]),
                ndvi=self._to_optional_float(row["ndvi"]),
                evi=self._to_optional_float(row["evi"]),
                ndwi=self._to_optional_float(row["ndwi"]),
                msavi=self._to_optional_float(row["msavi"]),
            )
            for _, row in best_daily_scenes.iterrows()
        ]

    def _build_summary(
        self,
        field_id: uuid.UUID,
        season_id: uuid.UUID | None,
        scene_catalog,
        timeseries,
    ) -> DzzSummarySchema:
        latest = timeseries[-1] if timeseries else None
        ndvi_delta = None

        if latest is None:
            return DzzSummarySchema(
                field_id=field_id,
                season_id=season_id,
                generated_at=datetime.now(UTC),
                total_scenes=len(scene_catalog),
                total_best_daily_scenes=0,
            )

        if len(timeseries) >= 2:
            prev = timeseries[-2]
            if latest.ndvi is not None and prev.ndvi is not None:
                ndvi_delta = round(latest.ndvi - prev.ndvi, 4)

        return DzzSummarySchema(
            field_id=field_id,
            season_id=season_id,
            generated_at=datetime.now(UTC),
            total_scenes=len(scene_catalog),
            total_best_daily_scenes=len(timeseries),
            latest_scene_date=latest.date,
            latest_sensor=latest.sensor,
            latest_cloud_cover=latest.cloud_cover,
            latest_ndvi=latest.ndvi,
            latest_evi=latest.evi,
            latest_ndwi=latest.ndwi,
            latest_msavi=latest.msavi,
            ndvi_delta=ndvi_delta,
        )

    def _load_scene_cube(self, item, aoi_geom):
        signed_item = planetary_computer.sign(item)
        band_mapping = self._resolve_band_mapping(item.collection_id)

        return stac_load(
            [signed_item],
            bands=[
                band_mapping["nir"],
                band_mapping["red"],
                band_mapping["green"],
            ],
            geopolygon=aoi_geom,
            resolution=band_mapping["resolution"],
            chunks={},
        ).squeeze()

    def _calc_indices(self, ds, collection_id: str):
        band_mapping = self._resolve_band_mapping(collection_id)
        nir = ds[band_mapping["nir"]].astype("float32")
        red = ds[band_mapping["red"]].astype("float32")
        green = ds[band_mapping["green"]].astype("float32")

        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = ((nir - red) / (nir + red + 1e-6)).clip(min=-1.0, max=1.0)
            evi = (
                2.5
                * (nir - red)
                / (nir + 6 * red - 7.5 * green + 1 + 1e-6)
            ).clip(min=-1.0, max=1.0)
            ndwi = ((green - nir) / (green + nir + 1e-6)).clip(min=-1.0, max=1.0)
            msavi = (
                (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2
            ).clip(min=-1.0, max=1.0)

        return {
            "ndvi": ndvi.where(np.isfinite(ndvi)),
            "evi": evi.where(np.isfinite(evi)),
            "ndwi": ndwi.where(np.isfinite(ndwi)),
            "msavi": msavi.where(np.isfinite(msavi)),
        }

    def _build_overlay(
        self,
        array,
        aoi_geom,
        scene: DzzSceneSchema,
        index_name: str,
        mode: str = "single",
    ) -> DzzRasterOverlaySchema:
        data = self._prepare_array_for_image(array)
        finite_mask = np.isfinite(data)
        if not finite_mask.any():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"No valid pixels found for {index_name.upper()} scene overlay.",
            )

        actual_min = round(float(np.nanmin(data[finite_mask])), 4)
        actual_max = round(float(np.nanmax(data[finite_mask])), 4)
        mean_value = round(float(np.nanmean(data[finite_mask])), 4)
        if mode == "diff":
            max_abs = max(abs(actual_min), abs(actual_max), 0.05)
            display_min = round(-max_abs, 4)
            display_max = round(max_abs, 4)
            palette = DIFF_PALETTE
        else:
            display_min = -1.0
            display_max = 1.0
            palette = INDEX_PALETTE

        rgba = self._colorize_array(data, display_min, display_max, palette)
        min_lon, min_lat, max_lon, max_lat = aoi_geom.bounds
        return DzzRasterOverlaySchema(
            scene_id=scene.scene_id,
            scene_date=scene.scene_date,
            sensor=scene.sensor,
            collection=scene.collection,
            index_name=index_name,
            image_url=self._to_png_data_url(rgba),
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            display_min=display_min,
            display_max=display_max,
            actual_min=actual_min,
            actual_max=actual_max,
            mean_value=mean_value,
            mode=mode,
        )

    @staticmethod
    def _prepare_array_for_image(array) -> np.ndarray:
        data = np.asarray(array.values, dtype=np.float32)
        if "y" in getattr(array, "coords", {}):
            y_values = np.asarray(array.coords["y"].values)
            if y_values.size > 1 and y_values[0] < y_values[-1]:
                data = np.flipud(data)
        return data

    @staticmethod
    def _colorize_array(
        data: np.ndarray,
        display_min: float,
        display_max: float,
        palette: list[tuple[int, int, int]],
    ) -> np.ndarray:
        safe_range = max(display_max - display_min, 1e-6)
        normalized = np.clip((data - display_min) / safe_range, 0.0, 1.0)
        finite_mask = np.isfinite(data)
        positions = np.linspace(0.0, 1.0, len(palette))
        rgba = np.zeros((*data.shape, 4), dtype=np.uint8)

        for channel in range(3):
            channel_values = np.array([color[channel] for color in palette], dtype=float)
            interpolated = np.interp(normalized, positions, channel_values)
            rgba[..., channel] = np.where(finite_mask, interpolated, 0).astype(np.uint8)

        rgba[..., 3] = np.where(finite_mask, 210, 0).astype(np.uint8)
        return rgba

    @staticmethod
    def _to_png_data_url(rgba: np.ndarray) -> str:
        image = Image.fromarray(rgba, mode="RGBA")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _validate_compare_pair(scene_a: DzzSceneSchema, scene_b: DzzSceneSchema) -> None:
        if scene_a.sensor != scene_b.sensor:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Comparison is supported only for scenes from the same sensor.",
            )

    @staticmethod
    def _should_refresh_cached_culture_context(
        culture_context: DzzCultureContextSchema | None,
    ) -> bool:
        if culture_context is None:
            return True
        return culture_context.contours_with_culture == 0

    @staticmethod
    def _align_array(array, reference):
        if array.shape == reference.shape:
            return array
        if "x" in array.coords and "y" in array.coords and "x" in reference.coords and "y" in reference.coords:
            return array.reindex(x=reference["x"], y=reference["y"], method="nearest")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Unable to align scene rasters for comparison.",
        )

    @staticmethod
    def _select_current_rotation(
        rotations: list[CropRotationSchema],
    ) -> CropRotationSchema | None:
        if not rotations:
            return None

        today = datetime.now(UTC).date()
        active = [
            rotation
            for rotation in rotations
            if rotation.startDate <= today
            and (rotation.endDate is None or rotation.endDate >= today)
        ]
        if active:
            active.sort(key=lambda item: item.startDate, reverse=True)
            return active[0]

        rotations.sort(key=lambda item: item.startDate, reverse=True)
        return rotations[0]

    @staticmethod
    def _find_scene(scene_catalog: list[DzzSceneSchema], scene_id: str) -> DzzSceneSchema:
        for scene in scene_catalog:
            if scene.scene_id == scene_id:
                return scene
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scene {scene_id} was not found for this field.",
        )

    @staticmethod
    def _find_item(items_by_id: dict, scene_id: str):
        item = items_by_id.get(scene_id)
        if item is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scene item {scene_id} is unavailable.",
            )
        return item

    @staticmethod
    def _ensure_authorization(authorization: str | None) -> None:
        if authorization:
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is required.",
        )

    @classmethod
    def clear_response_caches(cls) -> None:
        cls._dashboard_cache.clear()
        cls._culture_cache.clear()
        cls._overlay_cache.clear()
        cls._compare_cache.clear()

    @classmethod
    def _get_cached(cls, cache: dict[str, CachedPayload], key: str, force_refresh: bool):
        if force_refresh:
            return None
        cached = cache.get(key)
        if cached is None:
            return None
        if datetime.now(UTC) - cached.generated_at > cls._cache_ttl:
            return None
        return cached.data

    @staticmethod
    def _store_cached(cache: dict[str, CachedPayload], key: str, value: object) -> None:
        cache[key] = CachedPayload(generated_at=datetime.now(UTC), data=value)

    @staticmethod
    def _validate_index_name(index_name: str) -> str:
        normalized = (index_name or "").strip().lower()
        if normalized in VALID_INDICES:
            return normalized
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unsupported index '{index_name}'. Allowed values: {sorted(VALID_INDICES)}.",
        )

    @staticmethod
    def _resolve_date_range(
        date_from: date | None,
        date_to: date | None,
    ) -> tuple[date, date]:
        today = datetime.now(UTC).date()
        resolved_from = date_from or (today - timedelta(days=settings.DAYS_BACK))
        resolved_to = date_to or (today + timedelta(days=settings.DAYS_FORWARD))
        if resolved_from > resolved_to:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="dateFrom must be less than or equal to dateTo.",
            )
        return resolved_from, resolved_to

    def _get_catalog_client(self):
        if self._catalog is None:
            self._catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        return self._catalog

    @staticmethod
    def _resolve_sensor(collection_id: str) -> str:
        return "Landsat" if "landsat" in collection_id else "S2"

    @staticmethod
    def _resolve_band_mapping(collection_id: str) -> dict[str, str | int]:
        if "landsat" in collection_id:
            return {"nir": "nir08", "red": "red", "green": "green", "resolution": 30}
        return {"nir": "B08", "red": "B04", "green": "B03", "resolution": 10}

    @staticmethod
    def _extract_mean(array) -> float | None:
        value = array.mean(skipna=True).values
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(number):
            return None
        return round(number, 4)

    @staticmethod
    def _extract_stat(array, stat_name: str) -> float | None:
        values = np.asarray(array.values, dtype=np.float32)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return None
        if stat_name == "min":
            return round(float(finite_values.min()), 4)
        if stat_name == "max":
            return round(float(finite_values.max()), 4)
        return None

    @staticmethod
    def _to_optional_float(value) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(number):
            return None
        return round(number, 4)
