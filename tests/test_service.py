import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from fastapi import HTTPException

import src.service as service_module
from src.persistence import DzzSceneAnalyticsRecord
from src.schemas.dzz import DzzSceneSchema
from src.schemas.fields import ContourSchema, CoordinatesSchema, CropRotationSchema
from src.service import DzzService, FieldContext


def make_contour(contour_id: str, name: str) -> ContourSchema:
    return ContourSchema(
        id=contour_id,
        name=name,
        coordinates=[
            CoordinatesSchema(longitude=37.0, latitude=55.0),
            CoordinatesSchema(longitude=37.1, latitude=55.0),
            CoordinatesSchema(longitude=37.1, latitude=55.1),
            CoordinatesSchema(longitude=37.0, latitude=55.1),
        ],
    )


def make_scene(field_id: uuid.UUID, scene_id: str, date_shift: int) -> DzzSceneSchema:
    scene_date = (datetime.now(UTC) + timedelta(days=date_shift)).date()
    return DzzSceneSchema(
        field_id=field_id,
        season_id=None,
        scene_date=scene_date,
        collection="sentinel-2-l2a",
        sensor="S2",
        scene_id=scene_id,
        cloud_cover=2.5,
    )


def make_dataset(nir_scale: float) -> xr.Dataset:
    coords = {"y": np.array([1.0, 0.0]), "x": np.array([0.0, 1.0])}
    return xr.Dataset(
        data_vars={
            "B08": (("y", "x"), np.array([[0.8, 0.7], [0.6, 0.5]]) * nir_scale),
            "B04": (("y", "x"), np.array([[0.2, 0.3], [0.2, 0.2]])),
            "B03": (("y", "x"), np.array([[0.3, 0.3], [0.2, 0.2]])),
        },
        coords=coords,
    )


def make_shifted_dataset(nir_scale: float) -> xr.Dataset:
    coords = {"y": np.array([1.2, 0.2, -0.8]), "x": np.array([-0.2, 0.8, 1.8])}
    return xr.Dataset(
        data_vars={
            "B08": (
                ("y", "x"),
                np.array(
                    [
                        [0.82, 0.74, 0.68],
                        [0.66, 0.58, 0.52],
                        [0.48, 0.41, 0.37],
                    ]
                )
                * nir_scale,
            ),
            "B04": (("y", "x"), np.full((3, 3), 0.22)),
            "B03": (("y", "x"), np.full((3, 3), 0.26)),
        },
        coords=coords,
    )


def make_analytics_record(
    field_id: uuid.UUID,
    scene_id: str,
    scene_date,
    contour_id: str | None = None,
) -> DzzSceneAnalyticsRecord:
    return DzzSceneAnalyticsRecord(
        field_id=field_id,
        season_id=None,
        contour_id=contour_id,
        scene_id=scene_id,
        scene_date=scene_date,
        collection="sentinel-2-l2a",
        sensor="S2",
        cloud_cover=2.5,
        ndvi=0.41,
        evi=0.33,
        ndwi=-0.12,
        msavi=0.29,
        updated_at=datetime.now(UTC),
    )


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> DzzService:
    monkeypatch.setattr(service_module.Client, "open", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(
        service_module.storage,
        "fetch_scene_analytics",
        lambda *_args, **_kwargs: [],
    )
    instance = DzzService()
    return instance


@pytest.mark.asyncio
async def test_get_culture_context_uses_active_crop_rotation(
    service: DzzService, monkeypatch: pytest.MonkeyPatch
) -> None:
    field_id = uuid.uuid4()
    contours = [make_contour(str(uuid.uuid4()), "Контур A")]
    now = datetime.now(UTC).date()
    rotations = [
        CropRotationSchema(
            id=str(uuid.uuid4()),
            culture="Пшеница",
            cultivar="Сорт 1",
            startDate=now - timedelta(days=30),
            endDate=now + timedelta(days=30),
            description="Активная запись",
        ),
        CropRotationSchema(
            id=str(uuid.uuid4()),
            culture="Кукуруза",
            cultivar="Старый сорт",
            startDate=now - timedelta(days=400),
            endDate=now - timedelta(days=300),
            description="Архив",
        ),
    ]

    async def fake_get_field_contours(_field_id, _authorization):
        return contours

    async def fake_get_crop_rotations(_contour_id, _authorization):
        return rotations

    monkeypatch.setattr(service._fields_client, "get_field_contours", fake_get_field_contours)
    monkeypatch.setattr(
        service._fields_client, "get_contour_crop_rotations", fake_get_crop_rotations
    )

    result = await service.get_culture_context(field_id, None, "Bearer test")

    assert result.primary_culture == "Пшеница"
    assert result.primary_cultivar == "Сорт 1"
    assert result.contours_with_culture == 1
    assert result.contours[0].description == "Активная запись"


@pytest.mark.asyncio
async def test_get_index_map_returns_overlay_payload(
    service: DzzService, monkeypatch: pytest.MonkeyPatch
) -> None:
    field_id = uuid.uuid4()
    contours = [make_contour(str(uuid.uuid4()), "Контур A")]
    scene = make_scene(field_id, "scene-a", -1)
    items_by_id = {
        "scene-a": SimpleNamespace(
            id="scene-a",
            collection_id="sentinel-2-l2a",
            properties={"eo:cloud_cover": 2.5},
        )
    }

    async def fake_get_field_contours(_field_id, _authorization):
        return contours

    monkeypatch.setattr(service._fields_client, "get_field_contours", fake_get_field_contours)
    monkeypatch.setattr(
        service,
        "_search_scenes",
        lambda _field_id, _season_id, _aoi_geom, _date_from, _date_to: ([scene], items_by_id),
    )
    monkeypatch.setattr(
        service,
        "_load_scene_cube",
        lambda _item, _aoi_geom: make_dataset(1.0),
    )

    result = await service.get_index_map(field_id, None, "scene-a", "ndvi", "Bearer test")

    assert result.overlay.index_name == "ndvi"
    assert result.overlay.image_url.startswith("data:image/png;base64,")
    assert result.overlay.bounds == [[55.0, 37.0], [55.1, 37.1]]


@pytest.mark.asyncio
async def test_compare_scenes_returns_diff_overlay_and_summary(
    service: DzzService, monkeypatch: pytest.MonkeyPatch
) -> None:
    field_id = uuid.uuid4()
    contours = [make_contour(str(uuid.uuid4()), "Контур A")]
    scene_a = make_scene(field_id, "scene-a", -3)
    scene_b = make_scene(field_id, "scene-b", -1)
    items_by_id = {
        "scene-a": SimpleNamespace(
            id="scene-a",
            collection_id="sentinel-2-l2a",
            properties={"eo:cloud_cover": 3.2},
        ),
        "scene-b": SimpleNamespace(
            id="scene-b",
            collection_id="sentinel-2-l2a",
            properties={"eo:cloud_cover": 1.3},
        ),
    }

    async def fake_get_field_contours(_field_id, _authorization):
        return contours

    datasets = {
        "scene-a": make_dataset(1.0),
        "scene-b": make_dataset(1.15),
    }

    monkeypatch.setattr(service._fields_client, "get_field_contours", fake_get_field_contours)
    monkeypatch.setattr(
        service,
        "_search_scenes",
        lambda _field_id, _season_id, _aoi_geom, _date_from, _date_to: ([scene_a, scene_b], items_by_id),
    )
    monkeypatch.setattr(
        service,
        "_load_scene_cube",
        lambda item, _aoi_geom: datasets[item.id],
    )

    result = await service.compare_scenes(
        field_id,
        None,
        "scene-a",
        "scene-b",
        "ndvi",
        "Bearer test",
    )

    assert result.diff_overlay.mode == "diff"
    assert result.summary.mean_delta is not None
    assert result.summary.mean_delta > 0
    assert result.overlay_a.image_url.startswith("data:image/png;base64,")
    assert result.overlay_b.image_url.startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_compare_scenes_aligns_different_grids_without_scipy(
    service: DzzService, monkeypatch: pytest.MonkeyPatch
) -> None:
    field_id = uuid.uuid4()
    contours = [make_contour(str(uuid.uuid4()), "Контур A")]
    scene_a = make_scene(field_id, "scene-a", -3)
    scene_b = make_scene(field_id, "scene-b", -1)
    items_by_id = {
        "scene-a": SimpleNamespace(
            id="scene-a",
            collection_id="sentinel-2-l2a",
            properties={"eo:cloud_cover": 3.2},
        ),
        "scene-b": SimpleNamespace(
            id="scene-b",
            collection_id="sentinel-2-l2a",
            properties={"eo:cloud_cover": 1.3},
        ),
    }

    async def fake_get_field_contours(_field_id, _authorization):
        return contours

    datasets = {
        "scene-a": make_dataset(1.0),
        "scene-b": make_shifted_dataset(1.08),
    }

    monkeypatch.setattr(service._fields_client, "get_field_contours", fake_get_field_contours)
    monkeypatch.setattr(
        service,
        "_search_scenes",
        lambda _field_id, _season_id, _aoi_geom, _date_from, _date_to: ([scene_a, scene_b], items_by_id),
    )
    monkeypatch.setattr(
        service,
        "_load_scene_cube",
        lambda item, _aoi_geom: datasets[item.id],
    )

    result = await service.compare_scenes(
        field_id,
        None,
        "scene-a",
        "scene-b",
        "ndvi",
        "Bearer test",
    )

    assert result.overlay_b.image_url.startswith("data:image/png;base64,")
    assert result.diff_overlay.image_url.startswith("data:image/png;base64,")
    assert result.summary.mean_delta is not None


@pytest.mark.asyncio
async def test_get_index_map_rejects_unknown_index(
    service: DzzService, monkeypatch: pytest.MonkeyPatch
) -> None:
    field_id = uuid.uuid4()

    async def fake_get_field_contours(_field_id, _authorization):
        return [make_contour(str(uuid.uuid4()), "Контур A")]

    monkeypatch.setattr(service._fields_client, "get_field_contours", fake_get_field_contours)

    with pytest.raises(HTTPException) as exc:
        await service.get_index_map(field_id, None, "scene-a", "lai", "Bearer test")

    assert exc.value.status_code == 422


def test_get_or_build_scene_dataset_refreshes_from_stac_when_force_refresh_is_enabled(
    service: DzzService, monkeypatch: pytest.MonkeyPatch
) -> None:
    field_id = uuid.uuid4()
    contour_id = str(uuid.uuid4())
    start_date = datetime.now(UTC).date() - timedelta(days=5)
    end_date = datetime.now(UTC).date()
    first_scene = make_scene(field_id, "scene-a", -3)
    second_scene = make_scene(field_id, "scene-b", -1)
    field_context = FieldContext(
        contours=[make_contour(contour_id, "Контур A")],
        geometry=SimpleNamespace(),
        active_contour_id=contour_id,
    )

    persisted_record = make_analytics_record(
        field_id,
        first_scene.scene_id,
        first_scene.scene_date,
        contour_id=contour_id,
    )
    new_record = make_analytics_record(
        field_id,
        second_scene.scene_id,
        second_scene.scene_date,
        contour_id=contour_id,
    )
    upsert_calls = []

    monkeypatch.setattr(
        service,
        "_search_scenes",
        lambda _field_id, _season_id, _aoi_geom, _date_from, _date_to: (
            [first_scene, second_scene],
            {"scene-a": SimpleNamespace(id="scene-a"), "scene-b": SimpleNamespace(id="scene-b")},
        ),
    )
    monkeypatch.setattr(
        service_module.storage,
        "fetch_scene_analytics",
        lambda *_args, **_kwargs: [persisted_record],
    )
    monkeypatch.setattr(
        service,
        "_build_scene_analytics",
        lambda scenes, *_args, **_kwargs: [new_record]
        if [scene.scene_id for scene in scenes] == ["scene-b"]
        else [],
    )
    monkeypatch.setattr(
        service_module.storage,
        "upsert_scene_analytics",
        lambda rows: upsert_calls.append(rows),
    )

    catalog, analytics, items_by_id = service._get_or_build_scene_dataset(
        field_id,
        None,
        field_context,
        start_date,
        end_date,
        force_refresh=True,
    )

    assert [scene.scene_id for scene in catalog] == ["scene-a", "scene-b"]
    assert [row.scene_id for row in analytics] == ["scene-a", "scene-b"]
    assert sorted(items_by_id) == ["scene-a", "scene-b"]
    assert len(upsert_calls) == 1
    assert [row.scene_id for row in upsert_calls[0]] == ["scene-b"]


def test_get_or_build_scene_dataset_skips_stac_when_records_already_persisted(
    service: DzzService, monkeypatch: pytest.MonkeyPatch
) -> None:
    field_id = uuid.uuid4()
    contour_id = str(uuid.uuid4())
    start_date = datetime.now(UTC).date() - timedelta(days=5)
    end_date = datetime.now(UTC).date()
    persisted_record = make_analytics_record(
        field_id,
        "scene-a",
        datetime.now(UTC).date() - timedelta(days=2),
        contour_id=contour_id,
    )
    field_context = FieldContext(
        contours=[make_contour(contour_id, "Контур A")],
        geometry=SimpleNamespace(),
        active_contour_id=contour_id,
    )

    monkeypatch.setattr(
        service_module.storage,
        "fetch_scene_analytics",
        lambda *_args, **_kwargs: [persisted_record],
    )

    def fail_search(*_args, **_kwargs):
        raise AssertionError("STAC search must not be called when persisted rows already exist.")

    monkeypatch.setattr(service, "_search_scenes", fail_search)

    catalog, analytics, items_by_id = service._get_or_build_scene_dataset(
        field_id,
        None,
        field_context,
        start_date,
        end_date,
    )

    assert [scene.scene_id for scene in catalog] == ["scene-a"]
    assert [row.scene_id for row in analytics] == ["scene-a"]
    assert items_by_id == {}


def test_get_or_build_scene_dataset_returns_empty_without_stac_in_persisted_only_mode(
    service: DzzService, monkeypatch: pytest.MonkeyPatch
) -> None:
    field_id = uuid.uuid4()
    contour_id = str(uuid.uuid4())
    start_date = datetime.now(UTC).date() - timedelta(days=5)
    end_date = datetime.now(UTC).date()
    field_context = FieldContext(
        contours=[make_contour(contour_id, "Контур A")],
        geometry=SimpleNamespace(),
        active_contour_id=contour_id,
    )

    monkeypatch.setattr(
        service_module.storage,
        "fetch_scene_analytics",
        lambda *_args, **_kwargs: [],
    )

    def fail_search(*_args, **_kwargs):
        raise AssertionError("STAC search must not be called in persisted-only mode.")

    monkeypatch.setattr(service, "_search_scenes", fail_search)

    catalog, analytics, items_by_id = service._get_or_build_scene_dataset(
        field_id,
        None,
        field_context,
        start_date,
        end_date,
        use_persisted_only=True,
    )

    assert catalog == []
    assert analytics == []
    assert items_by_id == {}


def test_resolve_date_range_rejects_invalid_period(service: DzzService) -> None:
    with pytest.raises(HTTPException) as exc:
        service._resolve_date_range(
            datetime(2026, 4, 10, tzinfo=UTC).date(),
            datetime(2026, 4, 1, tzinfo=UTC).date(),
        )

    assert exc.value.status_code == 422


def test_iter_refresh_chunks_splits_period_by_setting(
    service: DzzService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module.settings, "REFRESH_CHUNK_DAYS", 3)

    chunks = service._iter_refresh_chunks(
        datetime(2026, 3, 1, tzinfo=UTC).date(),
        datetime(2026, 3, 8, tzinfo=UTC).date(),
    )

    assert chunks == [
        (
            datetime(2026, 3, 1, tzinfo=UTC).date(),
            datetime(2026, 3, 3, tzinfo=UTC).date(),
        ),
        (
            datetime(2026, 3, 4, tzinfo=UTC).date(),
            datetime(2026, 3, 6, tzinfo=UTC).date(),
        ),
        (
            datetime(2026, 3, 7, tzinfo=UTC).date(),
            datetime(2026, 3, 8, tzinfo=UTC).date(),
        ),
    ]


def test_get_or_build_scene_dataset_uses_contour_scope_for_storage(
    service: DzzService, monkeypatch: pytest.MonkeyPatch
) -> None:
    field_id = uuid.uuid4()
    contour_id = str(uuid.uuid4())
    start_date = datetime.now(UTC).date() - timedelta(days=10)
    end_date = datetime.now(UTC).date()
    scene = make_scene(field_id, "scene-a", -2)
    field_context = FieldContext(
        contours=[make_contour(contour_id, "Контур A")],
        geometry=SimpleNamespace(),
        active_contour_id=contour_id,
    )
    captured = {}

    monkeypatch.setattr(
        service,
        "_search_scenes",
        lambda *_args, **_kwargs: ([scene], {"scene-a": SimpleNamespace(id="scene-a")}),
    )

    def fake_fetch(_field_id, _season_id, scope_contour_id, _date_from, _date_to):
        captured["contour_id"] = scope_contour_id
        return []

    monkeypatch.setattr(service_module.storage, "fetch_scene_analytics", fake_fetch)
    monkeypatch.setattr(
        service,
        "_build_scene_analytics",
        lambda *_args, **_kwargs: [
            make_analytics_record(field_id, "scene-a", scene.scene_date, contour_id=contour_id)
        ],
    )
    monkeypatch.setattr(service_module.storage, "upsert_scene_analytics", lambda _rows: None)

    service._get_or_build_scene_dataset(
        field_id,
        None,
        field_context,
        start_date,
        end_date,
    )

    assert captured["contour_id"] == contour_id


@pytest.mark.asyncio
async def test_get_dashboard_returns_empty_series_for_empty_period(
    service: DzzService, monkeypatch: pytest.MonkeyPatch
) -> None:
    field_id = uuid.uuid4()
    contour_id = str(uuid.uuid4())
    contour = make_contour(contour_id, "Контур A")

    async def fake_build_field_context(_field_id, _authorization, _contour_id=None):
        return FieldContext(
            contours=[contour],
            geometry=SimpleNamespace(),
            active_contour_id=_contour_id,
        )

    async def fake_build_culture_context(field_id, season_id, contours, _authorization):
        return service_module.DzzCultureContextSchema(
            field_id=field_id,
            season_id=season_id,
            source="test",
            total_contours=len(contours),
            contours_with_culture=0,
            contours=[],
        )

    monkeypatch.setattr(service, "_build_field_context", fake_build_field_context)
    monkeypatch.setattr(
        service,
        "_get_or_build_scene_dataset",
        lambda *_args, **_kwargs: ([], [], {}),
    )
    monkeypatch.setattr(
        service,
        "_build_culture_context",
        fake_build_culture_context,
    )

    dashboard = await service.get_dashboard(
        field_id,
        None,
        "Bearer test",
        contour_id=contour_id,
        date_from=datetime.now(UTC).date() - timedelta(days=3),
        date_to=datetime.now(UTC).date(),
        force_refresh=True,
    )

    assert dashboard.catalog == []
    assert dashboard.timeseries == []
    assert dashboard.summary.total_scenes == 0


@pytest.mark.asyncio
async def test_get_dashboard_with_explicit_period_uses_persisted_only_mode(
    service: DzzService, monkeypatch: pytest.MonkeyPatch
) -> None:
    field_id = uuid.uuid4()
    contour_id = str(uuid.uuid4())
    contour = make_contour(contour_id, "Контур A")
    captured = {}

    async def fake_build_field_context(_field_id, _authorization, _contour_id=None):
        return FieldContext(
            contours=[contour],
            geometry=SimpleNamespace(),
            active_contour_id=_contour_id,
        )

    async def fake_build_culture_context(field_id, season_id, contours, _authorization):
        return service_module.DzzCultureContextSchema(
            field_id=field_id,
            season_id=season_id,
            source="test",
            total_contours=len(contours),
            contours_with_culture=0,
            contours=[],
        )

    def fake_get_or_build_scene_dataset(*args, **kwargs):
        captured["use_persisted_only"] = kwargs.get("use_persisted_only")
        return [], [], {}

    monkeypatch.setattr(service, "_build_field_context", fake_build_field_context)
    monkeypatch.setattr(service, "_build_culture_context", fake_build_culture_context)
    monkeypatch.setattr(service, "_get_or_build_scene_dataset", fake_get_or_build_scene_dataset)

    await service.get_dashboard(
        field_id,
        None,
        "Bearer test",
        contour_id=contour_id,
        date_from=datetime.now(UTC).date() - timedelta(days=10),
        date_to=datetime.now(UTC).date(),
    )

    assert captured["use_persisted_only"] is True
