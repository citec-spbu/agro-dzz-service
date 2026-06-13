import uuid
from datetime import UTC, date, datetime

from src import persistence as persistence_module
from src.persistence import (
    DzzPersistence,
    DzzSceneAnalyticsRecord,
    DzzSceneOverlayRecord,
)


def make_record(
    field_id: uuid.UUID,
    scene_id: str,
    scene_date: date,
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
        cloud_cover=1.5,
        ndvi=0.42,
        evi=0.31,
        ndwi=-0.15,
        msavi=0.28,
        updated_at=datetime.now(UTC),
    )


def test_persistence_filters_by_contour_and_date_range(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "dzz-test.sqlite"
    monkeypatch.setattr(
        persistence_module.settings,
        "DATABASE_URL",
        f"sqlite:///{db_path}",
    )
    storage = DzzPersistence()
    storage.init_schema()

    field_id = uuid.uuid4()
    contour_a = str(uuid.uuid4())
    contour_b = str(uuid.uuid4())
    storage.upsert_scene_analytics(
        [
            make_record(field_id, "scene-a", date(2026, 4, 1), contour_id=contour_a),
            make_record(field_id, "scene-b", date(2026, 4, 6), contour_id=contour_a),
            make_record(field_id, "scene-c", date(2026, 4, 4), contour_id=contour_b),
        ]
    )

    rows = storage.fetch_scene_analytics(
        field_id,
        None,
        contour_a,
        date(2026, 4, 2),
        date(2026, 4, 10),
    )

    assert [row.scene_id for row in rows] == ["scene-b"]


def make_overlay(
    field_id: uuid.UUID,
    scene_id: str,
    index_name: str,
    contour_id: str | None = None,
) -> DzzSceneOverlayRecord:
    return DzzSceneOverlayRecord(
        field_id=field_id,
        season_id=None,
        contour_id=contour_id,
        scene_id=scene_id,
        index_name=index_name,
        scene_date=date(2026, 4, 1),
        sensor="S2",
        collection="sentinel-2-l2a",
        mode="single",
        image_url="data:image/png;base64,AAAA",
        bounds=[[55.0, 37.0], [55.1, 37.1]],
        display_min=-1.0,
        display_max=1.0,
        actual_min=0.1,
        actual_max=0.6,
        mean_value=0.4,
        updated_at=datetime.now(UTC),
    )


def test_scene_overlay_round_trip_and_upsert(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "dzz-overlay-test.sqlite"
    monkeypatch.setattr(
        persistence_module.settings,
        "DATABASE_URL",
        f"sqlite:///{db_path}",
    )
    storage = DzzPersistence()
    storage.init_schema()

    field_id = uuid.uuid4()
    contour_id = str(uuid.uuid4())
    storage.upsert_scene_overlays([make_overlay(field_id, "scene-a", "ndvi", contour_id)])

    fetched = storage.fetch_scene_overlay(field_id, None, contour_id, "scene-a", "ndvi")
    assert fetched is not None
    assert fetched.image_url == "data:image/png;base64,AAAA"
    assert fetched.bounds == [[55.0, 37.0], [55.1, 37.1]]
    assert fetched.index_name == "ndvi"

    updated = make_overlay(field_id, "scene-a", "ndvi", contour_id)
    updated = DzzSceneOverlayRecord(
        **{**updated.__dict__, "image_url": "data:image/png;base64,BBBB"}
    )
    storage.upsert_scene_overlays([updated])

    refetched = storage.fetch_scene_overlay(field_id, None, contour_id, "scene-a", "ndvi")
    assert refetched is not None
    assert refetched.image_url == "data:image/png;base64,BBBB"

    missing = storage.fetch_scene_overlay(field_id, None, contour_id, "scene-a", "evi")
    assert missing is None
