import uuid
from datetime import UTC, date, datetime

from src import persistence as persistence_module
from src.persistence import DzzPersistence, DzzSceneAnalyticsRecord


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
