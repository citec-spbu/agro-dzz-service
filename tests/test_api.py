import uuid
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from src.main import create_app
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
)
from src.service import DzzService


class FakeDzzService:
    def __init__(self):
        self.calls = []

    async def get_dashboard(
        self,
        field_id,
        season_id,
        authorization,
        contour_id=None,
        force_refresh=False,
        date_from=None,
        date_to=None,
    ):
        self.calls.append(
            (
                "dashboard",
                field_id,
                season_id,
                authorization,
                contour_id,
                force_refresh,
                date_from,
                date_to,
            )
        )
        return DzzDashboardSchema(
            summary=DzzSummarySchema(
                field_id=field_id,
                season_id=season_id,
                generated_at=datetime.now(UTC),
                total_scenes=2,
                total_best_daily_scenes=1,
            ),
            culture_context=DzzCultureContextSchema(
                field_id=field_id,
                season_id=season_id,
                source="test",
                primary_culture="Пшеница",
                primary_cultivar="Сорт 1",
                total_contours=1,
                contours_with_culture=1,
                contours=[
                    DzzContourCultureSchema(
                        contour_id=str(uuid.uuid4()),
                        contour_name="Контур A",
                        culture="Пшеница",
                        cultivar="Сорт 1",
                        start_date=datetime.now(UTC).date(),
                    )
                ],
            ),
            catalog=[],
            timeseries=[],
        )

    async def get_culture_context(self, field_id, season_id, authorization, force_refresh=False):
        self.calls.append(("culture", field_id, season_id, authorization, force_refresh))
        return DzzCultureContextSchema(
            field_id=field_id,
            season_id=season_id,
            source="test",
            primary_culture="Пшеница",
            primary_cultivar="Сорт 1",
            total_contours=1,
            contours_with_culture=1,
            contours=[],
        )

    async def get_index_map(
        self,
        field_id,
        season_id,
        scene_id,
        index_name,
        authorization,
        contour_id=None,
        force_refresh=False,
        date_from=None,
        date_to=None,
    ):
        self.calls.append(
            (
                "index-map",
                field_id,
                season_id,
                scene_id,
                index_name,
                authorization,
                contour_id,
                force_refresh,
                date_from,
                date_to,
            )
        )
        scene = DzzSceneSchema(
            field_id=field_id,
            season_id=season_id,
            scene_date=datetime.now(UTC).date(),
            collection="sentinel-2-l2a",
            sensor="S2",
            scene_id=scene_id,
            cloud_cover=2.2,
        )
        overlay = DzzRasterOverlaySchema(
            scene_id=scene_id,
            scene_date=scene.scene_date,
            sensor=scene.sensor,
            collection=scene.collection,
            index_name=index_name,
            image_url="data:image/png;base64,test",
            bounds=[[55.0, 37.0], [55.1, 37.1]],
            display_min=-1.0,
            display_max=1.0,
            actual_min=-0.2,
            actual_max=0.6,
            mean_value=0.2,
        )
        return DzzIndexMapSchema(
            field_id=field_id,
            season_id=season_id,
            index_name=index_name,
            overlay=overlay,
        )

    async def compare_scenes(
        self,
        field_id,
        season_id,
        scene_id_a,
        scene_id_b,
        index_name,
        authorization,
        contour_id=None,
        force_refresh=False,
        date_from=None,
        date_to=None,
    ):
        self.calls.append(
            (
                "compare",
                field_id,
                season_id,
                scene_id_a,
                scene_id_b,
                index_name,
                authorization,
                contour_id,
                force_refresh,
                date_from,
                date_to,
            )
        )
        scene_a = DzzSceneSchema(
            field_id=field_id,
            season_id=season_id,
            scene_date=datetime.now(UTC).date(),
            collection="sentinel-2-l2a",
            sensor="S2",
            scene_id=scene_id_a,
            cloud_cover=1.1,
        )
        scene_b = scene_a.model_copy(update={"scene_id": scene_id_b})
        overlay = DzzRasterOverlaySchema(
            scene_id=scene_id_a,
            scene_date=scene_a.scene_date,
            sensor=scene_a.sensor,
            collection=scene_a.collection,
            index_name=index_name,
            image_url="data:image/png;base64,test",
            bounds=[[55.0, 37.0], [55.1, 37.1]],
            display_min=-1.0,
            display_max=1.0,
            actual_min=-0.3,
            actual_max=0.5,
            mean_value=0.1,
        )
        diff_overlay = overlay.model_copy(
            update={
                "scene_id": scene_id_b,
                "display_min": -0.2,
                "display_max": 0.2,
                "mode": "diff",
            }
        )
        return DzzCompareSchema(
            field_id=field_id,
            season_id=season_id,
            index_name=index_name,
            scene_a=scene_a,
            scene_b=scene_b,
            overlay_a=overlay,
            overlay_b=overlay.model_copy(update={"scene_id": scene_id_b}),
            diff_overlay=diff_overlay,
            summary=DzzCompareSummarySchema(
                mean_a=0.1,
                mean_b=0.2,
                mean_delta=0.1,
                min_delta=-0.05,
                max_delta=0.2,
            ),
        )


def test_new_dzz_routes_return_expected_payloads() -> None:
    fake_service = FakeDzzService()
    app = create_app()
    app.dependency_overrides[DzzService] = lambda: fake_service
    client = TestClient(app)

    field_id = uuid.uuid4()
    season_id = uuid.uuid4()
    headers = {"Authorization": "Bearer test"}

    culture_response = client.get(
        f"/api/dzz/{field_id}/culture-context",
        params={"seasonId": str(season_id)},
        headers=headers,
    )
    assert culture_response.status_code == 200
    assert culture_response.json()["primary_culture"] == "Пшеница"

    map_response = client.get(
        f"/api/dzz/{field_id}/index-map",
        params={"sceneId": "scene-a", "seasonId": str(season_id), "index": "ndvi"},
        headers=headers,
    )
    assert map_response.status_code == 200
    assert map_response.json()["overlay"]["image_url"].startswith("data:image/png;base64,")

    compare_response = client.get(
        f"/api/dzz/{field_id}/compare",
        params={
            "sceneIdA": "scene-a",
            "sceneIdB": "scene-b",
            "seasonId": str(season_id),
            "index": "evi",
        },
        headers=headers,
    )
    assert compare_response.status_code == 200
    assert compare_response.json()["diff_overlay"]["mode"] == "diff"

    assert any(call[0] == "culture" for call in fake_service.calls)
    assert any(call[0] == "index-map" for call in fake_service.calls)
    assert any(call[0] == "compare" for call in fake_service.calls)
