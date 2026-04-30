import uuid

from fastapi.testclient import TestClient

from src.main import create_app
from src.service import DzzService
from tests.test_api import FakeDzzService


def test_dashboard_related_endpoints_positive() -> None:
    fake_service = FakeDzzService()
    app = create_app()
    app.dependency_overrides[DzzService] = lambda: fake_service
    client = TestClient(app)

    field_id = uuid.uuid4()
    season_id = uuid.uuid4()
    params = {
        "seasonId": str(season_id),
        "contourId": "contour-1",
        "dateFrom": "2026-04-01",
        "dateTo": "2026-04-30",
    }
    headers = {"Authorization": "Bearer test"}

    dashboard = client.get(f"/api/dzz/{field_id}", params=params, headers=headers)
    assert dashboard.status_code == 200
    assert dashboard.json()["summary"]["field_id"] == str(field_id)

    summary = client.get(f"/api/dzz/{field_id}/summary", params=params, headers=headers)
    assert summary.status_code == 200
    assert summary.json()["total_scenes"] == 2

    catalog = client.get(f"/api/dzz/{field_id}/catalog", params=params, headers=headers)
    assert catalog.status_code == 200
    assert catalog.json() == []

    timeseries = client.get(f"/api/dzz/{field_id}/timeseries", params=params, headers=headers)
    assert timeseries.status_code == 200
    assert timeseries.json() == []

    refresh = client.post(f"/api/dzz/{field_id}/refresh", params=params, headers=headers)
    assert refresh.status_code == 200
    assert refresh.json()["summary"]["field_id"] == str(field_id)

    dashboard_calls = [call for call in fake_service.calls if call[0] == "dashboard"]
    assert len(dashboard_calls) == 5
    assert any(call[5] is True for call in dashboard_calls)


def test_index_map_requires_scene_id() -> None:
    fake_service = FakeDzzService()
    app = create_app()
    app.dependency_overrides[DzzService] = lambda: fake_service
    client = TestClient(app)

    field_id = uuid.uuid4()
    response = client.get(f"/api/dzz/{field_id}/index-map")
    assert response.status_code == 422


def test_compare_requires_both_scene_ids() -> None:
    fake_service = FakeDzzService()
    app = create_app()
    app.dependency_overrides[DzzService] = lambda: fake_service
    client = TestClient(app)

    field_id = uuid.uuid4()
    response = client.get(f"/api/dzz/{field_id}/compare", params={"sceneIdA": "scene-a"})
    assert response.status_code == 422


def test_dashboard_rejects_invalid_date_query() -> None:
    fake_service = FakeDzzService()
    app = create_app()
    app.dependency_overrides[DzzService] = lambda: fake_service
    client = TestClient(app)

    field_id = uuid.uuid4()
    response = client.get(f"/api/dzz/{field_id}", params={"dateFrom": "invalid-date"})
    assert response.status_code == 422
