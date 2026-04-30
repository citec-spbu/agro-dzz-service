import uuid

import pytest
from fastapi.testclient import TestClient

from src.main import create_app
from src.service import DzzService
from tests.test_api import FakeDzzService


@pytest.mark.parametrize(
    ("method", "path_template", "query", "expected_status"),
    [
        ("get", "/api/dzz/not-a-uuid", {}, 422),
        ("get", "/api/dzz/{field_id}", {"dateFrom": "bad"}, 422),
        ("get", "/api/dzz/{field_id}", {"dateTo": "bad"}, 422),
        ("get", "/api/dzz/{field_id}", {"dateFrom": "2026-04-01", "dateTo": "2026-04-30"}, 200),
        ("get", "/api/dzz/{field_id}/summary", {"seasonId": "bad-uuid"}, 422),
        ("get", "/api/dzz/{field_id}/summary", {"contourId": "x"}, 200),
        ("get", "/api/dzz/{field_id}/catalog", {"dateFrom": "2026-04-01"}, 200),
        ("get", "/api/dzz/{field_id}/timeseries", {"dateTo": "2026-04-30"}, 200),
        ("get", "/api/dzz/{field_id}/culture-context", {"seasonId": "bad-uuid"}, 422),
        ("get", "/api/dzz/{field_id}/culture-context", {"seasonId": str(uuid.uuid4())}, 200),
        ("get", "/api/dzz/{field_id}/index-map", {}, 422),
        ("get", "/api/dzz/{field_id}/index-map", {"sceneId": "scene-1"}, 200),
        ("get", "/api/dzz/{field_id}/index-map", {"sceneId": "scene-1", "seasonId": "bad-uuid"}, 422),
        ("get", "/api/dzz/{field_id}/compare", {}, 422),
        ("get", "/api/dzz/{field_id}/compare", {"sceneIdA": "a"}, 422),
        ("get", "/api/dzz/{field_id}/compare", {"sceneIdA": "a", "sceneIdB": "b"}, 200),
        ("get", "/api/dzz/{field_id}/compare", {"sceneIdA": "a", "sceneIdB": "b", "seasonId": "bad-uuid"}, 422),
        ("post", "/api/dzz/{field_id}/refresh", {"dateFrom": "bad"}, 422),
        ("post", "/api/dzz/{field_id}/refresh", {"dateFrom": "2026-04-01", "dateTo": "2026-04-30"}, 200),
        ("post", "/api/dzz/not-a-uuid/refresh", {}, 422),
    ],
)
def test_dzz_validation_matrix(
    method: str,
    path_template: str,
    query: dict[str, str],
    expected_status: int,
) -> None:
    fake_service = FakeDzzService()
    app = create_app()
    app.dependency_overrides[DzzService] = lambda: fake_service
    client = TestClient(app)

    path = path_template.format(field_id=uuid.uuid4())
    response = client.request(method.upper(), path, params=query)
    assert response.status_code == expected_status
