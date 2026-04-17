import uuid

from httpx import HTTPStatusError

from src.clients.base import BaseHttpClient
from src.config import settings
from src.schemas.fields import ContourSchema, CropRotationSchema, InternalFieldRefSchema


class FieldContoursClient(BaseHttpClient):
    async def get_field_contours(
        self, field_id: uuid.UUID, authorization: str | None
    ) -> list[ContourSchema]:
        headers = {"Content-Type": "application/json"}
        if authorization:
            headers["Authorization"] = authorization

        url = f"{settings.GATEWAY_URL}/api/fields-service/fields/{field_id}/contours"
        try:
            response_json = await self.get_json(url, headers=headers)
        except HTTPStatusError as exc:
            raise self.to_runtime_error(exc, "field contours") from exc

        return [ContourSchema.model_validate(item) for item in response_json]

    async def get_contour_crop_rotations(
        self, contour_id: str, authorization: str | None
    ) -> list[CropRotationSchema]:
        headers = {"Content-Type": "application/json"}
        if authorization:
            headers["Authorization"] = authorization

        url = f"{settings.GATEWAY_URL}/api/fields-service/contours/{contour_id}/crop-rotations"
        try:
            response_json = await self.get_json(url, headers=headers)
        except HTTPStatusError as exc:
            raise self.to_runtime_error(exc, "contour crop rotations") from exc

        return [CropRotationSchema.model_validate(item) for item in response_json]

    async def list_all_fields_internal(self) -> list[InternalFieldRefSchema]:
        url = (
            f"{settings.FIELDS_INTERNAL_BASE_URL}"
            "/api/internal/fields-service/fields/all-coordinates"
        )
        try:
            response_json = await self.get_json(url, headers={"Content-Type": "application/json"})
        except HTTPStatusError as exc:
            raise self.to_runtime_error(exc, "internal field list") from exc

        if not isinstance(response_json, list):
            raise RuntimeError("internal field list: expected JSON array")
        return [InternalFieldRefSchema.model_validate(item) for item in response_json]

    async def get_field_contours_internal(self, field_id: uuid.UUID) -> list[ContourSchema]:
        url = (
            f"{settings.FIELDS_INTERNAL_BASE_URL}"
            f"/api/internal/fields-service/fields/{field_id}/contours"
        )
        try:
            response_json = await self.get_json(url, headers={"Content-Type": "application/json"})
        except HTTPStatusError as exc:
            raise self.to_runtime_error(exc, "internal field contours") from exc

        if not isinstance(response_json, list):
            raise RuntimeError("internal field contours: expected JSON array")
        return [ContourSchema.model_validate(item) for item in response_json]
