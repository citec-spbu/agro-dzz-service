from httpx import AsyncClient, HTTPStatusError


class BaseHttpClient:
    def __init__(self, timeout: int = 60):
        self._timeout = timeout

    async def get_json(
        self, url: str, headers: dict[str, str] | None = None
    ) -> list | dict:
        async with AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def to_runtime_error(exc: HTTPStatusError, resource_name: str) -> RuntimeError:
        status = exc.response.status_code
        return RuntimeError(f"Failed to fetch {resource_name}: HTTP {status}")
