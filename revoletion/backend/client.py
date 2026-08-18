from types import TracebackType
from typing import Self

import httpx
from revoletion_core.model import ScenarioModel


class Client:
    def __init__(
        self,
        base_url: str,
        token: str,
    ) -> None:
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {token}",
            },
            timeout=httpx.Timeout(10.0),
        )

    async def get_scenario(self, scenario_hash: str) -> ScenarioModel:
        response = await self._client.get(f"/scenarios/{scenario_hash}")
        _ = response.raise_for_status()

        return ScenarioModel.model_validate(response.json())

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> bool | None:
        await self.close()
