from types import TracebackType
from typing import Self

import httpx
from revoletion_core.model import ScenarioModel
from revoletion_core.types import RemoteObject, Result


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

    async def get_remote_object(self, remote_object_id: str) -> RemoteObject:
        response = await self._client.get(f"/objects/{remote_object_id}")
        _ = response.raise_for_status()

        return RemoteObject.model_validate(response.json())

    async def upload_result(self, result: Result) -> None:
        response = await self._client.post("/results", json=result.model_dump(mode="json"))
        _ = response.raise_for_status()

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        await self.close()
