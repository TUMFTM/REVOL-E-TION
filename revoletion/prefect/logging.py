import logging
from typing import override

# Remove warnings of experimental PrefectBridgeHandler
# Becomes obsolete, once prefect logging is natively handled in revoletion.
# pyright: reportUnknownMemberType=false, reportUnusedClass=false, reportMissingParameterType=false, reportUnknownParameterType=false


class PrefectBridgeHandler(logging.Handler):
    """Routes root-logger records into the active Prefect flow-run log.

    Prefect's Docker worker only shows logs that go through its API-based
    PrefectHandler. The simulation logs via the Python root logger, which
    only reaches stdout/file — not the Prefect UI. This handler bridges the
    two systems so simulation output appears in the Prefect flow run log.
    """

    def __init__(self, prefect_logger):
        super().__init__()
        self._log = prefect_logger  # pyright: ignore[reportUnannotatedClassAttribute]

    @override
    def emit(self, record):
        # Avoid re-routing Prefect's own records back into itself.
        if record.name.startswith("prefect"):
            return
        try:
            msg = record.getMessage()
            if record.levelno >= logging.ERROR:
                self._log.error(msg)
            elif record.levelno >= logging.WARNING:
                self._log.warning(msg)
            elif record.levelno >= logging.INFO:
                self._log.info(msg)
            else:
                self._log.debug(msg)
        except Exception:  # noqa: BLE001
            self.handleError(record)
