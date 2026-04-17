import json
import logging
import logging.config
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from src.api import router as dzz_router
from src.config import settings
from src.persistence import storage
from src.planetary_sync import create_planetary_sync_scheduler

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    with Path("logs_config.json").open("r") as file:
        config = json.load(file)
    logging.config.dictConfig(config)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    setup_logging()
    storage.init_schema()
    scheduler = create_planetary_sync_scheduler()
    if scheduler is not None:
        scheduler.start()
        logger.info(
            "Planetary Computer sync scheduled daily at %02d:%02d UTC.",
            settings.PLANETARY_SYNC_HOUR_UTC,
            settings.PLANETARY_SYNC_MINUTE_UTC,
        )
    logger.info("Application startup complete.")
    yield
    if scheduler is not None and scheduler.running:
        scheduler.shutdown(wait=False)
    logger.info("Application shutdown complete.")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.TITLE,
        version=settings.VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )
    app.include_router(dzz_router)
    return app
