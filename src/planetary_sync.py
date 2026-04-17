import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import settings
from src.service import DzzService

logger = logging.getLogger(__name__)


def _run_async_sync_job() -> None:
    async def _run() -> None:
        await DzzService().run_daily_planetary_sync()

    try:
        asyncio.run(_run())
    except Exception:
        logger.exception("Planetary sync job failed.")


def create_planetary_sync_scheduler() -> AsyncIOScheduler | None:
    if not settings.PLANETARY_SYNC_ENABLED:
        logger.info("Planetary sync is disabled (APP_PLANETARY_SYNC_ENABLED=false).")
        return None

    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        _run_async_sync_job,
        CronTrigger(
            hour=settings.PLANETARY_SYNC_HOUR_UTC,
            minute=settings.PLANETARY_SYNC_MINUTE_UTC,
        ),
        id="planetary_daily_sync",
        replace_existing=True,
    )
    return scheduler
