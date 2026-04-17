import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import psycopg

from src.config import settings


@dataclass(frozen=True)
class DzzSceneAnalyticsRecord:
    field_id: uuid.UUID
    season_id: uuid.UUID | None
    contour_id: str | None
    scene_id: str
    scene_date: date
    collection: str
    sensor: str
    cloud_cover: float | None
    ndvi: float | None
    evi: float | None
    ndwi: float | None
    msavi: float | None
    updated_at: datetime


class DzzPersistence:
    def __init__(self) -> None:
        self._database_url = settings.DATABASE_URL
        self._legacy_sqlite_path = settings.LEGACY_SQLITE_PATH
        self._ensure_local_storage_dir()

    def init_schema(self) -> None:
        if self._is_postgres():
            self._init_postgres_schema()
            self._migrate_legacy_sqlite_if_needed()
            return

        self._init_sqlite_schema()

    def fetch_scene_analytics(
        self,
        field_id: uuid.UUID,
        season_id: uuid.UUID | None,
        contour_id: str | None,
        date_from: date,
        date_to: date,
    ) -> list[DzzSceneAnalyticsRecord]:
        scope_contour_id = contour_id or ""
        scope_season_id = str(season_id) if season_id else None

        if self._is_postgres():
            query = """
                SELECT
                    field_id,
                    season_id,
                    contour_id,
                    scene_id,
                    scene_date,
                    collection_name,
                    sensor,
                    cloud_cover,
                    ndvi,
                    evi,
                    ndwi,
                    msavi,
                    updated_at
                FROM dzz_scene_analytics
                WHERE field_id = %s
                  AND contour_id = %s
                  AND scene_date >= %s
                  AND scene_date <= %s
            """
            params: list[Any] = [str(field_id), scope_contour_id, date_from, date_to]
            if scope_season_id is None:
                query += " AND season_id IS NULL"
            else:
                query += " AND season_id = %s"
                params.append(scope_season_id)
            query += " ORDER BY scene_date ASC, COALESCE(cloud_cover, 9999) ASC, scene_id ASC"

            with self._pg_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
            return [self._to_record(row) for row in rows]

        query = """
            SELECT
                field_id,
                season_id,
                contour_id,
                scene_id,
                scene_date,
                collection_name,
                sensor,
                cloud_cover,
                ndvi,
                evi,
                ndwi,
                msavi,
                updated_at
            FROM dzz_scene_analytics
            WHERE field_id = ?
              AND contour_id = ?
              AND scene_date >= ?
              AND scene_date <= ?
        """
        params = [str(field_id), scope_contour_id, date_from.isoformat(), date_to.isoformat()]
        if scope_season_id is None:
            query += " AND season_id IS NULL"
        else:
            query += " AND season_id = ?"
            params.append(scope_season_id)
        query += " ORDER BY scene_date ASC, COALESCE(cloud_cover, 9999) ASC, scene_id ASC"

        with sqlite3.connect(self._sqlite_db_path()) as connection:
            rows = connection.execute(query, params).fetchall()
            return [self._to_record(row) for row in rows]

    def fetch_distinct_refresh_targets(
        self,
    ) -> list[tuple[uuid.UUID, uuid.UUID | None, str | None]]:
        """Ключи (поле, сезон, контур) из БД для повторной синхронизации с PC."""

        if self._is_postgres():
            query = """
                SELECT DISTINCT field_id, season_id, contour_id
                FROM dzz_scene_analytics
            """
            with self._pg_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    rows = cursor.fetchall()
            return [self._row_to_refresh_target(row) for row in rows]

        query = """
            SELECT DISTINCT field_id, season_id, contour_id
            FROM dzz_scene_analytics
        """
        with sqlite3.connect(self._sqlite_db_path()) as connection:
            rows = connection.execute(query).fetchall()
        return [self._row_to_refresh_target(row) for row in rows]

    def upsert_scene_analytics(
        self,
        rows: list[DzzSceneAnalyticsRecord],
    ) -> None:
        if not rows:
            return

        prepared_rows = [self._serialize_record(row) for row in rows]
        if self._is_postgres():
            with self._pg_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.executemany(
                        """
                        INSERT INTO dzz_scene_analytics (
                            field_id,
                            season_id,
                            contour_id,
                            scene_id,
                            scene_date,
                            collection_name,
                            sensor,
                            cloud_cover,
                            ndvi,
                            evi,
                            ndwi,
                            msavi,
                            updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT(field_id, season_id, contour_id, scene_id)
                        DO UPDATE SET
                            scene_date = excluded.scene_date,
                            collection_name = excluded.collection_name,
                            sensor = excluded.sensor,
                            cloud_cover = excluded.cloud_cover,
                            ndvi = excluded.ndvi,
                            evi = excluded.evi,
                            ndwi = excluded.ndwi,
                            msavi = excluded.msavi,
                            updated_at = excluded.updated_at
                        """,
                        prepared_rows,
                    )
                connection.commit()
            return

        sqlite_rows = [
            (
                row[0],
                row[1],
                row[2],
                row[3],
                row[4].isoformat(),
                row[5],
                row[6],
                row[7],
                row[8],
                row[9],
                row[10],
                row[11],
                row[12].isoformat(),
            )
            for row in prepared_rows
        ]
        with sqlite3.connect(self._sqlite_db_path()) as connection:
            connection.executemany(
                """
                INSERT INTO dzz_scene_analytics (
                    field_id,
                    season_id,
                    contour_id,
                    scene_id,
                    scene_date,
                    collection_name,
                    sensor,
                    cloud_cover,
                    ndvi,
                    evi,
                    ndwi,
                    msavi,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(field_id, season_id, contour_id, scene_id)
                DO UPDATE SET
                    scene_date = excluded.scene_date,
                    collection_name = excluded.collection_name,
                    sensor = excluded.sensor,
                    cloud_cover = excluded.cloud_cover,
                    ndvi = excluded.ndvi,
                    evi = excluded.evi,
                    ndwi = excluded.ndwi,
                    msavi = excluded.msavi,
                    updated_at = excluded.updated_at
                """,
                sqlite_rows,
            )
            connection.commit()

    def _init_postgres_schema(self) -> None:
        with self._pg_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS dzz_scene_analytics (
                        id BIGSERIAL PRIMARY KEY,
                        field_id TEXT NOT NULL,
                        season_id TEXT NULL,
                        contour_id TEXT NOT NULL DEFAULT '',
                        scene_id TEXT NOT NULL,
                        scene_date DATE NOT NULL,
                        collection_name TEXT NOT NULL,
                        sensor TEXT NOT NULL,
                        cloud_cover DOUBLE PRECISION NULL,
                        ndvi DOUBLE PRECISION NULL,
                        evi DOUBLE PRECISION NULL,
                        ndwi DOUBLE PRECISION NULL,
                        msavi DOUBLE PRECISION NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        UNIQUE(field_id, season_id, contour_id, scene_id)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_dzz_scene_analytics_lookup
                    ON dzz_scene_analytics (field_id, season_id, contour_id, scene_date)
                    """
                )
            connection.commit()

    def _init_sqlite_schema(self) -> None:
        with sqlite3.connect(self._sqlite_db_path()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dzz_scene_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    field_id TEXT NOT NULL,
                    season_id TEXT NULL,
                    contour_id TEXT NOT NULL DEFAULT '',
                    scene_id TEXT NOT NULL,
                    scene_date TEXT NOT NULL,
                    collection_name TEXT NOT NULL,
                    sensor TEXT NOT NULL,
                    cloud_cover REAL NULL,
                    ndvi REAL NULL,
                    evi REAL NULL,
                    ndwi REAL NULL,
                    msavi REAL NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(field_id, season_id, contour_id, scene_id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_dzz_scene_analytics_lookup
                ON dzz_scene_analytics (field_id, season_id, contour_id, scene_date)
                """
            )
            connection.commit()

    def _migrate_legacy_sqlite_if_needed(self) -> None:
        legacy_path = self._legacy_sqlite_path
        if not legacy_path:
            return

        resolved_path = Path(legacy_path)
        if not resolved_path.is_absolute():
            resolved_path = Path.cwd() / resolved_path
        if not resolved_path.exists():
            return

        with self._pg_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM dzz_scene_analytics")
                existing_rows = cursor.fetchone()[0]
            connection.commit()
        if existing_rows > 0:
            return

        with sqlite3.connect(resolved_path) as legacy_connection:
            tables = {
                row[0]
                for row in legacy_connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "dzz_scene_analytics" not in tables:
                return
            legacy_rows = legacy_connection.execute(
                """
                SELECT
                    field_id,
                    season_id,
                    contour_id,
                    scene_id,
                    scene_date,
                    collection_name,
                    sensor,
                    cloud_cover,
                    ndvi,
                    evi,
                    ndwi,
                    msavi,
                    updated_at
                FROM dzz_scene_analytics
                """
            ).fetchall()

        if not legacy_rows:
            return

        self.upsert_scene_analytics([self._to_record(row) for row in legacy_rows])

    def _pg_connection(self):
        return psycopg.connect(self._database_url)

    def _is_postgres(self) -> bool:
        normalized = self._database_url.lower()
        return normalized.startswith("postgresql://") or normalized.startswith("postgres://")

    def _sqlite_db_path(self) -> str:
        if self._database_url.startswith("sqlite:///"):
            return self._database_url.removeprefix("sqlite:///")
        raise RuntimeError("SQLite path is requested for a non-sqlite DATABASE_URL.")

    def _ensure_local_storage_dir(self) -> None:
        if self._database_url.startswith("sqlite:///"):
            Path(self._sqlite_db_path()).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        if self._legacy_sqlite_path:
            legacy_path = Path(self._legacy_sqlite_path)
            if not legacy_path.is_absolute():
                legacy_path = Path.cwd() / legacy_path
            try:
                legacy_path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                pass

    @staticmethod
    def _row_to_refresh_target(
        row: tuple,
    ) -> tuple[uuid.UUID, uuid.UUID | None, str | None]:
        raw_field, raw_season, raw_contour = row[0], row[1], row[2]
        field_id = uuid.UUID(str(raw_field))
        season_id = uuid.UUID(str(raw_season)) if raw_season else None
        contour_id: str | None = str(raw_contour) if raw_contour else None
        if contour_id == "":
            contour_id = None
        return field_id, season_id, contour_id

    @staticmethod
    def _serialize_record(row: DzzSceneAnalyticsRecord) -> tuple[Any, ...]:
        return (
            str(row.field_id),
            str(row.season_id) if row.season_id else None,
            row.contour_id or "",
            row.scene_id,
            row.scene_date,
            row.collection,
            row.sensor,
            row.cloud_cover,
            row.ndvi,
            row.evi,
            row.ndwi,
            row.msavi,
            row.updated_at,
        )

    @staticmethod
    def _to_record(row: tuple) -> DzzSceneAnalyticsRecord:
        updated_at = row[12]
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=UTC)
        else:
            updated_at = updated_at.astimezone(UTC)

        scene_date = row[4]
        if isinstance(scene_date, str):
            scene_date = date.fromisoformat(scene_date)

        return DzzSceneAnalyticsRecord(
            field_id=uuid.UUID(str(row[0])),
            season_id=uuid.UUID(str(row[1])) if row[1] else None,
            contour_id=row[2] or None,
            scene_id=row[3],
            scene_date=scene_date,
            collection=row[5],
            sensor=row[6],
            cloud_cover=row[7],
            ndvi=row[8],
            evi=row[9],
            ndwi=row[10],
            msavi=row[11],
            updated_at=updated_at,
        )


storage = DzzPersistence()
