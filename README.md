# agro-dzz-service

Микросервис ДЗЗ-аналитики: поиск сцен, агрегации индексов и отдача данных для фронтенда.

## Стек
- Python 3.11
- FastAPI
- Planetary Computer STAC
- Shapely / Rasterio / ODC-STAC

## Быстрый запуск
```bash
docker network create agronetwork 2>/dev/null || true
docker compose up -d --build
```

Сервис доступен на `http://localhost:8005`, Swagger - `http://localhost:8005/docs`.

## Переменные окружения
Конфигурация хранится в `.env` и подключается через `docker-compose.yml`.

## Возможности
- получение контуров поля;
- поиск спутниковых сцен;
- расчёт временных рядов NDVI/EVI/NDWI/MSAVI;
- выдача агрегированных данных для интерфейса.
