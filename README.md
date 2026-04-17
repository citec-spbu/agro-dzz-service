# DZZ microservice

Микросервис для ДЗЗ-аналитики в "Цифровом двойнике".

## Стек
- Python 3.11
- FastAPI
- Planetary Computer STAC
- odc-stac
- Shapely

## Что умеет MVP
- получать контуры поля через `api-gateway`;
- искать спутниковые сцены по полю;
- строить каталог сцен;
- считать временной ряд NDVI/EVI/NDWI/MSAVI;
- отдавать summary для фронта.

## Документация
- Пользовательское описание полей, сцен, статусов и экранов: `../docs/dzz-user-guide.md`

## Запуск
Если не создана docker-сеть `agronetwork`, то:

```bash
docker network create agronetwork
```

Из корневой папки проекта:

```bash
docker compose up -d
```

Swagger: `http://0.0.0.0:8005/docs`
