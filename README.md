# NeuroDispatch 🚀

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://www.postgresql.org/)
[![TimescaleDB](https://img.shields.io/badge/TimescaleDB-latest-orange.svg)](https://www.timescale.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[English](#english) | [Русский](#russian)**

---

<a name="english"></a>
## 🇬🇧 English

### Overview

**NeuroDispatch** is an intelligent logistics management system — a production-ready backend for food delivery services (similar to Uber Eats, DoorDash, Yandex.Eats). The system provides real-time order dispatching, demand forecasting, and dynamic pricing capabilities.

### 🎯 Project Status: Phase 1 Complete ✅

We have successfully completed **Phase 1: Foundation & Data Layer**:

- ✅ Docker Compose infrastructure with PostgreSQL (TimescaleDB) and Redis
- ✅ Data Generator: simulates order flow and courier movement across Moscow
- ✅ Telemetry ingestion service writing courier coordinates to TimescaleDB
- ✅ H3 hexagonal grid integration for geo-spatial operations
- ✅ SQLAlchemy 2.0 async ORM models with proper enum handling
- ✅ Structured logging with structlog
- ✅ API skeleton with FastAPI

### 🏗️ Architecture

Modular monolith architecture with microservices-ready design:

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway (FastAPI)                     │
├─────────────────────────────────────────────────────────────────┤
│  Dispatch Engine  │  Demand Forecast  │  Pricing Service        │
│  (The Brain)      │  (Oracle)         │  (Balancer)             │
├─────────────────────────────────────────────────────────────────┤
│            PostgreSQL + TimescaleDB    │    Redis                │
└─────────────────────────────────────────────────────────────────┘
```

#### Core Services

| Service | Description | Status |
|---------|-------------|--------|
| **Dispatch Engine** | Order-to-courier assignment using VRP/Hungarian Algorithm | 🔄 In Progress |
| **Demand Forecast** | ML-based demand prediction per H3 hexagon | 📋 Planned |
| **Pricing Service** | Dynamic surge pricing based on supply/demand | 📋 Planned |
| **ETA Service** | Delivery time estimation with ML correction | 📋 Planned |

### 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| **Core** | Python 3.11+, FastAPI, Pydantic v2, Uvicorn |
| **Database** | PostgreSQL 15, TimescaleDB (time-series), Redis (cache) |
| **ORM** | SQLAlchemy 2.0 (async), Alembic (migrations) |
| **Geo** | H3 (Uber's hexagonal grid system) |
| **ML** | XGBoost, LightGBM, scikit-learn, MLflow |
| **Infrastructure** | Docker, Docker Compose, (K8s ready) |
| **Observability** | structlog (structured logging), Prometheus (planned) |

### 📁 Project Structure

```
neuro_dispatch/
├── src/
│   ├── common/                 # Shared utilities
│   │   ├── models/             # SQLAlchemy ORM models
│   │   ├── schemas/            # Pydantic schemas
│   │   ├── config.py           # Configuration management
│   │   ├── database.py         # Database connections
│   │   ├── redis_client.py     # Redis client
│   │   └── data_generator.py   # Synthetic data generation
│   ├── dispatch_engine/        # Order dispatching service
│   ├── demand_forecast/        # ML demand prediction
│   └── pricing_service/        # Dynamic pricing
├── infra/
│   └── docker/                 # Dockerfiles & init scripts
├── tests/                      # Test suite
├── docker-compose.yml          # Local development stack
├── pyproject.toml              # Poetry dependencies
└── Makefile                    # Development commands
```

### 🚀 Quick Start

#### Prerequisites

- Docker & Docker Compose v2+
- Python 3.11+
- Poetry (dependency management)

#### Installation

```bash
# Clone the repository
git clone https://github.com/vasdz/neuro_dispatch.git
cd neuro_dispatch

# Install dependencies
poetry install

# Start infrastructure
docker compose up -d

# Wait for services to be healthy
docker compose ps

# Generate sample data
poetry run python -m src.common.data_generator

# Run the API server
poetry run uvicorn main:app --reload
```

#### API Endpoints

Once running, access:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### 🗺️ Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Foundation & Data Layer | ✅ Complete |
| **Phase 2** | Demand Forecasting (ML Core I) | 📋 Next |
| **Phase 3** | Dynamic Pricing Algorithms | 📋 Planned |
| **Phase 4** | Dispatch Optimization (VRP/Assignment) | 📋 Planned |
| **Phase 5** | MLOps & Production Grade | 📋 Planned |

### 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run specific tests
poetry run pytest tests/test_dispatch.py -v
```

### 📊 Data Model

The system uses H3 hexagonal grids (resolution 8) for geo-spatial operations, providing:
- Consistent hexagon sizes (~0.7 km² each)
- Efficient neighbor lookups
- Seamless aggregation across zoom levels

### 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<a name="russian"></a>
## 🇷🇺 Русский

### Обзор

**NeuroDispatch** — интеллектуальная система управления логистикой. Production-ready бэкенд для сервисов доставки еды (аналог Яндекс.Еды, Uber Eats, Delivery Club). Система обеспечивает распределение заказов в реальном времени, прогнозирование спроса и динамическое ценообразование.

### 🎯 Статус проекта: Этап 1 завершён ✅

Успешно завершён **Этап 1: Фундамент и слой данных**:

- ✅ Docker Compose инфраструктура с PostgreSQL (TimescaleDB) и Redis
- ✅ Генератор данных: эмуляция потока заказов и движения курьеров по Москве
- ✅ Сервис телеметрии, записывающий координаты курьеров в TimescaleDB
- ✅ Интеграция H3 гексагональной сетки для гео-операций
- ✅ SQLAlchemy 2.0 async ORM модели с корректной обработкой enum
- ✅ Структурированное логирование через structlog
- ✅ Скелет API на FastAPI

### 🏗️ Архитектура

Модульный монолит с возможностью распила на микросервисы:

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway (FastAPI)                     │
├─────────────────────────────────────────────────────────────────┤
│  Dispatch Engine  │  Demand Forecast  │  Pricing Service        │
│  (Мозг)           │  (Оракул)         │  (Балансировщик)        │
├─────────────────────────────────────────────────────────────────┤
│            PostgreSQL + TimescaleDB    │    Redis                │
└─────────────────────────────────────────────────────────────────┘
```

#### Основные сервисы

| Сервис | Описание | Статус |
|--------|----------|--------|
| **Dispatch Engine** | Назначение курьеров на заказы (VRP, Венгерский алгоритм) | 🔄 В разработке |
| **Demand Forecast** | ML-прогнозирование спроса по H3 гексагонам | 📋 Планируется |
| **Pricing Service** | Динамическое ценообразование (surge pricing) | 📋 Планируется |
| **ETA Service** | Расчёт времени доставки с ML-корректировкой | 📋 Планируется |

### 🛠️ Технологический стек

| Категория | Технологии |
|-----------|------------|
| **Core** | Python 3.11+, FastAPI, Pydantic v2, Uvicorn |
| **База данных** | PostgreSQL 15, TimescaleDB (временные ряды), Redis (кеш) |
| **ORM** | SQLAlchemy 2.0 (async), Alembic (миграции) |
| **Гео** | H3 (гексагональная сетка Uber) |
| **ML** | XGBoost, LightGBM, scikit-learn, MLflow |
| **Инфраструктура** | Docker, Docker Compose, (готов к K8s) |
| **Observability** | structlog (структурное логирование), Prometheus (планируется) |

### 📁 Структура проекта

```
neuro_dispatch/
├── src/
│   ├── common/                 # Общие утилиты
│   │   ├── models/             # SQLAlchemy ORM модели
│   │   ├── schemas/            # Pydantic схемы
│   │   ├── config.py           # Управление конфигурацией
│   │   ├── database.py         # Подключения к БД
│   │   ├── redis_client.py     # Redis клиент
│   │   └── data_generator.py   # Генерация синтетических данных
│   ├── dispatch_engine/        # Сервис диспетчеризации
│   ├── demand_forecast/        # ML-прогнозирование спроса
│   └── pricing_service/        # Динамическое ценообразование
├── infra/
│   └── docker/                 # Dockerfile и init-скрипты
├── tests/                      # Тесты
├── docker-compose.yml          # Локальный стек разработки
├── pyproject.toml              # Poetry зависимости
└── Makefile                    # Команды разработки
```

### 🚀 Быстрый старт

#### Требования

- Docker & Docker Compose v2+
- Python 3.11+
- Poetry (менеджер зависимостей)

#### Установка

```bash
# Клонирование репозитория
git clone https://github.com/vasdz/neuro_dispatch.git
cd neuro_dispatch

# Установка зависимостей
poetry install

# Запуск инфраструктуры
docker compose up -d

# Ожидание готовности сервисов
docker compose ps

# Генерация тестовых данных
poetry run python -m src.common.data_generator

# Запуск API сервера
poetry run uvicorn main:app --reload
```

#### API Endpoints

После запуска доступны:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### 🗺️ Дорожная карта

| Этап | Описание | Статус |
|------|----------|--------|
| **Этап 1** | Фундамент и слой данных | ✅ Завершён |
| **Этап 2** | Прогнозирование спроса (ML Core I) | 📋 Следующий |
| **Этап 3** | Алгоритмы динамического ценообразования | 📋 Планируется |
| **Этап 4** | Оптимизация диспетчеризации (VRP/Assignment) | 📋 Планируется |
| **Этап 5** | MLOps и Production Grade | 📋 Планируется |

### 🧪 Тестирование

```bash
# Запуск всех тестов
poetry run pytest

# Запуск с покрытием
poetry run pytest --cov=src --cov-report=html

# Запуск конкретных тестов
poetry run pytest tests/test_dispatch.py -v
```

### 📊 Модель данных

Система использует H3 гексагональные сетки (разрешение 8) для гео-операций:
- Единообразный размер гексагонов (~0.7 км² каждый)
- Эффективный поиск соседей
- Бесшовная агрегация по уровням масштаба

### 🤝 Вклад в проект

1. Сделайте форк репозитория
2. Создайте ветку для фичи (`git checkout -b feature/amazing-feature`)
3. Закоммитьте изменения (`git commit -m 'Add amazing feature'`)
4. Запушьте в ветку (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

### 📄 Лицензия

Этот проект лицензирован под MIT License — см. файл [LICENSE](LICENSE) для деталей.

---

<p align="center">
  Made with ❤️ by the NeuroDispatch Team
</p>

