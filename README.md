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

### 🎯 Project Status: Phase 5 Complete ✅

We have successfully completed **Phase 5: MLOps & Production Grade**:

#### Phase 1: Foundation & Data Layer ✅
- ✅ Docker Compose infrastructure with PostgreSQL (TimescaleDB) and Redis
- ✅ Data Generator: simulates order flow and courier movement across Moscow
- ✅ Telemetry ingestion service writing courier coordinates to TimescaleDB
- ✅ H3 hexagonal grid integration for geo-spatial operations
- ✅ SQLAlchemy 2.0 async ORM models with proper enum handling
- ✅ Structured logging with structlog
- ✅ API skeleton with FastAPI

#### Phase 2: Demand Forecasting ✅
- ✅ H3 hexagonal grid with city zones classification (center/inner/middle/outer)
- ✅ Advanced feature extraction pipeline (temporal, spatial, lag, rolling features)
- ✅ XGBoost & LightGBM gradient boosting models with unified API
- ✅ Ensemble model with weighted averaging
- ✅ Training pipeline with time-series cross-validation
- ✅ Model registry with hot-reload support
- ✅ Prediction caching for performance optimization
- ✅ Batch prediction API for efficiency
- ✅ Demand heatmap generation with surge pricing
- ✅ Hotspot detection algorithm
- ✅ 35+ comprehensive unit tests with 100% pass rate

#### Phase 3: Dynamic Pricing ✅
- ✅ Multiple pricing strategies: Rule-based, ML-based, Hybrid, Time-decay
- ✅ Strategy Pattern architecture for extensibility
- ✅ Market State aggregator with supply/demand analysis
- ✅ A/B testing engine for strategy comparison
- ✅ Fairness constraints (rate limiting, price caps)
- ✅ Loyalty discounts for returning customers
- ✅ Elasticity-aware pricing for customer segments
- ✅ Zone-based pricing with location adjustments
- ✅ Time-decay pricing for aging orders
- ✅ Urgent order priority handling
- ✅ 36+ comprehensive unit tests with 100% pass rate

#### Phase 4: Dispatch Optimization ✅
- ✅ Multiple assignment algorithms: Greedy, Hungarian (Kuhn-Munkres), Auction, Batch
- ✅ Cost matrix builder with distance, time, fairness, quality factors
- ✅ Constraint handling (distance, time, capacity, vehicle type)
- ✅ Multi-objective optimization (Pareto frontier)
- ✅ Courier rebalancing recommendations
- ✅ Real-time single-order assignment
- ✅ Batch optimization for global efficiency
- ✅ Performance tracking and statistics
- ✅ scipy integration for optimal assignment
- ✅ 30+ comprehensive unit tests with 100% pass rate

#### Security Module (Senior+ Level) ✅
- ✅ JWT-based authentication with access/refresh tokens
- ✅ Password hashing with bcrypt (cost factor 12)
- ✅ Role-Based Access Control (RBAC) with granular permissions
- ✅ Token blacklisting and revocation
- ✅ Rate limiting (Token Bucket + Sliding Window algorithms)
- ✅ Input validation and sanitization (SQL injection, XSS, command injection prevention)
- ✅ Security audit logging (GDPR/SOC2/PCI-DSS compliant)
- ✅ Path traversal prevention
- ✅ 46+ comprehensive security tests with 100% pass rate

#### Phase 5: MLOps & Production Grade ✅
- ✅ **Feature Store**: Centralized feature management with online (Redis) and offline (PostgreSQL) stores
- ✅ **Feature Definitions**: 40+ features across courier, hexagon, order, restaurant, temporal groups
- ✅ **Feature Registry**: Schema versioning, validation, and lineage tracking
- ✅ **Prometheus Metrics**: Request latency, ML inference, dispatch, pricing, infrastructure metrics
- ✅ **Grafana Dashboards**: Production monitoring with 15+ panels
- ✅ **Health Checks**: Kubernetes-ready liveness, readiness, startup probes
- ✅ **Circuit Breaker**: Cascading failure prevention for health checks
- ✅ **Airflow DAGs**: Daily model retraining and hourly feature materialization pipelines
- ✅ **Multi-stage Docker Build**: Optimized production image with non-root user
- ✅ **MLflow Integration**: Experiment tracking and model versioning
- ✅ **Hyperparameter Tuning**: Optuna-based Bayesian optimization
- ✅ **Production Config**: Environment-based settings with MLOps parameters
- ✅ 50+ comprehensive MLOps tests with 100% pass rate

#### ETA Service (Chronos) ✅
- ✅ **Routing Engine**: Haversine distance + road factor simulation (OSRM/GraphHopper ready)
- ✅ **Traffic Modeling**: Rush hour, night, weekend adjustments
- ✅ **ML Correction**: Rule-based and Gradient Boosting correctors with fallback
- ✅ **Multi-phase Breakdown**: Courier→Restaurant, Preparation, Restaurant→Customer
- ✅ **Confidence Intervals**: Lower/upper bounds for delivery time
- ✅ **Transport Types**: Foot, Bike, Car, Scooter with realistic speeds
- ✅ **Weather & Queue Adjustments**: Dynamic ETA based on conditions
- ✅ **Batch API**: Efficient multi-route calculations
- ✅ 45+ comprehensive ETA tests with 100% pass rate

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
| **Dispatch Engine** | Order-to-courier assignment using VRP/Hungarian Algorithm | ✅ Complete |
| **Demand Forecast** | ML-based demand prediction per H3 hexagon | ✅ Complete |
| **Pricing Service** | Dynamic surge pricing based on supply/demand | ✅ Complete |
| **Security Module** | Authentication, authorization, rate limiting, audit | ✅ Complete |
| **Feature Store** | Centralized feature management for ML models | ✅ Complete |
| **MLOps Pipeline** | Model training, versioning, and deployment automation | ✅ Complete |
| **ETA Service** | Delivery time estimation with ML correction | ✅ Complete |

### 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| **Core** | Python 3.11+, FastAPI, Pydantic v2, Uvicorn |
| **Database** | PostgreSQL 15, TimescaleDB (time-series), Redis (cache) |
| **ORM** | SQLAlchemy 2.0 (async), Alembic (migrations) |
| **Geo** | H3 (Uber's hexagonal grid system) |
| **ML** | XGBoost, LightGBM, scikit-learn, MLflow, Optuna |
| **MLOps** | Feature Store, Airflow DAGs, Model Registry |
| **Infrastructure** | Docker, Docker Compose, Kubernetes-ready |
| **Observability** | Prometheus, Grafana, structlog |

### 📁 Project Structure

```
neuro_dispatch/
├── src/
│   ├── common/                 # Shared utilities
│   │   ├── models/             # SQLAlchemy ORM models
│   │   ├── schemas/            # Pydantic schemas
│   │   ├── security/           # Auth, RBAC, rate limiting
│   │   ├── config.py           # Configuration management
│   │   ├── database.py         # Database connections
│   │   ├── metrics.py          # Prometheus metrics
│   │   ├── health.py           # Health checks
│   │   └── redis_client.py     # Redis client
│   ├── dispatch_engine/        # Order dispatching service
│   ├── demand_forecast/        # ML demand prediction
│   ├── pricing_service/        # Dynamic pricing
│   └── feature_store/          # Centralized feature management
├── airflow_dags/               # ML pipeline DAGs
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
- **Readiness**: http://localhost:8000/ready
- **Liveness**: http://localhost:8000/live
- **Metrics**: http://localhost:8000/metrics
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### 🗺️ Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Foundation & Data Layer | ✅ Complete |
| **Phase 2** | Demand Forecasting (ML Core I) | ✅ Complete |
| **Phase 3** | Dynamic Pricing Algorithms | ✅ Complete |
| **Phase 4** | Dispatch Optimization + Security | ✅ Complete |
| **Phase 5** | MLOps & Production Grade | ✅ Complete |

### 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run specific test modules
poetry run pytest tests/test_mlops.py -v
poetry run pytest tests/test_security.py -v

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

### 🎯 Статус проекта: Этап 4 завершён ✅

Успешно завершён **Этап 4: Оптимизация диспетчеризации + Модуль безопасности**:

#### Этап 1: Фундамент и слой данных ✅
- ✅ Docker Compose инфраструктура с PostgreSQL (TimescaleDB) и Redis
- ✅ Генератор данных: эмуляция потока заказов и движения курьеров по Москве
- ✅ Сервис телеметрии, записывающий координаты курьеров в TimescaleDB
- ✅ Интеграция H3 гексагональной сетки для гео-операций
- ✅ SQLAlchemy 2.0 async ORM модели с корректной обработкой enum
- ✅ Структурированное логирование через structlog
- ✅ Скелет API на FastAPI

#### Этап 2: Прогнозирование спроса ✅
- ✅ H3 гексагональная сетка с классификацией зон города (центр/внутренняя/средняя/внешняя)
- ✅ Продвинутый пайплайн извлечения фич (временные, пространственные, лаговые, скользящие)
- ✅ XGBoost & LightGBM gradient boosting модели с унифицированным API
- ✅ Ансамблевая модель со взвешенным усреднением
- ✅ Training pipeline с time-series кросс-валидацией
- ✅ Реестр моделей с поддержкой hot-reload
- ✅ Кеширование предсказаний для оптимизации производительности
- ✅ Batch Prediction API для эффективности
- ✅ Генерация тепловой карты спроса с surge pricing
- ✅ Алгоритм обнаружения хотспотов
- ✅ 35+ комплексных unit-тестов с 100% прохождением

#### Этап 3: Динамическое ценообразование ✅
- ✅ Множество стратегий: Rule-based, ML-based, Hybrid, Time-decay
- ✅ Архитектура на паттерне Strategy для расширяемости
- ✅ Market State агрегатор с анализом спроса/предложения
- ✅ A/B тестирование для сравнения стратегий
- ✅ Fairness constraints (rate limiting, ценовые ограничения)
- ✅ Скидки за лояльность для постоянных клиентов
- ✅ Elasticity-aware pricing для сегментов клиентов
- ✅ Зональное ценообразование с учётом локации
- ✅ Time-decay pricing для устаревающих заказов
- ✅ Приоритетная обработка срочных заказов
- ✅ 36+ комплексных unit-тестов с 100% прохождением

#### Этап 4: Оптимизация диспетчеризации ✅
- ✅ Множество алгоритмов назначения: Greedy, Hungarian (Куна-Манкреса), Auction, Batch
- ✅ Построитель матрицы стоимости с учётом расстояния, времени, справедливости, качества
- ✅ Обработка ограничений (расстояние, время, ёмкость, тип транспорта)
- ✅ Многокритериальная оптимизация (фронт Парето)
- ✅ Рекомендации по ребалансировке курьеров
- ✅ Назначение одного заказа в реальном времени
- ✅ Пакетная оптимизация для глобальной эффективности
- ✅ Отслеживание производительности и статистика
- ✅ Интеграция scipy для оптимального назначения
- ✅ 30+ комплексных unit-тестов с 100% прохождением

#### Модуль безопасности (Senior+ уровень) ✅
- ✅ JWT аутентификация с access/refresh токенами
- ✅ Хеширование паролей bcrypt (cost factor 12)
- ✅ RBAC с гранулярными правами доступа
- ✅ Blacklist и отзыв токенов
- ✅ Rate limiting (Token Bucket + Sliding Window)
- ✅ Валидация и санитизация ввода (защита от SQL injection, XSS, command injection)
- ✅ Аудит безопасности (соответствие GDPR/SOC2/PCI-DSS)
- ✅ Защита от path traversal
- ✅ 46+ тестов безопасности с 100% прохождением

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
| **Dispatch Engine** | Назначение курьеров на заказы (VRP, Венгерский алгоритм) | ✅ Завершён |
| **Demand Forecast** | ML-прогнозирование спроса по H3 гексагонам | ✅ Завершён |
| **Pricing Service** | Динамическое ценообразование (surge pricing) | ✅ Завершён |
| **Security Module** | Аутентификация, авторизация, rate limiting, аудит | ✅ Завершён |
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
| **Этап 2** | Прогнозирование спроса (ML Core I) | ✅ Завершён |
| **Этап 3** | Алгоритмы динамического ценообразования | ✅ Завершён |
| **Этап 4** | Оптимизация диспетчеризации + Безопасность | ✅ Завершён |
| **Этап 5** | MLOps и Production Grade | 📋 Следующий |

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

