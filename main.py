"""
NeuroDispatch - Intelligent Logistics Management System.

Main FastAPI application entry point.

Production-ready with:
- Prometheus metrics
- Health checks (liveness, readiness, startup)
- Request tracing
- Graceful shutdown
"""

from contextlib import asynccontextmanager
from typing import Any
import time
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from src.common.config import settings
from src.common.logging import get_logger
from src.common.database import check_db_connection
from src.common.redis_client import redis_client
from src.common.schemas import HealthResponse
from src.common.metrics import metrics
from src.common.health import health_checker

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info(
        "Starting NeuroDispatch API",
        env=settings.app_env,
        debug=settings.debug,
    )

    # Connect to Redis
    try:
        await redis_client.connect()
    except Exception as e:
        logger.warning("Redis connection failed, continuing without cache", error=str(e))

    yield

    # Shutdown
    logger.info("Shutting down NeuroDispatch API")
    await redis_client.disconnect()


app = FastAPI(
    title="NeuroDispatch",
    description="Intelligent Logistics Management System - Backend API",
    version="0.1.0",
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_middleware(request: Request, call_next) -> Response:
    """Add request ID, timing and metrics to all requests."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()

    # Add request ID to response headers
    response = await call_next(request)

    process_time = time.perf_counter() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time:.4f}"

    # Collect metrics
    endpoint = request.url.path
    method = request.method
    status = str(response.status_code)

    metrics.inc_counter(
        "http_requests_total",
        labels={"method": method, "endpoint": endpoint, "status": status},
    )
    metrics.observe_histogram(
        "http_request_duration_seconds",
        process_time,
        labels={"method": method, "endpoint": endpoint},
    )

    # Log request
    logger.info(
        "Request processed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(process_time * 1000, 2),
        request_id=request_id,
    )

    return response


@app.get("/", tags=["root"])
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "service": "NeuroDispatch",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    db_healthy = await check_db_connection()
    redis_healthy = await redis_client.health_check() if redis_client._client else False

    return HealthResponse(
        status="healthy" if db_healthy and redis_healthy else "degraded",
        version="0.1.0",
        database=db_healthy,
        redis=redis_healthy,
    )


@app.get("/ready", tags=["health"])
async def readiness_check() -> dict[str, Any]:
    """Readiness check for Kubernetes."""
    report = await health_checker.readiness()

    if report.status.value == "unhealthy":
        return JSONResponse(
            status_code=503,
            content=report.to_dict(),
        )

    return report.to_dict()


@app.get("/live", tags=["health"])
async def liveness_check() -> dict[str, Any]:
    """Liveness check for Kubernetes."""
    report = await health_checker.liveness()

    if report.status.value == "unhealthy":
        return JSONResponse(
            status_code=503,
            content=report.to_dict(),
        )

    return report.to_dict()


@app.get("/startup", tags=["health"])
async def startup_check() -> dict[str, Any]:
    """Startup check for Kubernetes."""
    report = await health_checker.startup()
    return report.to_dict()


@app.get("/metrics", tags=["monitoring"])
async def prometheus_metrics() -> PlainTextResponse:
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        content=metrics.export_prometheus_format(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# API Router imports
from src.dispatch_engine.api import router as dispatch_router
from src.demand_forecast.api import router as forecast_router
from src.pricing_service.api import router as pricing_router
from src.feature_store.api import router as feature_store_router
from src.eta_service.api import router as eta_router

app.include_router(dispatch_router, prefix="/api/v1/dispatch", tags=["dispatch"])
app.include_router(forecast_router, prefix="/api/v1/forecast", tags=["forecast"])
app.include_router(pricing_router, prefix="/api/v1/pricing", tags=["pricing"])
app.include_router(feature_store_router, prefix="/api/v1/features", tags=["feature_store"])
app.include_router(eta_router, prefix="/api/v1/eta", tags=["eta"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
    )
