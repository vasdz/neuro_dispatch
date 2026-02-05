"""
NeuroDispatch - Intelligent Logistics Management System.

Main FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from typing import Any
import time
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.common.config import settings
from src.common.logging import get_logger
from src.common.database import check_db_connection
from src.common.redis_client import redis_client
from src.common.schemas import HealthResponse

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
    """Add request ID and timing to all requests."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()

    # Add request ID to response headers
    response = await call_next(request)

    process_time = time.perf_counter() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time:.4f}"

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
    db_healthy = await check_db_connection()

    if not db_healthy:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "database": False},
        )

    return {"status": "ready", "database": True}


# API Router imports
from src.dispatch_engine.api import router as dispatch_router
from src.demand_forecast.api import router as forecast_router
from src.pricing_service.api import router as pricing_router

app.include_router(dispatch_router, prefix="/api/v1/dispatch", tags=["dispatch"])
app.include_router(forecast_router, prefix="/api/v1/forecast", tags=["forecast"])
app.include_router(pricing_router, prefix="/api/v1/pricing", tags=["pricing"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
    )
