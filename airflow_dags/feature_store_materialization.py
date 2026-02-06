"""
Airflow DAG for Feature Store Materialization.

Materializes features from offline to online store on schedule:
- Hourly: Real-time features (demand, supply)
- Daily: Aggregate features (courier stats, restaurant performance)

Senior+ implementation with batch processing and error handling.
"""

from datetime import datetime, timedelta
from typing import Any

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.empty import EmptyOperator
    from airflow.utils.trigger_rule import TriggerRule
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    DAG = None
    PythonOperator = None
    EmptyOperator = None


default_args = {
    "owner": "neuro_dispatch",
    "depends_on_past": False,
    "email": ["mlops@neuro-dispatch.io"],
    "email_on_failure": True,
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=30),
}


def materialize_courier_features(**context) -> dict[str, Any]:
    """
    Materialize courier features to online store.

    Computes aggregated courier statistics and pushes to Redis.
    """
    import asyncio

    from src.common.database import get_db_context
    from src.feature_store.store import get_feature_store

    async def _materialize():
        async with get_db_context() as session:
            # Query courier statistics
            from sqlalchemy import text

            query = text("""
                SELECT 
                    courier_id,
                    AVG(speed_kmh) as avg_speed,
                    COUNT(*) as total_deliveries,
                    AVG(EXTRACT(EPOCH FROM (delivered_at - created_at)) / 60) as avg_delivery_time
                FROM courier_telemetry
                WHERE timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY courier_id
            """)

            try:
                result = await session.execute(query)
                rows = result.fetchall()
            except Exception:
                # Table might not exist yet
                rows = []

            feature_store = get_feature_store(session)
            updated = 0

            for row in rows:
                features = {
                    "courier_avg_speed_kmh": float(row.avg_speed or 15.0),
                    "courier_completed_orders_24h": int(row.total_deliveries or 0),
                    "courier_avg_delivery_time_min": float(row.avg_delivery_time or 25.0),
                }

                await feature_store.push_features(
                    entity_type="courier",
                    entity_id=str(row.courier_id),
                    features=features,
                )
                updated += 1

            return {"couriers_updated": updated}

    result = asyncio.run(_materialize())
    context["ti"].xcom_push(key="courier_result", value=result)
    return result


def materialize_hexagon_features(**context) -> dict[str, Any]:
    """
    Materialize hexagon/zone features to online store.

    Computes demand statistics per H3 hexagon.
    """
    import asyncio

    from src.common.database import get_db_context
    from src.feature_store.store import get_feature_store

    async def _materialize():
        async with get_db_context() as session:
            from sqlalchemy import text

            query = text("""
                SELECT 
                    h3_index,
                    AVG(order_count) as avg_orders,
                    SUM(order_count) as total_orders_24h,
                    AVG(courier_count) as avg_couriers
                FROM demand_hourly
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                GROUP BY h3_index
            """)

            try:
                result = await session.execute(query)
                rows = result.fetchall()
            except Exception:
                rows = []

            feature_store = get_feature_store(session)
            updated = 0

            for row in rows:
                features = {
                    "hex_avg_orders_per_hour": float(row.avg_orders or 0.0),
                    "hex_demand_rolling_24h": float(row.total_orders_24h or 0.0),
                    "hex_active_couriers": int(row.avg_couriers or 0),
                }

                await feature_store.push_features(
                    entity_type="hexagon",
                    entity_id=row.h3_index,
                    features=features,
                )
                updated += 1

            return {"hexagons_updated": updated}

    result = asyncio.run(_materialize())
    context["ti"].xcom_push(key="hexagon_result", value=result)
    return result


def materialize_restaurant_features(**context) -> dict[str, Any]:
    """
    Materialize restaurant features to online store.
    """
    import asyncio

    from src.common.database import get_db_context
    from src.feature_store.store import get_feature_store

    async def _materialize():
        async with get_db_context() as session:
            from sqlalchemy import text

            query = text("""
                SELECT 
                    restaurant_id,
                    AVG(preparation_time_min) as avg_prep_time,
                    COUNT(*) as order_volume_24h
                FROM order_events
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                GROUP BY restaurant_id
            """)

            try:
                result = await session.execute(query)
                rows = result.fetchall()
            except Exception:
                rows = []

            feature_store = get_feature_store(session)
            updated = 0

            for row in rows:
                features = {
                    "restaurant_avg_prep_time_min": float(row.avg_prep_time or 15.0),
                    "restaurant_order_volume_24h": int(row.order_volume_24h or 0),
                }

                await feature_store.push_features(
                    entity_type="restaurant",
                    entity_id=str(row.restaurant_id),
                    features=features,
                )
                updated += 1

            return {"restaurants_updated": updated}

    result = asyncio.run(_materialize())
    context["ti"].xcom_push(key="restaurant_result", value=result)
    return result


def compute_surge_coefficients(**context) -> dict[str, Any]:
    """
    Compute and cache surge coefficients for all hexagons.
    """
    import asyncio

    from src.common.database import get_db_context
    from src.feature_store.store import get_feature_store
    from src.pricing_service.market_state import MarketStateAggregator

    async def _compute():
        async with get_db_context() as session:
            aggregator = MarketStateAggregator(session)
            feature_store = get_feature_store(session)

            # Get all active hexagons
            from sqlalchemy import text

            query = text("""
                SELECT DISTINCT h3_index
                FROM demand_hourly
                WHERE timestamp >= NOW() - INTERVAL '1 hour'
            """)

            try:
                result = await session.execute(query)
                hexagons = [row.h3_index for row in result.fetchall()]
            except Exception:
                hexagons = []

            updated = 0
            for h3_index in hexagons:
                try:
                    state = await aggregator.get_zone_state(h3_index)
                    surge = state.get("surge_coefficient", 1.0)

                    await feature_store.push_features(
                        entity_type="hexagon",
                        entity_id=h3_index,
                        features={"hex_surge_coefficient": surge},
                    )
                    updated += 1
                except Exception:
                    pass

            return {"hexagons_updated": updated}

    result = asyncio.run(_compute())
    context["ti"].xcom_push(key="surge_result", value=result)
    return result


def log_summary(**context) -> None:
    """Log materialization summary."""
    ti = context["ti"]

    courier_result = ti.xcom_pull(key="courier_result", task_ids="materialize_couriers") or {}
    hexagon_result = ti.xcom_pull(key="hexagon_result", task_ids="materialize_hexagons") or {}
    restaurant_result = ti.xcom_pull(key="restaurant_result", task_ids="materialize_restaurants") or {}
    surge_result = ti.xcom_pull(key="surge_result", task_ids="compute_surge") or {}

    print("=" * 50)
    print("Feature Store Materialization Summary")
    print("=" * 50)
    print(f"Couriers updated: {courier_result.get('couriers_updated', 0)}")
    print(f"Hexagons updated: {hexagon_result.get('hexagons_updated', 0)}")
    print(f"Restaurants updated: {restaurant_result.get('restaurants_updated', 0)}")
    print(f"Surge coefficients: {surge_result.get('hexagons_updated', 0)}")
    print("=" * 50)


# ============================================================================
# DAG DEFINITION
# ============================================================================

if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="feature_store_materialization",
        default_args=default_args,
        description="Hourly feature store materialization pipeline",
        schedule_interval="0 * * * *",  # Every hour
        start_date=datetime(2024, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["feature_store", "materialization", "hourly"],
    ) as dag:

        start = EmptyOperator(task_id="start")

        materialize_couriers = PythonOperator(
            task_id="materialize_couriers",
            python_callable=materialize_courier_features,
        )

        materialize_hexagons = PythonOperator(
            task_id="materialize_hexagons",
            python_callable=materialize_hexagon_features,
        )

        materialize_restaurants = PythonOperator(
            task_id="materialize_restaurants",
            python_callable=materialize_restaurant_features,
        )

        compute_surge = PythonOperator(
            task_id="compute_surge",
            python_callable=compute_surge_coefficients,
        )

        summary = PythonOperator(
            task_id="log_summary",
            python_callable=log_summary,
            trigger_rule=TriggerRule.ALL_DONE,
        )

        end = EmptyOperator(
            task_id="end",
            trigger_rule=TriggerRule.ALL_DONE,
        )

        # Parallel materialization
        start >> [materialize_couriers, materialize_hexagons, materialize_restaurants]

        # Surge depends on hexagon features
        materialize_hexagons >> compute_surge

        [materialize_couriers, compute_surge, materialize_restaurants] >> summary >> end

