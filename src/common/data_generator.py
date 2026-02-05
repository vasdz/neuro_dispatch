"""
Synthetic data generator for NeuroDispatch.

Generates realistic fake data for:
- Restaurants (distributed across city)
- Couriers (with random movement patterns)
- Orders (with time-based demand patterns)
"""

import asyncio
import random
import uuid
from datetime import datetime, timedelta
from typing import Any

import h3
import numpy as np

from src.common.config import settings
from src.common.database import get_db_context
from src.common.logging import get_logger
from src.common.models import (
    Courier,
    CourierStatus,
    Order,
    OrderStatus,
    Restaurant,
    CourierTelemetry,
    DemandHourly,
)

logger = get_logger(__name__)

# Moscow city boundaries (approximate)
CITY_BOUNDS = {
    "min_lat": 55.55,
    "max_lat": 55.95,
    "min_lng": 37.35,
    "max_lng": 37.85,
}

# Restaurant name prefixes
RESTAURANT_PREFIXES = [
    "Вкусный", "Быстрый", "Домашний", "Золотой", "Сытный",
    "Острый", "Свежий", "Горячий", "Любимый", "Уютный",
]

RESTAURANT_TYPES = [
    "Суши", "Пицца", "Бургер", "Шашлык", "Паста",
    "Вок", "Салат", "Кебаб", "Тако", "Рамен",
]

# Russian first names for couriers
COURIER_NAMES = [
    "Алексей", "Дмитрий", "Сергей", "Андрей", "Михаил",
    "Иван", "Николай", "Владимир", "Артём", "Максим",
    "Евгений", "Александр", "Денис", "Роман", "Павел",
]


def random_point_in_city() -> tuple[float, float]:
    """Generate random point within city bounds."""
    lat = random.uniform(CITY_BOUNDS["min_lat"], CITY_BOUNDS["max_lat"])
    lng = random.uniform(CITY_BOUNDS["min_lng"], CITY_BOUNDS["max_lng"])
    return lat, lng


def get_h3_index(lat: float, lng: float, resolution: int | None = None) -> str:
    """Get H3 index for coordinates."""
    res = resolution or settings.h3_resolution
    return h3.geo_to_h3(lat, lng, res)


def generate_restaurant_name() -> str:
    """Generate random restaurant name."""
    prefix = random.choice(RESTAURANT_PREFIXES)
    type_ = random.choice(RESTAURANT_TYPES)
    return f"{prefix} {type_}"


def generate_phone() -> str:
    """Generate random Russian phone number."""
    return f"+7{random.randint(900, 999)}{random.randint(1000000, 9999999)}"


class DataGenerator:
    """Generator for synthetic logistics data."""

    def __init__(
        self,
        num_restaurants: int = 50,
        num_couriers: int = 100,
        num_orders: int = 500,
    ):
        self.num_restaurants = num_restaurants
        self.num_couriers = num_couriers
        self.num_orders = num_orders
        self.restaurants: list[Restaurant] = []
        self.couriers: list[Courier] = []
        self.orders: list[Order] = []

    async def generate_all(self) -> dict[str, int]:
        """Generate all synthetic data."""
        logger.info("Starting data generation...")

        async with get_db_context() as session:
            # Generate restaurants
            await self._generate_restaurants(session)
            await session.flush()

            # Generate couriers
            await self._generate_couriers(session)
            await session.flush()

            # Generate orders
            await self._generate_orders(session)
            await session.flush()

            # Generate historical demand data
            await self._generate_demand_history(session)

            await session.commit()

        result = {
            "restaurants": len(self.restaurants),
            "couriers": len(self.couriers),
            "orders": len(self.orders),
        }

        logger.info("Data generation complete", **result)
        return result

    async def _generate_restaurants(self, session: Any) -> None:
        """Generate restaurant data."""
        logger.info(f"Generating {self.num_restaurants} restaurants...")

        for i in range(self.num_restaurants):
            lat, lng = random_point_in_city()

            restaurant = Restaurant(
                external_id=f"rest_{uuid.uuid4().hex[:8]}",
                name=generate_restaurant_name(),
                latitude=lat,
                longitude=lng,
                h3_index=get_h3_index(lat, lng),
                avg_prep_time_minutes=random.randint(10, 30),
                is_active=random.random() > 0.1,  # 90% active
            )

            session.add(restaurant)
            self.restaurants.append(restaurant)

    async def _generate_couriers(self, session: Any) -> None:
        """Generate courier data."""
        logger.info(f"Generating {self.num_couriers} couriers...")

        statuses = [
            (CourierStatus.AVAILABLE, 0.5),
            (CourierStatus.BUSY, 0.3),
            (CourierStatus.OFFLINE, 0.15),
            (CourierStatus.RETURNING, 0.05),
        ]

        for i in range(self.num_couriers):
            lat, lng = random_point_in_city()

            # Weighted random status
            status = random.choices(
                [s[0] for s in statuses],
                weights=[s[1] for s in statuses],
            )[0]

            courier = Courier(
                external_id=f"cour_{uuid.uuid4().hex[:8]}",
                name=random.choice(COURIER_NAMES),
                phone=generate_phone(),
                latitude=lat,
                longitude=lng,
                h3_index=get_h3_index(lat, lng),
                status=status,
                avg_speed_kmh=random.uniform(10, 25),
                rating=round(random.uniform(4.0, 5.0), 2),
                total_deliveries=random.randint(0, 500),
            )

            session.add(courier)
            self.couriers.append(courier)

    async def _generate_orders(self, session: Any) -> None:
        """Generate order data with realistic time patterns."""
        logger.info(f"Generating {self.num_orders} orders...")

        now = datetime.utcnow()

        for i in range(self.num_orders):
            # Random restaurant
            restaurant = random.choice(self.restaurants)

            # Customer location (near restaurant, within ~3km)
            cust_lat = restaurant.latitude + random.uniform(-0.02, 0.02)
            cust_lng = restaurant.longitude + random.uniform(-0.03, 0.03)

            # Clamp to city bounds
            cust_lat = max(CITY_BOUNDS["min_lat"], min(CITY_BOUNDS["max_lat"], cust_lat))
            cust_lng = max(CITY_BOUNDS["min_lng"], min(CITY_BOUNDS["max_lng"], cust_lng))

            # Random time in last 24 hours
            created_offset = timedelta(hours=random.uniform(0, 24))
            created_at = now - created_offset

            # Determine status based on age
            hours_ago = created_offset.total_seconds() / 3600

            if hours_ago < 0.5:
                status = random.choice([OrderStatus.PENDING, OrderStatus.ASSIGNED])
                courier_id = None
                if status == OrderStatus.ASSIGNED:
                    available = [c for c in self.couriers if c.status == CourierStatus.AVAILABLE]
                    if available:
                        courier_id = random.choice(available).id
            elif hours_ago < 2:
                status = random.choice([OrderStatus.IN_TRANSIT, OrderStatus.PICKED_UP])
                courier_id = random.choice(self.couriers).id if self.couriers else None
            else:
                status = random.choice([OrderStatus.DELIVERED, OrderStatus.CANCELLED])
                courier_id = random.choice(self.couriers).id if self.couriers else None

            # Surge based on hour of day (higher at lunch/dinner)
            hour = created_at.hour
            if 11 <= hour <= 14 or 18 <= hour <= 21:
                surge = random.uniform(1.2, 2.0)
            else:
                surge = random.uniform(1.0, 1.3)

            order = Order(
                external_id=f"ord_{uuid.uuid4().hex[:8]}",
                restaurant_id=restaurant.id,
                courier_id=courier_id,
                customer_latitude=cust_lat,
                customer_longitude=cust_lng,
                customer_h3_index=get_h3_index(cust_lat, cust_lng),
                status=status,
                surge_coefficient=round(surge, 2),
                estimated_prep_time_minutes=restaurant.avg_prep_time_minutes,
                estimated_delivery_time_minutes=random.randint(15, 45),
                created_at=created_at,
            )

            if status == OrderStatus.DELIVERED:
                order.delivered_at = created_at + timedelta(minutes=random.randint(20, 60))
                order.actual_delivery_time_minutes = random.randint(20, 60)

            session.add(order)
            self.orders.append(order)

    async def _generate_demand_history(self, session: Any) -> None:
        """Generate historical hourly demand data for ML training."""
        logger.info("Generating demand history...")

        now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

        # Get unique H3 indexes
        h3_indexes = set()
        for r in self.restaurants:
            h3_indexes.add(r.h3_index)
        for o in self.orders:
            h3_indexes.add(o.customer_h3_index)

        # Generate 7 days of hourly data
        for hours_ago in range(24 * 7):
            time = now - timedelta(hours=hours_ago)
            hour = time.hour

            # Base demand pattern
            if 11 <= hour <= 14:  # Lunch peak
                base_demand = random.randint(5, 15)
            elif 18 <= hour <= 21:  # Dinner peak
                base_demand = random.randint(8, 20)
            elif 22 <= hour or hour <= 6:  # Night
                base_demand = random.randint(0, 3)
            else:
                base_demand = random.randint(2, 8)

            for h3_idx in list(h3_indexes)[:20]:  # Limit for demo
                demand = DemandHourly(
                    time=time,
                    h3_index=h3_idx,
                    order_count=max(0, base_demand + random.randint(-3, 3)),
                    avg_surge=round(random.uniform(1.0, 1.5), 2),
                    available_couriers=random.randint(2, 10),
                )
                session.add(demand)


async def main() -> None:
    """Run data generator."""
    generator = DataGenerator(
        num_restaurants=50,
        num_couriers=100,
        num_orders=500,
    )
    result = await generator.generate_all()
    print(f"Generated: {result}")


if __name__ == "__main__":
    asyncio.run(main())

