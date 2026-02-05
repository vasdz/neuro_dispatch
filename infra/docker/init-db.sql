-- Initialize TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;


-- Enum types
CREATE TYPE order_status AS ENUM ('pending', 'assigned', 'picked_up', 'in_transit', 'delivered', 'cancelled');
CREATE TYPE courier_status AS ENUM ('offline', 'available', 'busy', 'returning');

-- Restaurants table
CREATE TABLE IF NOT EXISTS restaurants (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    h3_index VARCHAR(15) NOT NULL,
    avg_prep_time_minutes INTEGER DEFAULT 15,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Couriers table
CREATE TABLE IF NOT EXISTS couriers (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    phone VARCHAR(20),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    h3_index VARCHAR(15),
    status courier_status DEFAULT 'offline',
    current_order_id INTEGER,
    avg_speed_kmh DOUBLE PRECISION DEFAULT 15.0,
    rating DOUBLE PRECISION DEFAULT 5.0,
    total_deliveries INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(64) UNIQUE NOT NULL,
    restaurant_id INTEGER REFERENCES restaurants(id),
    courier_id INTEGER REFERENCES couriers(id),
    customer_latitude DOUBLE PRECISION NOT NULL,
    customer_longitude DOUBLE PRECISION NOT NULL,
    customer_h3_index VARCHAR(15) NOT NULL,
    status order_status DEFAULT 'pending',
    surge_coefficient DOUBLE PRECISION DEFAULT 1.0,
    estimated_prep_time_minutes INTEGER,
    estimated_delivery_time_minutes INTEGER,
    actual_delivery_time_minutes INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    assigned_at TIMESTAMPTZ,
    picked_up_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Courier telemetry table (time-series)
CREATE TABLE IF NOT EXISTS courier_telemetry (
    time TIMESTAMPTZ NOT NULL,
    courier_id INTEGER NOT NULL REFERENCES couriers(id),
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    h3_index VARCHAR(15) NOT NULL,
    speed_kmh DOUBLE PRECISION,
    heading DOUBLE PRECISION,
    battery_level INTEGER
);

-- Convert to hypertable for TimescaleDB
SELECT create_hypertable('courier_telemetry', 'time', if_not_exists => TRUE);

-- Order events table (time-series for analytics)
CREATE TABLE IF NOT EXISTS order_events (
    time TIMESTAMPTZ NOT NULL,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    event_type VARCHAR(50) NOT NULL,
    h3_index VARCHAR(15),
    event_metadata JSONB
);

SELECT create_hypertable('order_events', 'time', if_not_exists => TRUE);

-- Demand aggregation table (hourly)
CREATE TABLE IF NOT EXISTS demand_hourly (
    time TIMESTAMPTZ NOT NULL,
    h3_index VARCHAR(15) NOT NULL,
    order_count INTEGER DEFAULT 0,
    avg_surge DOUBLE PRECISION DEFAULT 1.0,
    available_couriers INTEGER DEFAULT 0,
    PRIMARY KEY (time, h3_index)
);

SELECT create_hypertable('demand_hourly', 'time', if_not_exists => TRUE);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_restaurants_h3 ON restaurants(h3_index);
CREATE INDEX IF NOT EXISTS idx_couriers_h3 ON couriers(h3_index);
CREATE INDEX IF NOT EXISTS idx_couriers_status ON couriers(status);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_h3 ON orders(customer_h3_index);
CREATE INDEX IF NOT EXISTS idx_telemetry_courier ON courier_telemetry(courier_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_demand_h3 ON demand_hourly(h3_index, time DESC);

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_restaurants_updated_at
    BEFORE UPDATE ON restaurants
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_couriers_updated_at
    BEFORE UPDATE ON couriers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_orders_updated_at
    BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

