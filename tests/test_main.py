"""
Tests for main API endpoints.
"""

import pytest


def test_root_endpoint(client):
    """Test root endpoint returns service info."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["service"] == "NeuroDispatch"
    assert data["version"] == "0.1.0"
    assert data["status"] == "running"


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    # May return 200 even if DB not connected
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "redis" in data

