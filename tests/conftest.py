"""
Test configuration and fixtures.
"""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_coordinates():
    """Sample Moscow coordinates."""
    return {
        "latitude": 55.7558,
        "longitude": 37.6173,
    }

