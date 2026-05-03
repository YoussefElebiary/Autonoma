import pytest
import polars as pl
from unittest.mock import patch, MagicMock

import sys
import os

# Ensure the src directory is in the path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

# Patch the init_state to avoid loading the real CSV when importing server
with patch("Autonoma.server.pl.read_csv", return_value=pl.DataFrame({"mock": [1, 2, 3]})):
    from Autonoma.server import STATE

@pytest.fixture(autouse=True)
def reset_server_state():
    """Reset the global STATE dictionary in server.py before each test."""
    STATE['df'] = None
    STATE['X_train'] = None
    STATE['y_train'] = None
    STATE['X_val'] = None
    STATE['y_val'] = None
    STATE['X_test'] = None
    STATE['y_test'] = None
    STATE['model_path'] = ""
    yield

@pytest.fixture
def mock_df():
    """A simple mock Polars DataFrame for testing."""
    return pl.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": ["a", "b", "a", "b", "a"]
    })
