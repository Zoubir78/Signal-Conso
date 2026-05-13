import sys
from pathlib import Path

import pytest
from prefect_gcp.credentials import GcpSecret

sys.path.insert(0, str(Path(__file__).parent / "src"))


@pytest.fixture(autouse=True)
def setup_prefect_test_blocks():
    # Ce bloc sera créé dans la DB temporaire du test
    GcpSecret(value="test-secret").save(name="prefectgcp", overwrite=True)
