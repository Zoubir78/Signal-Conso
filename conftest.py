import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent / "src"))


@pytest.fixture(autouse=True)
def mock_gcp_secret():
    with patch("prefect_gcp.secret_manager.GcpSecret.load") as mock:
        mock_obj = MagicMock()
        mock_obj.get.return_value = "fake-secret-value"
        mock.return_value = mock_obj
        yield mock
