from pathlib import Path

import pytest

from nanorouter.toolspec import load_spec


@pytest.fixture(scope="session")
def spec():
    return load_spec(Path(__file__).parents[1] / "configs" / "tools.yaml")
