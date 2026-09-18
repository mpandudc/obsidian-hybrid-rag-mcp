import pytest
from pathlib import Path
from src.server import serialize_f32, get_db


def test_serialize_f32():
    vec = [0.1, 0.2, 0.3, -0.4]
    data = serialize_f32(vec)
    assert isinstance(data, bytes)
    assert len(data) == 4 * 4  # 4 floats * 4 bytes each


def test_get_db_non_existent():
    with pytest.raises(FileNotFoundError):
        get_db("/path/to/definitely/non_existent_vault.db")
