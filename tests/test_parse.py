import pytest
from src.cvl.data_prep import parse_line_filename

def test_parse_basic():
    assert parse_line_filename("0050-8-4.tif") == ("0050", "8", 4)

def test_parse_writer_padding_preserved():
    assert parse_line_filename("0001-1-0.tif") == ("0001", "1", 0)

def test_parse_invalid_raises():
    with pytest.raises(ValueError):
        parse_line_filename("garbage.tif")
