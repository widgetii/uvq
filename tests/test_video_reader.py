"""Tests for utils/video_reader.py helpers."""

import numpy as np

from utils.video_reader import _extend_array


class TestExtendArray:
    def test_extend_shorter_array(self):
        arr = bytearray(b"\x01\x02\x03")
        result = _extend_array(arr, 6)
        assert len(result) == 6
        assert result[:3] == bytearray(b"\x01\x02\x03")
        assert result[3:] == bytearray(3)  # zero-filled

    def test_truncate_longer_array(self):
        arr = bytearray(b"\x01\x02\x03\x04\x05")
        result = _extend_array(arr, 3)
        assert len(result) == 3
        assert result == bytearray(b"\x01\x02\x03")

    def test_exact_match(self):
        arr = bytearray(b"\xAA\xBB")
        result = _extend_array(arr, 2)
        assert len(result) == 2
        assert result == bytearray(b"\xAA\xBB")

    def test_extend_empty(self):
        arr = bytearray()
        result = _extend_array(arr, 4)
        assert len(result) == 4
        assert result == bytearray(4)

    def test_truncate_to_zero(self):
        arr = bytearray(b"\x01\x02")
        result = _extend_array(arr, 0)
        assert len(result) == 0


class TestNormalizationFormula:
    """Verify the normalization (uint8/255 - 0.5)*2 maps [0,255] -> [-1,1]."""

    def test_zero_maps_to_neg_one(self):
        val = np.array([0], dtype=np.uint8)
        norm = (val / 255.0 - 0.5) * 2
        assert norm[0] == pytest.approx(-1.0, abs=0.008)

    def test_255_maps_to_pos_one(self):
        val = np.array([255], dtype=np.uint8)
        norm = (val / 255.0 - 0.5) * 2
        assert norm[0] == pytest.approx(1.0, abs=0.008)

    def test_128_maps_near_zero(self):
        val = np.array([128], dtype=np.uint8)
        norm = (val / 255.0 - 0.5) * 2
        assert abs(norm[0]) < 0.01


# Need pytest for approx
import pytest  # noqa: E402
