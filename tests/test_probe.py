"""Tests for utils/probe.py — all 4 ffprobe wrapper functions."""

from unittest.mock import MagicMock, patch

import pytest

from utils import probe


# ---------------------------------------------------------------------------
# get_video_duration — uses text=False (bytes output)
# ---------------------------------------------------------------------------

class TestGetVideoDuration:
    @patch("utils.probe.subprocess.run")
    def test_normal_duration(self, mock_run):
        mock_run.return_value = MagicMock(stdout=b"12.345\n")
        result = probe.get_video_duration("video.mp4")
        assert result == pytest.approx(12.345)

    @patch("utils.probe.subprocess.run")
    def test_integer_duration(self, mock_run):
        mock_run.return_value = MagicMock(stdout=b"60")
        result = probe.get_video_duration("video.mp4")
        assert result == 60.0

    @patch("utils.probe.subprocess.run")
    def test_exception_returns_none(self, mock_run):
        mock_run.side_effect = Exception("ffprobe not found")
        result = probe.get_video_duration("video.mp4")
        assert result is None

    @patch("utils.probe.subprocess.run")
    def test_custom_ffprobe_path(self, mock_run):
        mock_run.return_value = MagicMock(stdout=b"5.0")
        probe.get_video_duration("video.mp4", ffprobe_path="/usr/local/bin/ffprobe")
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/local/bin/ffprobe"

    @patch("utils.probe.subprocess.run")
    def test_uses_bytes_mode(self, mock_run):
        """get_video_duration does NOT pass text=True."""
        mock_run.return_value = MagicMock(stdout=b"1.0")
        probe.get_video_duration("video.mp4")
        kwargs = mock_run.call_args[1]
        assert "text" not in kwargs or kwargs["text"] is False


# ---------------------------------------------------------------------------
# get_r_frame_rate — uses text=True (string output)
# ---------------------------------------------------------------------------

class TestGetRFrameRate:
    @patch("utils.probe.subprocess.run")
    def test_fraction(self, mock_run):
        mock_run.return_value = MagicMock(stdout="30000/1001\n")
        result = probe.get_r_frame_rate("video.mp4")
        # ceil(30000/1001) = ceil(29.97..) = 30
        assert result == 30

    @patch("utils.probe.subprocess.run")
    def test_integer_string(self, mock_run):
        mock_run.return_value = MagicMock(stdout="25\n")
        result = probe.get_r_frame_rate("video.mp4")
        assert result == 25

    @patch("utils.probe.subprocess.run")
    def test_empty_output_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(stdout="\n")
        result = probe.get_r_frame_rate("video.mp4")
        assert result is None

    @patch("utils.probe.subprocess.run")
    def test_zero_denominator_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(stdout="0/0\n")
        result = probe.get_r_frame_rate("video.mp4")
        assert result is None

    @patch("utils.probe.subprocess.run")
    def test_exception_returns_none(self, mock_run):
        mock_run.side_effect = Exception("fail")
        result = probe.get_r_frame_rate("video.mp4")
        assert result is None

    @patch("utils.probe.subprocess.run")
    def test_custom_ffprobe_path(self, mock_run):
        mock_run.return_value = MagicMock(stdout="30\n")
        probe.get_r_frame_rate("video.mp4", ffprobe_path="/opt/ffprobe")
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/opt/ffprobe"


# ---------------------------------------------------------------------------
# get_nb_frames — uses text=True (string output)
# ---------------------------------------------------------------------------

class TestGetNbFrames:
    @patch("utils.probe.subprocess.run")
    def test_normal(self, mock_run):
        mock_run.return_value = MagicMock(stdout="750\n")
        result = probe.get_nb_frames("video.mp4")
        assert result == 750

    @patch("utils.probe.subprocess.run")
    def test_na_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(stdout="N/A\n")
        result = probe.get_nb_frames("video.mp4")
        assert result is None

    @patch("utils.probe.subprocess.run")
    def test_empty_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(stdout="\n")
        result = probe.get_nb_frames("video.mp4")
        assert result is None

    @patch("utils.probe.subprocess.run")
    def test_exception_returns_none(self, mock_run):
        mock_run.side_effect = Exception("fail")
        result = probe.get_nb_frames("video.mp4")
        assert result is None


# ---------------------------------------------------------------------------
# get_dimensions — uses text=True (string output)
# ---------------------------------------------------------------------------

class TestGetDimensions:
    @patch("utils.probe.subprocess.run")
    def test_normal(self, mock_run):
        mock_run.return_value = MagicMock(stdout="1920x1080\n")
        result = probe.get_dimensions("video.mp4")
        assert result == (1920, 1080)

    @patch("utils.probe.subprocess.run")
    def test_no_x_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(stdout="unknown\n")
        result = probe.get_dimensions("video.mp4")
        assert result is None

    @patch("utils.probe.subprocess.run")
    def test_empty_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(stdout="\n")
        result = probe.get_dimensions("video.mp4")
        assert result is None

    @patch("utils.probe.subprocess.run")
    def test_exception_returns_none(self, mock_run):
        mock_run.side_effect = Exception("fail")
        result = probe.get_dimensions("video.mp4")
        assert result is None

    @patch("utils.probe.subprocess.run")
    def test_custom_ffprobe_path(self, mock_run):
        mock_run.return_value = MagicMock(stdout="1280x720\n")
        probe.get_dimensions("video.mp4", ffprobe_path="/custom/ffprobe")
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/custom/ffprobe"
