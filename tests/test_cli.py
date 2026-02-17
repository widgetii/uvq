"""Tests for uvq_inference.py CLI parsing and dispatch."""

from unittest.mock import patch, MagicMock

import pytest

from uvq_inference import setup_parser, main


class TestSetupParser:
    def test_defaults(self):
        parser = setup_parser()
        args = parser.parse_args(["video.mp4"])
        assert args.input == "video.mp4"
        assert args.model_version == "1.5"
        assert args.transpose is False
        assert args.output == ""
        assert args.device == "cpu"
        assert args.fps == 1
        assert args.output_all_stats is False
        assert args.ffmpeg_path == "ffmpeg"
        assert args.ffprobe_path == "ffprobe"

    def test_model_version_1p0(self):
        parser = setup_parser()
        args = parser.parse_args(["video.mp4", "--model_version", "1.0"])
        assert args.model_version == "1.0"

    def test_cuda_device(self):
        parser = setup_parser()
        args = parser.parse_args(["video.mp4", "--device", "cuda"])
        assert args.device == "cuda"

    def test_mlx_device(self):
        parser = setup_parser()
        args = parser.parse_args(["video.mp4", "--device", "mlx"])
        assert args.device == "mlx"

    def test_rknn_device(self):
        parser = setup_parser()
        args = parser.parse_args(["video.mp4", "--device", "rknn"])
        assert args.device == "rknn"

    def test_transpose_flag(self):
        parser = setup_parser()
        args = parser.parse_args(["video.mp4", "--transpose"])
        assert args.transpose is True

    def test_fps_override(self):
        parser = setup_parser()
        args = parser.parse_args(["video.mp4", "--fps", "30"])
        assert args.fps == 30

    def test_output_all_stats(self):
        parser = setup_parser()
        args = parser.parse_args(["video.mp4", "--output_all_stats"])
        assert args.output_all_stats is True

    def test_custom_ffmpeg_paths(self):
        parser = setup_parser()
        args = parser.parse_args([
            "video.mp4",
            "--ffmpeg_path", "/opt/ffmpeg",
            "--ffprobe_path", "/opt/ffprobe",
        ])
        assert args.ffmpeg_path == "/opt/ffmpeg"
        assert args.ffprobe_path == "/opt/ffprobe"

    def test_invalid_model_version(self):
        parser = setup_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["video.mp4", "--model_version", "2.0"])


class TestMainDispatch:
    @patch("uvq_inference.run_single_inference")
    @patch("uvq_inference.setup_parser")
    def test_single_file_dispatches_to_single(self, mock_parser, mock_single):
        mock_args = MagicMock()
        mock_args.input = "video.mp4"
        mock_args.device = "cpu"
        mock_parser.return_value.parse_args.return_value = mock_args
        main()
        mock_single.assert_called_once_with(mock_args)

    @patch("uvq_inference.run_batch_inference")
    @patch("uvq_inference.setup_parser")
    def test_txt_file_dispatches_to_batch(self, mock_parser, mock_batch):
        mock_args = MagicMock()
        mock_args.input = "videos.txt"
        mock_args.device = "cpu"
        mock_args.gradcam = False
        mock_parser.return_value.parse_args.return_value = mock_args
        main()
        mock_batch.assert_called_once_with(mock_args)

    @patch("uvq_inference.torch")
    @patch("uvq_inference.setup_parser")
    def test_cuda_unavailable_prints_error(self, mock_parser, mock_torch, capsys):
        mock_args = MagicMock()
        mock_args.device = "cuda"
        mock_args.input = "video.mp4"
        mock_parser.return_value.parse_args.return_value = mock_args
        mock_torch.cuda.is_available.return_value = False
        main()
        captured = capsys.readouterr()
        assert "CUDA is not available" in captured.out
