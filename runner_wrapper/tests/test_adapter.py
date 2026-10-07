from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from runner_wrapper.adapter import (
    DEFAULT_PARAMETERS,
    OUTPUT_METADATA,
    _command,
    _parameters,
    _prepare_input,
    run_job,
)


class AdapterTests(unittest.TestCase):
    def test_upstream_defaults(self):
        parameters = _parameters({})
        self.assertEqual(parameters, DEFAULT_PARAMETERS)
        self.assertEqual(parameters["iterations"], 10000)
        self.assertEqual(parameters["pano_width"], 2048)
        self.assertEqual(parameters["perspective_size"], 512)
        self.assertEqual(parameters["geometry_iterations"], 1500)
        self.assertEqual(OUTPUT_METADATA["scene_coordinate_system"], "FLU")
        command = _command(Path("/input"), Path("/model"), parameters)
        self.assertIn("train.py", command)
        self.assertNotIn("--self_refinement", command)
        self.assertIn("--disable_gui", command)

    def test_calibrated_output_metadata(self):
        self.assertEqual(OUTPUT_METADATA["scene_scale"], 0.07)
        self.assertEqual(OUTPUT_METADATA["scene_coordinate_system"], "FLU")
        self.assertEqual(OUTPUT_METADATA["scene_units"], "relative")

    def test_bad_parameters(self):
        for raw in (
            {"iterations": True},
            {"iterations": 0},
            {"pano_width": 513},
            {"perspective_size": 129},
            {"geometry_iterations": 1501},
            {"data_device": "auto"},
            {"unknown": 1},
            [],
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parameters(raw)

    def test_panorama_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jpg"
            Image.new("RGB", (128, 64)).save(source)
            request = {
                "job": {"job_type": "generation", "primary_sample": "sample"},
                "inputs": {"data": {"sample": {"image": str(source)}}},
            }
            self.assertEqual(
                _prepare_input(request, root / "input.png"), ("sample", source)
            )
            request["job"]["primary_sample_metadata"] = {"projection": "perspective"}
            with self.assertRaisesRegex(ValueError, "equirectangular"):
                _prepare_input(request, root / "input.png")
            request["job"].pop("primary_sample_metadata")
            Image.new("RGB", (128, 128)).save(source)
            with self.assertRaisesRegex(ValueError, "2:1"):
                _prepare_input(request, root / "input.png")

    def test_only_one_sample_and_data_role(self):
        request = {
            "job": {"job_type": "generation", "primary_sample": "a"},
            "inputs": {"data": {"a": {}, "b": {}}},
        }
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _prepare_input(request, Path("/unused"))
        request["inputs"] = {"candidate": {}}
        with self.assertRaisesRegex(ValueError, "inputs.data"):
            _prepare_input(request, Path("/unused"))

    def test_result_publication_and_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (128, 64)).save(image)
            workspace = root / "workspace"
            request = {
                "job": {
                    "job_type": "generation",
                    "primary_sample": "sample",
                    "parameters": {},
                },
                "inputs": {"data": {"sample": {"image": str(image)}}},
                "runtime": {"workspace_dir": str(workspace)},
            }

            def train(command, **kwargs):
                self.assertEqual(
                    kwargs["env"]["TCNN_RTC_CACHE_DIR"],
                    str(workspace / "tinycudann-rtc"),
                )
                model = Path(command[command.index("-m") + 1])
                exported = model / "point_cloud/iteration_10000/point_cloud.ply"
                exported.parent.mkdir(parents=True)
                exported.write_bytes(b"test export")

            with (
                patch("runner_wrapper.adapter.configure_weights"),
                patch("runner_wrapper.adapter.run_logged_command", side_effect=train),
                patch("runner_wrapper.adapter._validate_ply", return_value=100),
            ):
                result = run_job(request)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["output_metadata"], OUTPUT_METADATA)
            self.assertEqual(set(result["output_files"]["sample"]), {"3dgs"})
            for item in result["artifacts"]:
                self.assertTrue((workspace / item["path"]).is_file())
            summary = next(
                item
                for item in result["artifacts"]
                if item["artifact_type"] == "metric_summary"
            )
            report = json.loads((workspace / summary["path"]).read_text())
            self.assertEqual(report["output_files"], result["output_files"])
            self.assertEqual(report["output_metadata"], result["output_metadata"])
            request["job"]["parameters"] = {"iterations": -1}
            result = run_job(request)
            self.assertEqual(result["failure"]["code"], "INVALID_INPUT")


if __name__ == "__main__":
    unittest.main()
