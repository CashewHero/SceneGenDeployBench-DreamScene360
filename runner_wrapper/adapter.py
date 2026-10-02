"""Run upstream DreamScene360 panorama training through the shared runner API."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from runner_wrapper.job_logging import run_logged_command, tee_job_output
from runner_wrapper.measurements import ResourceMonitor
from runner_wrapper.weights import configure_weights

DEFAULT_PARAMETERS = {
    "pano_width": 2048,
    "perspective_size": 512,
    "geometry_iterations": 1500,
    "iterations": 10000,
    "data_device": "cuda",
}
# Center ray is +X, image-right is -Y, and image-up is +Z in upstream code.
OUTPUT_METADATA = {
    "scene_coordinate_system": "FLU",
    "scene_scale": 1.0,
    "scene_units": "relative",
    "scene_origin": "primary_viewpoint",
}


def _parameters(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("parameters must be an object")
    unknown = set(raw) - set(DEFAULT_PARAMETERS)
    if unknown:
        raise ValueError(f"unsupported job parameters: {sorted(unknown)}")
    parameters = {**DEFAULT_PARAMETERS, **raw}
    ranges = {
        "pano_width": (256, 4096, 2),
        "perspective_size": (128, 512, 32),
        "geometry_iterations": (1, 1500, 1),
        "iterations": (1, 10000, 1),
    }
    for name, (minimum, maximum, multiple) in ranges.items():
        value = parameters[name]
        if (
            type(value) is not int
            or not minimum <= value <= maximum
            or value % multiple
        ):
            raise ValueError(
                f"{name} must be an integer in {minimum}..{maximum}, multiple of {multiple}"
            )
    if parameters["data_device"] not in ("cuda", "cpu"):
        raise ValueError(
            "data_device must be cuda or cpu; training always requires CUDA"
        )
    return parameters


def _prepare_input(request: dict[str, Any], destination: Path) -> tuple[str, Path]:
    from PIL import Image

    job = request["job"]
    if job.get("job_type") not in ("generation", "generator"):
        raise ValueError("DreamScene360 is a generator")
    inputs = request.get("inputs", {})
    if not isinstance(inputs, dict) or set(inputs) - {"data"}:
        raise ValueError("only inputs.data is supported")
    primary = job.get("primary_sample")
    samples = inputs.get("data", {})
    if not isinstance(samples, dict) or list(samples) != [primary]:
        raise ValueError("exactly one primary panorama sample is required")
    sample = samples[primary]
    if not isinstance(sample, dict) or not isinstance(sample.get("image"), str):
        raise ValueError("the primary sample requires an image path")
    source = Path(sample["image"])
    if not source.is_absolute() or not source.is_file():
        raise ValueError(f"image must be an existing absolute file path: {source}")
    metadata = job.get("primary_sample_metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("primary_sample_metadata must be an object")
    projection = metadata.get("projection", "equirectangular")
    if projection != "equirectangular":
        raise ValueError("image projection must be equirectangular")
    for name, expected in (("horizontal_fov_deg", 360), ("vertical_fov_deg", 180)):
        if name in metadata:
            value = metadata[name]
            if type(value) not in (int, float) or not math.isclose(value, expected):
                raise ValueError("a full 360 by 180 degree panorama is required")
    with Image.open(source) as image:
        width, height = image.size
        if width != 2 * height or height < 64:
            raise ValueError("image must be a full 2:1 panorama, at least 128 by 64")
        # Upstream discovers a .png inside a writable source folder.
        image.convert("RGB").save(destination)
    return primary, source


def _command(source: Path, model: Path, parameters: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        "train.py",
        "-s",
        str(source),
        "-m",
        str(model),
        "--pano_width",
        str(parameters["pano_width"]),
        "--perspective_size",
        str(parameters["perspective_size"]),
        "--geometry_iterations",
        str(parameters["geometry_iterations"]),
        "--iterations",
        str(parameters["iterations"]),
        "--data_device",
        parameters["data_device"],
        # Keep only the final export. Optimization and losses remain upstream's.
        "--save_iterations",
        str(parameters["iterations"]),
        "--test_iterations",
        str(parameters["iterations"]),
        # Disable interactive viewer connections on the headless runner.
        "--disable_gui",
    ]


def _validate_ply(path: Path) -> int:
    import numpy as np
    from plyfile import PlyData

    if not path.is_file():
        raise RuntimeError("upstream training did not export the final Gaussian PLY")
    vertices = PlyData.read(path)["vertex"].data
    names = set(vertices.dtype.names or ())
    required = {"x", "y", "z", "opacity"}
    required.update(f"f_dc_{i}" for i in range(3))
    required.update(f"scale_{i}" for i in range(3))
    required.update(f"rot_{i}" for i in range(4))
    required.update(f"f_rest_{i}" for i in range(45))
    if not len(vertices) or not required <= names:
        raise RuntimeError(
            "upstream export is empty or lacks the degree-3 Gaussian fields"
        )
    for name in names:
        if not np.isfinite(vertices[name]).all():
            raise RuntimeError(f"upstream export contains non-finite {name}")
    return len(vertices)


def _timestamp(value: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))


def run_job(request: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    workspace = Path(request["runtime"]["workspace_dir"])
    workspace.mkdir(parents=True, exist_ok=True)
    raw = request.get("job", {}).get("parameters")
    if raw is None:
        raw = {}
    digest = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:10]
    variant = f"scene-{digest}"
    log_path = workspace / f"runner-{variant}.log"
    monitor = None
    with tee_job_output(log_path):
        try:
            parameters = _parameters(raw)
            source_dir = workspace / "source"
            source_dir.mkdir()
            primary, source = _prepare_input(request, source_dir / "panorama.png")
            monitor = ResourceMonitor(
                sample_data={"image": str(source)}, output_dir=workspace
            )
            monitor.start()
            configure_weights(workspace)
            model_dir = workspace / "model"
            print(
                f"Running upstream DreamScene360 train.py with {parameters}", flush=True
            )
            run_logged_command(
                _command(source_dir, model_dir, parameters),
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "TQDM_MININTERVAL": "10"},
            )
            generated = (
                model_dir
                / "point_cloud"
                / f"iteration_{parameters['iterations']}"
                / "point_cloud.ply"
            )
            count = _validate_ply(generated)
            ply_name = f"3DGS-{variant}.ply"
            generated.rename(workspace / ply_name)
            outputs = {primary: {"3dgs": ply_name}}
            metrics = monitor.stop()
            monitor = None
            metrics.extend(
                [
                    {
                        "namespace": "model",
                        "name": "gaussian_count",
                        "type": "integer",
                        "value": count,
                        "source": "model",
                    },
                    {
                        "namespace": "model",
                        "name": "inference_steps",
                        "type": "integer",
                        "value": parameters["iterations"],
                        "source": "model",
                    },
                ]
            )
            report_name = f"metrics-{variant}.json"
            report = {
                "inputs": request["inputs"],
                "output_files": outputs,
                "parameters": parameters,
                "output_metadata": OUTPUT_METADATA,
                "resource_metrics": metrics,
            }
            (workspace / report_name).write_text(json.dumps(report, indent=2) + "\n")
            print(f"Completed with {count} Gaussians", flush=True)
            return {
                "status": "completed",
                "started_at": _timestamp(started),
                "completed_at": _timestamp(time.time()),
                "output_files": outputs,
                "output_metadata": OUTPUT_METADATA,
                "metrics": metrics,
                "artifacts": [
                    {"artifact_type": "job_log", "path": log_path.name},
                    {"artifact_type": "metric_summary", "path": report_name},
                ],
                "failure": None,
            }
        except Exception as exc:
            traceback.print_exc()
            return {
                "status": "failed",
                "started_at": _timestamp(started),
                "completed_at": _timestamp(time.time()),
                "metrics": monitor.stop() if monitor else [],
                "artifacts": [{"artifact_type": "job_log", "path": log_path.name}],
                "failure": {
                    "code": "INVALID_INPUT"
                    if isinstance(exc, ValueError)
                    else "MODEL_ERROR",
                    "message": str(exc),
                    "retryable": isinstance(exc, (OSError, TimeoutError)),
                    "stage": "adapter",
                },
            }
