"""Download original checkpoints atomically under a shared-cache lock."""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from runner_wrapper.files import publish_file

OMNIDATA_URL = "https://www.dropbox.com/scl/fo/348s01x0trt0yxb934cwe/h?rlkey=a96g2incso7g53evzamzo0j0y&dl=1"
OMNIDATA_SHA256 = "a0fab23fee64aa9e4bbe0b520b18b196ea7594a7f719c1d8c10cf11dcb6e4a1e"
DINO_SHA256 = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
DINO_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
)


def _download(url: str, path: Path) -> None:
    print(f"Downloading {path.name}", flush=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "DeployBench-DreamScene360/0.1.0"}
    )
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        path.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle, length=1024 * 1024)


def _check_hash(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError(f"checkpoint checksum mismatch: {path}")


def configure_weights(workspace: Path) -> None:
    cache = Path(os.getenv("PATH_MODEL_CACHE", "/data/model_cache")) / "dreamscene360"
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".weights.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        omni = Path(
            os.getenv(
                "DREAMSCENE360_OMNIDATA_CHECKPOINT",
                str(cache / "omnidata_dpt_depth_v2.ckpt"),
            )
        )
        if not omni.is_file():
            if os.getenv("DREAMSCENE360_OMNIDATA_CHECKPOINT"):
                raise FileNotFoundError(
                    f"configured Omnidata checkpoint does not exist: {omni}"
                )
            archive = workspace / "omnidata-checkpoints.zip"
            extracted = workspace / "omnidata_dpt_depth_v2.ckpt"
            _download(OMNIDATA_URL, archive)
            with zipfile.ZipFile(archive) as bundle:
                matches = [
                    item
                    for item in bundle.infolist()
                    if Path(item.filename).name == omni.name
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        "upstream archive must contain exactly one Omnidata depth checkpoint"
                    )
                with bundle.open(matches[0]) as source, extracted.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
            _check_hash(extracted, OMNIDATA_SHA256)
            publish_file(extracted, omni)
            # Remove only these job-local download temporaries, not shared files.
            archive.unlink()
            extracted.unlink()
        _check_hash(omni, OMNIDATA_SHA256)
        os.environ["DREAMSCENE360_OMNIDATA_CHECKPOINT"] = str(omni)
        dino = cache / "dinov2_vitb14_pretrain.pth"
        if not dino.is_file():
            downloaded = workspace / dino.name
            _download(DINO_URL, downloaded)
            _check_hash(downloaded, DINO_SHA256)
            publish_file(downloaded, dino)
            downloaded.unlink()
        _check_hash(dino, DINO_SHA256)
        os.environ["DREAMSCENE360_DINO_CHECKPOINT"] = str(dino)
        os.environ["TORCH_HOME"] = str(cache / "torch")
