import hashlib
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from runner_wrapper.weights import _check_hash, configure_weights


class WeightTests(unittest.TestCase):
    def test_download_publish_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            content = b"checkpoint fixture"

            def download(url, path):
                if path.suffix == ".zip":
                    with zipfile.ZipFile(path, "w") as bundle:
                        bundle.writestr("folder/omnidata_dpt_depth_v2.ckpt", content)
                else:
                    path.write_bytes(content)

            with (
                patch.dict(
                    os.environ, {"PATH_MODEL_CACHE": str(root / "cache")}, clear=True
                ),
                patch(
                    "runner_wrapper.weights.OMNIDATA_SHA256",
                    hashlib.sha256(content).hexdigest(),
                ),
                patch(
                    "runner_wrapper.weights._download", side_effect=download
                ) as fetch,
                patch(
                    "runner_wrapper.weights.DINO_SHA256",
                    hashlib.sha256(content).hexdigest(),
                ),
            ):
                configure_weights(workspace)
                self.assertEqual(fetch.call_count, 2)
                configure_weights(workspace)
                self.assertEqual(fetch.call_count, 2)
                self.assertTrue(
                    Path(os.environ["DREAMSCENE360_OMNIDATA_CHECKPOINT"]).is_file()
                )
                self.assertTrue(
                    Path(os.environ["DREAMSCENE360_DINO_CHECKPOINT"]).is_file()
                )
            self.assertEqual(list(workspace.iterdir()), [])

    def test_rejects_wrong_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.ckpt"
            path.write_bytes(b"wrong")
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                _check_hash(path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
