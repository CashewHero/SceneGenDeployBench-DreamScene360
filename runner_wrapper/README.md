# DreamScene360 DeployBench runner

`dreamscene360@0.1.2` is a generator that accepts one full 2:1 equirectangular panorama `image` and produces a degree-3 Gaussian PLY as `3dgs`. The adapter stages a PNG in the job workspace and launches the repository's `train.py`. It preserves Omnidata depth prediction, panoramic hash-grid depth alignment, Gaussian optimization, and the original depth and DINOv2 regularization stages. It does not run text-to-panorama generation or GPT prompt refinement.

The distributable catalog is [config/runners/dreamscene360.yaml](config/runners/dreamscene360.yaml). Copy it to the deployment's runner-config directory. The image is `ghcr.io/cashewhero/scenegendeploybench-dreamscene360:0.1.2`. The [Runner API](docs/api.md) defines the shared HTTP and filesystem contract.

## Defaults and lighter jobs

The actual upstream code defaults are 2048×1024 panorama resolution, 512×512 perspective views, 1500 iterations in each of the two geometry-alignment phases, and 10,000 Gaussian training iterations. The upstream README's generic 30,000-iteration description does not match `arguments/OptimizationParams`.

Job parameters use those defaults:

| Parameter | Default | Optional lighter setting |
| --- | --- | --- |
| `pano_width` | 2048 | 512 or 1024, height is half the width |
| `perspective_size` | 512 | 128 or 256, must be a multiple of 32 |
| `geometry_iterations` | 1500 | Fewer iterations per phase |
| `iterations` | 10000 | Fewer Gaussian training iterations |
| `data_device` | cuda | cpu stores reference RGB images in system RAM |

Reducing `iterations` below 5401 skips the upstream novel-view regularization. Iterations after 6600 and 7800 introduce larger camera perturbations. Lower-resolution smoke output is a runtime check, not a quality benchmark. Black background, degree-3 SH, depth normalization, disabled densification, optimizer settings, and the repository's stage boundaries remain unchanged. Only final exports and final test renders are requested; intermediate exports and the interactive GUI are unnecessary in a headless runner.

Scheduling defaults are a batch of four sequential jobs, two attempts, a 120-minute job timeout, and a five-minute startup timeout. This is single-GPU code. The upstream repository has no multi-GPU training path. The catalog selects `device=0`; change the Docker `gpus` setting to select another device or its UUID. Exposing all GPUs does not distribute a job across them. Use a larger GPU for the intended defaults. The local 11GB test uses smaller images.

## Model assets

Checkpoints stay under `PATH_MODEL_CACHE/dreamscene360`, never in the image or repository. The first job automatically downloads:

- The repository's PeRF Omnidata depth checkpoint from its documented Dropbox archive. The archive is about 4.3GB and includes other files, but the runner extracts and caches only `omnidata_dpt_depth_v2.ckpt`, about 1.9GB. It checks the checkpoint SHA-256 before use.
- The official DINOv2 ViT-B/14 pretrained weights, about 350MB, also checked by SHA-256.

Downloads use a shared file lock and atomic publication. No token or OpenAI API key is required. `DREAMSCENE360_OMNIDATA_CHECKPOINT` can select an existing copy of the exact checkpoint to avoid downloading the archive. The default DINO code is pinned inside the image. `timm==0.6.13` matches the vendored Omnidata ViT interfaces. The complete depth checkpoint supplies the backbone weights, so there is no separate ImageNet download.

The image uses CUDA 12.4 and the upstream documented Torch 2.4.0. It builds the repository's depth rasterizer and simple-knn extensions, plus a pinned tiny-cuda-nn hash-grid encoding, for SM 75, 80, 86, and 89. The rasterizer includes compute-89 PTX. The multi-stage runtime image excludes compilers, training datasets, panorama diffusion dependencies, and viewers. Research/evaluation use is subject to the repository's [LICENSE.md](../LICENSE.md).

Version 0.1.1 patches the pinned tiny-cuda-nn package to honor `TCNN_RTC_CACHE_DIR`. Training uses a cache inside `runtime.workspace_dir`, not the root-owned package installation. This cache is temporary and is not published. Checkpoints still use `publish_file`, and the shared server publishes job outputs. The image uses `/tmp/tinycudann-rtc` for imports outside a job. No model settings or training losses change.

## Local build and smoke

From the repository root:

```bash
runner_wrapper/localtest.sh test
RUNNER_DATA_DIR=/mnt/sata1/deploybench runner_wrapper/localtest.sh smoke
```

The helper copies the original `data/alley_pano/alley.png` sample only when the smoke input is absent. It builds the image, starts the GPU HTTP runner, submits [examples/local_smoke_job_request.json](examples/local_smoke_job_request.json), and checks for `finished`. The smoke uses 512×256 panorama and 128px perspective views, both geometry phases at 1500 iterations, 5500 Gaussian iterations, and CPU reference-image storage. It includes 100 iterations of the real DINOv2 and novel-view depth regularization. The later larger-perturbation stages and intended full-resolution configuration are not covered by this smaller smoke.

To reuse a checkpoint already mounted under `/data`:

```bash
RUNNER_DATA_DIR=/mnt/sata1/deploybench \
DREAMSCENE360_OMNIDATA_CHECKPOINT=/data/model_cache/pano2room/checkpoints/omnidata_dpt_depth_v2.ckpt \
runner_wrapper/localtest.sh smoke
```

The helper runs as the host UID/GID, matching DeployBench's non-root Docker launcher. Mounted cache and output directories must be writable by that user. `RUNNER_USER` can override the UID/GID. Set `RUNNER_CONTAINER` to a fresh name if the default local-test container already exists. `runner_wrapper/localtest.sh down` stops it. Published files appear under `output/dreamscene360@0.1.2/smoke/sample-1` in the data mount. They include the Gaussian PLY, job log, and metrics JSON. Source panorama, intermediate COLMAP data, and downloaded archives stay job-local.

The 0.1.1 RTX 2080 Ti smoke passed as UID/GID 1000:1000 in 124 seconds with 131,072 finite Gaussians and about 5.5GiB peak GPU memory. It reused the exact cached Omnidata and official DINOv2 checkpoints. A separate tiny-cuda-nn import check passed as non-root with a read-only root filesystem. DeployBench's fr-iqa renderer loaded the exported PLY and rendered finite RGB, radial depth, and alpha. Mean alpha was 0.895.

Output metadata reports `scene_coordinate_system: FLU` and `scene_scale: 0.07`. The upstream panorama center faces +X, image-right is -Y, and image-up is +Z. The scene uses normalized relative depth, not metric units. The scale converts dataset camera displacement into scene units; it does not rescale the exported PLY or change training.

Version 0.1.2 uses a calibrated default from the full-resolution 0.1.1 run across `tartanair-pano-test`, pipeline `pipeline_20261004T023723500449_34252090`. All 100 trajectories converged. Their median best scale was 0.0683699 and geometric mean was 0.0719366, supporting the rounded default 0.07. The middle 50% ranged from about 0.038 to 0.127, so per-scene calibration remains preferable when ground-truth depth is available. Model weights, resolution, iteration counts, and training losses are unchanged.

## Integration changes

Small changes outside the wrapper expose the previously hardcoded panorama resolution, perspective size, and geometry iteration count without changing their defaults; let both Omnidata loaders use the shared checkpoint path; let DINO load pinned local code and cached official weights; avoid an unused ImageNet download before loading the full depth checkpoint; and disable the GUI socket for headless jobs. Progress updates respect tqdm's refresh interval to avoid per-step Docker log noise. The generation algorithms and training losses remain upstream's.

The release workflow runs the contract tests, then builds and publishes the GHCR image on a `v*.*.*` tag. Local build and real GPU smoke must pass before tagging.
