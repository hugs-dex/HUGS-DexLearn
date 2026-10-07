# Installation

The recommended setup is Linux x86_64 with Python 3.10 and
[uv](https://docs.astral.sh/uv/getting-started/installation/).
Run all commands below from the repository root.

## Prerequisites

GPU training and sampling require an NVIDIA driver and CUDA toolkit, plus a
C++ compiler, Python development headers, and OpenBLAS for extension builds.
On Ubuntu, the development packages are `build-essential`, `python3.10-dev`,
and `libopenblas-dev`.

Check `nvidia-smi` and `nvcc --version` before installing. Set `CUDA_HOME` to
your CUDA toolkit directory if it is not detected automatically. The tested
stack uses PyTorch 2.2.2 (CUDA 12.1), CUDA toolkit 12.4, GCC 11.4, and an RTX 4090.

## Recommended Installation

```bash
git submodule update --init
uv venv --python 3.10 --seed .venv
source .venv/bin/activate

# Prepare PyTorch and extension build tools first.
uv pip install --python .venv/bin/python -e '.[build]'

# Install models, CUDA extensions, visualization, and tests.
uv pip install --python .venv/bin/python -e '.[runtime,cuda,visualize,contract]'
```

Dependencies, source revisions, editable installs, and build options are
managed in [pyproject.toml](../pyproject.toml). PyTorch3D 0.7.8 is built from a
pinned official source revision; no conda archive or manual unpacking is
needed. The first installation compiles PyTorch3D and MinkowskiEngine and may
take several minutes. Keep `.venv` activated because MinkowskiEngine's build
script invokes `pip` internally.

## Optional Dependencies

- Omit `visualize` if you do not need visualization or MANO support.
- For CPU model tests, omit `cuda`; PyTorch3D still requires a C++ compiler.
- For configuration and interface checks only, use the minimal setup below.
  It does not require submodules or the extension build step.

```bash
uv venv --python 3.10 --seed .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -e '.[contract]'
python -m pytest tests/public -q
```

## Environment Checks

For the full installation:

```bash
uv pip check --python .venv/bin/python
python -m dexlearn.main task=sample --cfg job --resolve
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests -q
OMP_NUM_THREADS=2 python scripts/check_environment.py --cuda --visualize
```

These checks require no dataset, checkpoint, or MANO model. They cover
configuration resolution, CPU model tests, CUDA extension operations, the
HUGS point-cloud backbone, and visualization imports. The last command runs
GPU operations with `--cuda`; omit the corresponding flags if you skipped
optional dependencies. See [validation](validation.md) for recorded results
and the limits of these checks.

## Data and Local Paths

Download and prepare inputs using [Data and Assets](../README.md#data-and-assets)
and the [data contracts](contracts.md). For local use, keep path exports in
an ignored `.env.local` at the repository root:

```bash
export HUGS_DATASET_ROOT=/path/to/hugs-dataset
export HUGS_ASSET_ROOT=/path/to/robot-assets
export MANO_ROOT=/path/to/licensed-mano-models
```

Run `source .env.local` before starting a task. The file is not loaded
automatically. MANO models are obtained separately; place `MANO_RIGHT.pkl`
and `MANO_LEFT.pkl` directly under `MANO_ROOT` for Human visualization.

## Troubleshooting

- **CUDA toolkit not found:** check `nvcc --version` and `CUDA_HOME`.
  An installed NVIDIA driver alone does not provide the compilation toolkit.
- **Extension build fails:** check the compiler, Python headers, and OpenBLAS,
  then confirm that the `.[build]` installation completed before installing
  runtime extras. Keep the virtual environment active for the build.
- **Imports fail after installation:** activate `.venv` and run `uv pip check`
  with `.venv/bin/python` to check the environment used by the commands.
- **Missing data or MANO files:** check the exported roots and directory
  layout. Installing Python dependencies does not download these assets.
