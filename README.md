# HUGS-DexLearn

Learning human grasp priors and robot grasp models for HUGS. This repository provides training, sampling, evaluation, and visualization, with human prior export for HUGS-BODex.

For the overall project code structure and links to all components, see [HUGS-Main](https://github.com/hugs-dex/HUGS-Main).

- **Human grasp priors:** hierarchical modeling with Independent, Joint, Reverse, and Legacy baselines.
- **Robot grasp learning:** Shadow and Leap-SP hands, with Leap compatibility.
- **Research workflows:** scene-budget prediction and multi-GPU training and sampling.

## Installation

Use **Linux x86_64, Python 3.10, and [uv](https://docs.astral.sh/uv/getting-started/installation/)**. GPU training also requires an NVIDIA driver, CUDA toolkit, C++ compiler, Python development headers, and OpenBLAS. On Ubuntu, the development packages are `build-essential`, `python3.10-dev`, and `libopenblas-dev`.

Check `nvidia-smi` and `nvcc --version` before installing. Set `CUDA_HOME` if your toolkit is not detected automatically. The tested setup uses PyTorch 2.2.2 (CUDA 12.1), GCC 11.4, CUDA toolkit 12.4, and an RTX 4090.

Run these commands from the repository root:

```bash
git submodule update --init
uv venv --python 3.10 --seed .venv
source .venv/bin/activate

# Prepare PyTorch and the tools needed to build extensions.
uv pip install --python .venv/bin/python -e '.[build]'

# Install models, CUDA extensions, visualization, and tests.
uv pip install --python .venv/bin/python -e '.[runtime,cuda,visualize,contract]'
```

Dependencies, source revisions, editable installs, and build options are managed in `pyproject.toml`. PyTorch3D 0.7.8 is built from a pinned official source revision; no conda archive or manual unpacking is needed. The first installation compiles PyTorch3D and MinkowskiEngine and may take several minutes. Keep `.venv` activated because MinkowskiEngine's build script invokes `pip` internally.

Omit `visualize` if you do not need visualization or MANO support. For CPU model tests, omit `cuda`; PyTorch3D still requires a C++ compiler. For configuration checks only, install `'.[contract]'` and run `python -m pytest tests/public -q`; the build step and submodules are unnecessary.

### Verify

```bash
uv pip check --python .venv/bin/python
python -m dexlearn.main task=sample --cfg job --resolve
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests -q
OMP_NUM_THREADS=2 python scripts/check_environment.py --cuda --visualize
```

These checks require no dataset, checkpoint, or MANO model. They cover CPU model tests, CUDA extension operations, the HUGS point-cloud backbone, and visualization imports. Omit the corresponding check flags if you skipped optional dependencies. Obtain MANO models separately as described below.

## Data and Assets

Download and prepare the required data and assets separately, then set their paths:

```bash
export HUGS_DATASET_ROOT=/path/to/hugs-dataset
export HUGS_ASSET_ROOT=/path/to/robot-assets
export MANO_ROOT=/path/to/licensed-mano-models
```

The data root should contain `object/` and `OurHumanGraspFormat/`. Robot assets use `robot/shadow_hand/` and `robot/leap_hand/`. Tasks that use MANO require `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` under `MANO_ROOT`.

For local use, keep these exports in an ignored `.env.local` and run `source .env.local` before starting a task. The file is not loaded automatically.

Datasets, checkpoints, MANO models, and mesh archives are not bundled with the code. See [data and checkpoint formats](docs/contracts.md) for the required layout.

## Usage

The examples below assume that dependencies and input assets are available. Replace checkpoint paths with your trained models.

The two sections below are separate pipeline stages. Run the Human Prior stage when
you need human-derived initialization for downstream synthesis, and run the Robot
Grasp stage when you need a learned robot grasp model.

### Human Prior

Preview augmented training samples in Viser (requires MANO models):

```bash
python -m dexlearn.scripts.check_human_dataloader data=humanMulti
```

Open `http://localhost:8080`. Use **Next Batch** to browse and **Display → View → single**
to inspect one sample. Optional overrides: `+check_batch_size=8` and `+viser_port=8081`.

#### Full-Data Prior Export

Train both Human Prior branches on all human objects, then export priors for
downstream HUGS-BODex synthesis. The default independent training mode saves
the pose and score checkpoints under `<exp_name>_diffusion` and
`<exp_name>_type` (10,000 and 300 iterations, respectively).

```bash
python -m dexlearn.main task=train algo=humanMultiHierar data=humanMulti \
  data.sampling.train_split=all exp_name=<exp_name>

python -m dexlearn.main task=obj_human_prior_export \
  algo=humanMultiHierar data=humanMulti test_data=DGNMulti \
  task.score_exp_name=<exp_name>_type task.score_ckpt=000300 \
  task.pose_exp_name=<exp_name>_diffusion task.pose_ckpt=010000 \
  exp_name=<exp_name>
```

View exported priors in Viser (requires MANO models):

```bash
CUDA_VISIBLE_DEVICES=0 python -m dexlearn.main task=visualize_human_prior \
  algo=humanMultiHierar data=humanMulti device=cuda:0 \
  task.prior_dir=<path_to_exported_prior> task.visualize_mode=one_scene
```

Set `prior_dir` to the robot-specific directory, e.g.
`output/humanMulti_humanMultiHierar_<exp_name>/obj_human_prior/step_007500_000100/DGN_2k/shadow_hand`.
Open `http://localhost:8080`; choose a scene and **Grasp Type**, then click
**Apply Selection**. Use **Next Batch** for more poses and **Next Scene** to change
scenes. Switch to `random_objects` with `0_any` to view type scores.
Hand meshes use fixed, flat fingers to illustrate the exported positions and rotations.

#### Train and Evaluate

Use a different `<exp_name>` from the full-data run: training resumes existing
checkpoints by default. Train on the default `train` split, then sample `0_any`
scores and poses for all five grasp types on `all` human objects.
`type_eval` reports Human train/test score metrics and DGN score diagnostics
(DGN has no human labels). `diffusion_eval` measures pose recall and surface
distance on the Human `test` split. Both tasks read saved samples.

```bash
# Train separate pose and score branches on train.json.
python -m dexlearn.main task=train algo=humanMultiHierar data=humanMulti exp_name=<exp_name>

# Sample contact mode probability (scores) for all Human objects.
python -m dexlearn.main task=sample algo=humanMultiHierar data=humanMulti test_data=humanMulti \
  algo.model.train_type_only=true 'test_data.grasp_type_lst=["0_any"]' \
  test_data.test_split=all exp_name=<exp_name>_type ckpt=000300

# Sample wrist poses for all contact modes and all Human objects.
python -m dexlearn.main task=sample algo=humanMultiHierar data=humanMulti test_data=humanMulti \
  test_data.test_split=all exp_name=<exp_name>_diffusion ckpt=010000
```

View the saved samples in Viser (requires MANO models). Run each command
separately and open `http://localhost:8080`. The pose view shows generated
hands; the score view shows point clouds and the trained five-type scores.
Use the Selection panel to switch between all, train, and test objects.

```bash
# View contact-mode scores.
python -m dexlearn.main task=visualize algo=humanMultiHierar data=humanMulti test_data=humanMulti \
  exp_name=<exp_name>_type ckpt=000300 \
  task.visualize_mode=random_objects task.target_grasp_type_id=0

# View sampled wrist poses, grouped by object and grasp type.
python -m dexlearn.main task=visualize algo=humanMultiHierar data=humanMulti test_data=humanMulti \
  exp_name=<exp_name>_diffusion ckpt=010000 task.visualize_mode=one_object \
  task.human_scores_exp_name=<exp_name>_type task.human_scores_ckpt=000300
```

The pose view shows the selected object's five type-branch scores in the GUI;
these are object means from up to 20 `0_any` samples, not per-pose scores.
The type-branch sample directory is derived from the experiment name and
checkpoint. Use `task.human_scores_dir` only when the samples are stored in a
custom directory.
The score view labels show the same five scores as an ordered array.

```bash
# Evaluate scores and write an additional report without both_three.
python -m dexlearn.main task=type_eval algo=humanMultiHierar data=humanMulti test_data=humanMulti \
  exp_name=<exp_name>_type ckpt=000100 task.exclude_both_three=true

# Evaluate generated Human poses against test.json.
python -m dexlearn.main task=diffusion_eval algo=humanMultiHierar data=humanMulti test_data=humanMulti \
  exp_name=<exp_name>_diffusion ckpt=007500
```

Samples are saved under `output/humanMulti_humanMultiHierar_<branch>/tests/step_<ckpt>/`,
where `<branch>` is `<exp_name>_type` or `<exp_name>_diffusion`, with dataset and
grasp-type subdirectories. Reports are saved in `evaluation/` under the type
step directory and `diffusion_eval/` under the diffusion step directory.
For Human-only score evaluation, omit DGN sampling and set `task.run_dgn_1b=false`.

The human scale-anchor baseline uses object-scale and grasp-type statistics
from `task=stat` in the separate HumanGraspData preprocessing repository.
Before running `type_eval`, generate the JSON from the same formatted human
data and `object/valid_split/train.json`. The JSON contains both `all` and
`train` scopes; the baseline uses `train` statistics to predict distributions
from the nearest object-scale anchor on `test` objects. After setting up
HumanGraspData, write the JSON to the location expected by `type_eval`:

```bash
# Run from the HumanGraspData repository root.
python src/main.py task=stat exp_name=scale_anchor task.data_name=Ours \
  task.data_path="${HUGS_DATASET_ROOT}/OurHumanGraspFormat" \
  task.object_scale_type_distribution_path="${HUGS_DATASET_ROOT}/metadata/Ours_object_scale_type_distribution.json"
```

`type_eval` writes `evaluation_human_scale_anchor_baseline_{predictions,metrics,summary}.csv`
in the type branch's `evaluation/` directory. For an existing JSON elsewhere,
set `task.human_scale_anchor_distribution_json=/path/to/statistics.json`.
If the JSON is absent, this baseline is skipped; to omit it intentionally, set
`task.run_human_scale_anchor_baseline=false`.

### Robot Grasp

Train a robot grasp model. Use `data=leapspMulti` for Leap-SP.

```bash
python -m dexlearn.main task=train algo=robotMultiHierar data=shadowMulti exp_name=shadow
```

Sample and visualize robot grasps:

```bash
python -m dexlearn.main task=sample algo=robotMultiHierar \
  data=shadowMulti test_data=shadowMulti exp_name=shadow ckpt=/path/to/model.pth

python -m dexlearn.main task=visualize algo=robotMultiHierar \
  data=shadowMulti test_data=shadowMulti exp_name=shadow ckpt=/path/to/model.pth
```

See the [workflow guide](docs/workflows.md) for score sampling, pose evaluation,
baseline configurations, scene-budget prediction, and multi-GPU execution.

## Documentation

- [Data, prior, and checkpoint formats](docs/contracts.md)
- [Workflows and baselines](docs/workflows.md)
- [Validation and known limitations](docs/validation.md)

This release includes source code, configurations, and interface tests. Configuration and CPU model/export tests have been validated; full training and end-to-end GPU workflows have not yet been verified for this release.

## License

This code is provided for research use only; see [RESEARCH_ONLY.md](RESEARCH_ONLY.md). Third-party code and assets retain their original terms.
