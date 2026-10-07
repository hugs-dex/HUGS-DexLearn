<h1 align="center">HUGS-DexLearn</h1>

<p align="center">Learn human grasp priors and robot grasp models for HUGS.</p>

<p align="center">
  <a href="https://github.com/hugs-dex/HUGS-Main">HUGS Project</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#documentation">Documentation</a>
</p>

- **Human grasp priors:** independently train contact-mode and conditional pose branches, then export priors for HUGS-BODex synthesis.
- **Robot grasp learning:** train, sample, and visualize grasps for Shadow and Leap-SP hands.

## Installation

Use **Linux x86_64, Python 3.10, and [uv](https://docs.astral.sh/uv/getting-started/installation/)**. GPU workflows require an NVIDIA driver, CUDA toolkit, C++ compiler, Python development headers, and OpenBLAS.

Run from the repository root:

```bash
git submodule update --init
uv venv --python 3.10 --seed .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -e '.[build]'
uv pip install --python .venv/bin/python -e '.[runtime,cuda,visualize,contract]'
```

The first installation compiles PyTorch3D and MinkowskiEngine. Keep `.venv` activated during installation. See the [installation guide](docs/installation.md) for the tested stack, optional dependencies, and troubleshooting.

Check that the configuration resolves without running a task:

```bash
python -m dexlearn.main task=sample --cfg job --resolve
```

## Data and Assets

Download the [object scenes](https://huggingface.co/datasets/MingruiYu/HUGS) and [human grasps](https://huggingface.co/datasets/MingruiYu/HUGS-Human), then prepare the [expected directory layout](docs/contracts.md):

```bash
export HUGS_DATASET_ROOT=/path/to/hugs-dataset
export HUGS_ASSET_ROOT=/path/to/robot-assets
export MANO_ROOT=/path/to/licensed-mano-models
```

- **Human training and export:** human data in `OurHumanGraspFormat/`; export also uses object scenes in `object/DGN_2k/`.
- **Human visualization:** separately licensed `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` under `MANO_ROOT`.
- **Robot workflows:** prepared robot grasp data (e.g. `human_DGN2k_full/shadow/`) and robot assets (e.g. `robot/shadow_hand/` under `HUGS_ASSET_ROOT`).

Prepare robot training inputs with [Bench dataset assembly](https://github.com/hugs-dex/HUGS-DexGraspBench/blob/main/docs/workflows.md#prepare-a-robot-training-dataset). The current downloads do not include model checkpoints, exported priors, or the synthesized robot grasp dataset. See [HUGS-Main data preparation](https://github.com/hugs-dex/HUGS-Main#data) for availability and additional asset requirements.

## Quick Start

Choose the workflow for your task. These examples train the required models; with compatible checkpoints already available, skip training and update the experiment names and checkpoint steps.

### Human Prior

Train the default Independent prior on all human objects and export contact-mode scores and hand positions/orientations for downstream synthesis. Held-out [sampling and evaluation](docs/workflows.md#train-and-evaluate) uses a separate training split and experiment.

Optionally preview the training data first (requires MANO), then open `http://localhost:8080`:

```bash
python -m dexlearn.scripts.check_human_dataloader data=humanMulti
```

Train and export:

```bash
python -m dexlearn.main task=train algo=humanMultiHierar data=humanMulti \
  data.sampling.train_split=all exp_name=human_prior_full

python -m dexlearn.main task=obj_human_prior_export \
  algo=humanMultiHierar data=humanMulti test_data=DGNMulti \
  task.score_exp_name=human_prior_full_type task.score_ckpt=000100 \
  task.pose_exp_name=human_prior_full_diffusion task.pose_ckpt=007500 \
  exp_name=human_prior_full
```

Training saves separate `_type` and `_diffusion` runs for 300 and 10,000 iterations. The export example selects checkpoints `000100` and `007500`; these are example steps, not a best-checkpoint recommendation.

View the exported priors (requires MANO):

```bash
python -m dexlearn.main task=visualize_human_prior \
  algo=humanMultiHierar data=humanMulti \
  task.prior_dir=output/humanMulti_humanMultiHierar_human_prior_full/obj_human_prior/step_007500_000100/DGN_2k/shadow_hand \
  task.visualize_mode=one_scene
```

Open `http://localhost:8080`. Flat-finger hand meshes illustrate the exported positions and orientations; HUGS-BODex uses these priors to synthesize complete robot grasps. Use these priors in [BODex human-initialized synthesis](https://github.com/hugs-dex/HUGS-BODex#human-prior). See [export options and viewer controls](docs/workflows.md#object-human-prior-train-and-export).

### Robot Grasp

With robot training data and assets prepared, train a Shadow model, sample its final checkpoint, and view the saved grasps:

```bash
python -m dexlearn.main task=train algo=robotMultiHierar \
  data=shadowMulti exp_name=shadow

python -m dexlearn.main task=sample algo=robotMultiHierar \
  data=shadowMulti test_data=shadowMulti exp_name=shadow ckpt=050000

python -m dexlearn.main task=visualize algo=robotMultiHierar \
  data=shadowMulti test_data=shadowMulti exp_name=shadow ckpt=050000
```

Open `http://localhost:8080`. For Leap-SP, replace both `data=shadowMulti` and `test_data=shadowMulti` with `leapspMulti` where present. See the [robot workflow](docs/workflows.md#robot-workflow) for sampling and visualization options.

Evaluate saved robot samples with [HUGS-DexGraspBench](https://github.com/hugs-dex/HUGS-DexGraspBench#learned-robot-grasps). Human score/pose evaluation is covered separately in the [Human workflow](docs/workflows.md#train-and-evaluate).

## Documentation

- [Installation and environment checks](docs/installation.md)
- [Training, sampling, evaluation, and visualization](docs/workflows.md)
- [Human prior architectures](docs/human_prior_architectures.md)
- [Data, prior, and checkpoint formats](docs/contracts.md)
- [Validation and known limitations](docs/validation.md)

Configuration and CPU model/export tests have been validated; full training and end-to-end GPU workflows have not yet been verified for this release.

## License

[Research use only](RESEARCH_ONLY.md). Third-party code and assets retain their original terms. See the [HUGS citation](https://github.com/hugs-dex/HUGS-Main#citation).
