# Workflow Guide

Prepare the environment with the [installation guide](installation.md) and
inputs with the [data contracts](contracts.md). These are usage instructions;
see [validation](validation.md) for the workflows actually verified in this release.

## Algorithm and Data Configurations

| Algorithm | Data |
| --- | --- |
| nflow, diffusion | bodex_shadow or the corresponding older robot data config |
| humanNflow, humanDiffusion | human |
| humanBiDiffusion | humanbi |
| humanMultiDiffusion, humanMultiHierar, humanMultiJoint, humanMultiReverse | humanMulti |
| robotMultiHierar | shadowMulti, leapspMulti, leapMulti |

Older single-hand and bimanual data configurations are not interchangeable
with the `humanMulti` interface.

## Multi-GPU Sampling

```bash
python dexlearn/scripts/launch_multi_sample.py --exp-names example --gpus 0 1 --dry-run
```

Use `--common-extra-overrides`, `--score-extra-overrides`, and
`--pose-extra-overrides` for experiment-specific settings. GPU numbers select
physical devices through `CUDA_VISIBLE_DEVICES`; each child uses `device=cuda:0`.
The dry run prints commands without creating logs or running GPU tasks.
Actual sampling reports worker failures and returns a nonzero exit status.
Train with the `python -m dexlearn.main task=train` commands below.

## Arguments and Outputs

- `exp_name`: experiment name used in output paths.
- `data` and `test_data`: training and test dataset configuration names.
- `ckpt`: checkpoint step (e.g. `007500`) or checkpoint file path.
- Training checkpoints: `output/<data>_<algo>_<exp_name>/ckpts/`.
- Saved samples: `output/<data>_<algo>_<exp_name>/tests/step_<ckpt>/`.

`visualize`, `type_eval`, and `diffusion_eval` read saved samples. Run
`task=sample` with the same experiment and checkpoint first.
Append `--cfg job --resolve` to a main command to inspect its configuration
without running the task.

## Robot Workflow

Examples use Shadow. For Leap-SP, replace both `data=shadowMulti` and
`test_data=shadowMulti` with `leapspMulti` where present; `leapMulti` remains
available for compatibility. Robot training grasps and assets must be prepared
separately, as described in [Data and Assets](../README.md#data-and-assets).

### Check Dataloader

Inspect robot samples before training. The script buffers samples to display
a fixed grasp-type order (`1 2 3 4 5`) when available.

```bash
python tests/check_robot_dataloader.py data=shadowMulti exp_name=shadow
```

### Train

`robotMultiHierar` uses `RobotHierarchicalModel`, binary contact-mode
availability prediction, and `single_stage` training. Existing robot checkpoint
parameter names and shapes are preserved.

```bash
python -m dexlearn.main task=train algo=robotMultiHierar data=shadowMulti exp_name=shadow
```

### Sample

The default training schedule runs for 50,000 iterations. Sample that checkpoint:

```bash
python -m dexlearn.main task=sample algo=robotMultiHierar \
  data=shadowMulti test_data=shadowMulti exp_name=shadow ckpt=050000
```

For `grasp_type_id=0`, the model samples its predicted available real grasp types.
The default availability threshold is `0.9`; override it with
`algo.model.type_availability.score_threshold=0.35` if needed.
The default selection keeps 10 candidates per available type.

### Visualize

```bash
python -m dexlearn.main task=visualize algo=robotMultiHierar \
  data=shadowMulti test_data=shadowMulti exp_name=shadow ckpt=050000
```

The default Viser viewer opens at `http://localhost:8080`. Its Selection panel
switches views and applies object or grasp-type selections. Set
`task.viser_port=8081` to use another port. Useful overrides:

- `task.visualize_mode=random_objects task.max_grasps=20`
- `task.visualize_mode=one_scene` (select a scene in the GUI)
- `task.visualize_mode=grasp_type task.target_grasp_type_id=1`

## Human Workflow

The default `humanMultiHierar` prior trains independent type and conditional
pose branches. [Human prior architectures](human_prior_architectures.md)
covers the available model options and architecture-specific commands.

### Preprocess

When using `data.hand_pos_source=index_mcp` (the default), input grasps must
contain `index_mcp_pos`. If absent, compute it with MANO before training.
This command modifies the source grasp files; use a working copy of the data.

```bash
python -m dexlearn.main task=human_preprocess data=humanMulti exp_name=human_preprocess
```

### Check Dataloader

Preview augmented training samples, using the configured `hand_pos_source`.
This is a data preview and requires MANO models, but no checkpoint.

```bash
python -m dexlearn.scripts.check_human_dataloader data=humanMulti
```

Open `http://localhost:8080`. Use **Next Batch** to browse and
**Display → View → single** to inspect one sample. Optional overrides:
`+check_batch_size=8`, `+viser_port=8081`, and `data.hand_pos_source=wrist`
(to preview wrist positions instead of index-MCP positions).

### Train and Evaluate

Use the default `train` split for held-out evaluation. Keep this experiment
separate from the [full-data synthesis prior](#object-human-prior-train-and-export).
The default `independent_from_scratch` mode starts two separate runs with
checkpoint loading and resume disabled: `human_prior_eval_diffusion`
(10,000 iterations) and `human_prior_eval_type` (300 iterations).
Use new experiment names to keep outputs from different runs separate.

```bash
python -m dexlearn.main task=train algo=humanMultiHierar data=humanMulti exp_name=human_prior_eval
```

To train only one branch, append `algo.training.independent.run=type` or
`algo.training.independent.run=diffusion`.
The examples below select score checkpoint `000100` and pose checkpoint
`007500`; use the same selected steps for sampling, viewing, and evaluation.

#### Sample Scores and Poses

Sample `0_any` scores and poses for all five contact modes on `all` Human objects:

```bash
python -m dexlearn.main task=sample algo=humanMultiHierar data=humanMulti test_data=humanMulti \
  algo.model.train_type_only=true 'test_data.grasp_type_lst=["0_any"]' \
  test_data.test_split=all exp_name=human_prior_eval_type ckpt=000100

python -m dexlearn.main task=sample algo=humanMultiHierar data=humanMulti test_data=humanMulti \
  test_data.test_split=all exp_name=human_prior_eval_diffusion ckpt=007500
```

Pose sampling defaults to 100 candidates per type and keeps 20 using
`algo.sample_selection.mode=prob_pose`: first keep the top 50 by probability,
then select diverse poses. Alternatives include `prob` (probability ranking),
`random`, and `pose_diversity` (diversity without probability preselection).

#### View Saved Samples

Run each viewer separately and open `http://localhost:8080` (requires MANO).
The Selection panel switches between all, train, and test objects.

```bash
# Contact-mode scores on point clouds.
python -m dexlearn.main task=visualize algo=humanMultiHierar data=humanMulti test_data=humanMulti \
  exp_name=human_prior_eval_type ckpt=000100 \
  task.visualize_mode=random_objects task.target_grasp_type_id=0

# Generated poses, grouped by object and grasp type.
python -m dexlearn.main task=visualize algo=humanMultiHierar data=humanMulti test_data=humanMulti \
  exp_name=human_prior_eval_diffusion ckpt=007500 task.visualize_mode=one_object \
  task.human_scores_exp_name=human_prior_eval_type task.human_scores_ckpt=000100
```

The pose viewer shows five type-branch scores for the selected object: means
from up to 20 `0_any` samples, rather than per-pose scores. The score view
labels use the same five scores as an ordered array. The score directory is
derived from the experiment name and checkpoint; set `task.human_scores_dir`
only for a custom sample directory.

The Human Viser viewer supports `random_objects` and `one_object` views.
Use the Selection panel to choose an object or grasp type, and
`task.viser_port=8081` to use a different port.

#### Evaluate Saved Samples

`type_eval` reports Human train/test score metrics by default.
`task.exclude_both_three=true` adds a report with that class removed and the
remaining four probabilities renormalized. DGN diagnostics and the scale-anchor
baseline are optional and disabled by default.

```bash
python -m dexlearn.main task=type_eval algo=humanMultiHierar data=humanMulti test_data=humanMulti \
  exp_name=human_prior_eval_type ckpt=000100 task.exclude_both_three=true

python -m dexlearn.main task=diffusion_eval algo=humanMultiHierar data=humanMulti test_data=humanMulti \
  exp_name=human_prior_eval_diffusion ckpt=007500
```

`diffusion_eval` measures record-level pose recall and index-MCP-to-object-surface
sanity metrics on the Human `test` split. It uses saved index-MCP translations
and wrist quaternions, without generating new samples or running MANO recovery.

Samples live under
`output/humanMulti_humanMultiHierar_human_prior_eval_type/tests/step_000100/`
and
`output/humanMulti_humanMultiHierar_human_prior_eval_diffusion/tests/step_007500/`,
with dataset and grasp-type subdirectories. Score reports go in `evaluation/`
under the type step directory; pose reports go in `diffusion_eval/` under the
pose step directory.

#### Optional DGN Score Diagnostics

DGN has no human grasp labels. Sample its scores first, then explicitly enable
the diagnostic section alongside Human evaluation:

```bash
python -m dexlearn.main task=sample algo=humanMultiHierar data=humanMulti test_data=DGNMulti \
  algo.model.train_type_only=true 'test_data.grasp_type_lst=["0_any"]' \
  test_data.test_split=all exp_name=human_prior_eval_type ckpt=000100

python -m dexlearn.main task=type_eval algo=humanMultiHierar data=humanMulti test_data=humanMulti \
  exp_name=human_prior_eval_type ckpt=000100 task.run_dgn_1b=true
```

#### Optional Human Scale-Anchor Baseline

This baseline uses object-scale and grasp-type statistics generated by
`task=stat` in the separate HumanGraspData preprocessing repository. Generate
them from the same formatted human data and its `object/valid_split/train.json`.
The JSON contains `all` and `train` scopes; the baseline uses `train` statistics
to predict grasp-type distributions from the nearest scale anchor on `test` objects.

After setting up HumanGraspData, run this command **from that repository**:

```bash
python src/main.py task=stat exp_name=scale_anchor task.data_name=Ours \
  task.data_path="${HUGS_DATASET_ROOT}/OurHumanGraspFormat" \
  task.object_scale_type_distribution_path="${HUGS_DATASET_ROOT}/metadata/Ours_object_scale_type_distribution.json"
```

Then return to DexLearn and enable the baseline:

```bash
python -m dexlearn.main task=type_eval algo=humanMultiHierar data=humanMulti test_data=humanMulti \
  exp_name=human_prior_eval_type ckpt=000100 task.run_human_scale_anchor_baseline=true
```

It writes `evaluation_human_scale_anchor_baseline_{predictions,metrics,summary}.csv`
in the type branch's `evaluation/` directory. For a JSON stored elsewhere, set
`task.human_scale_anchor_distribution_json=/path/to/statistics.json`.
If the JSON is absent, the baseline is skipped.

### Object Human Prior Train and Export

Train on all Human objects for downstream synthesis, rather than held-out evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py \
  task=train algo=humanMultiHierar data=humanMulti \
  algo.training.mode=independent_from_scratch \
  data.sampling.train_split=all \
  exp_name=human_prior_full
```

Export object-scene human prior scores and hand-position seeds for downstream
BODex synthesis. The default task writes one 5-type budget score vector per
scene, generates `algo.test_grasp_num` pose candidates per scene and grasp
type, and keeps `algo.test_topk` samples according to `algo.sample_selection`.
Set `task.robot_name` and `task.robot_size` to condition the export on a target
robot hand. The test point cloud is scaled by `1 / task.robot_size` for human
prior inference, while saved pose translations are mapped back to physical scene
units. When `data.hand_pos_source=index_mcp`, the export stores
`index_mcp_pos` and `wrist_quat` directly without running MANO to infer
`wrist_pos`.

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py \
  task=obj_human_prior_export \
  data=humanMulti \
  algo=humanMultiHierar \
  test_data=DGNMulti \
  test_data.test_split=all \
  algo.batch_size=1024 \
  task.skip_existing=false \
  task.robot_name=shadow_hand \
  task.robot_size=1.0 \
  task.score_ckpt=000100 \
  task.pose_ckpt=007500 \
  wandb.mode=disabled \
  exp_name=human_prior_full \
  task.score_exp_name=human_prior_full_type \
  task.pose_exp_name=human_prior_full_diffusion
```

Outputs are written to
`output/humanMulti_humanMultiHierar_<EXP_NAME>/obj_human_prior/step_<CKPT>/`
for single-checkpoint export, or
`output/humanMulti_humanMultiHierar_<EXP_NAME>/obj_human_prior/step_<POSE_CKPT>_<SCORE_CKPT>/`
for independent score/pose export, unless `task.output_dir` is set. When
`task.score_ckpt` and `task.pose_ckpt` are both set, the top-level `ckpt`
override is not used. Per-scene files are stored under a subdirectory named
after `test_data.object_path`'s final component and `task.robot_name`,
preserving the original scene id hierarchy, for example
`.../step_<POSE_CKPT>_<SCORE_CKPT>/DGN_2k/shadow_hand/<object>/<env>/<scene>.npy`.

For Reverse export, use the two Reverse checkpoints with
`algo=humanMultiReverse`:

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py \
  task=obj_human_prior_export \
  data=humanMulti \
  algo=humanMultiReverse \
  test_data=DGNMulti \
  test_data.test_split=all \
  algo.batch_size=1024 \
  task.skip_existing=false \
  task.robot_name=leap \
  task.robot_size=1.8 \
  task.score_ckpt=000300 \
  task.pose_ckpt=010000 \
  wandb.mode=disabled \
  exp_name=<EXP_NAME> \
  task.score_exp_name=<EXP_NAME>_type_posterior \
  task.pose_exp_name=<EXP_NAME>_pose_marginal
```

Its default export root is
`output/humanMulti_humanMultiReverse_<EXP_NAME>/obj_human_prior/step_<POSE_CKPT>_<SCORE_CKPT>/`.

For Joint export, use the single coupled checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py \
  task=obj_human_prior_export \
  data=humanMulti \
  data.hand_pos_source=index_mcp \
  algo=humanMultiJoint \
  test_data=DGNMulti \
  test_data.test_split=all \
  task.robot_name=leap \
  task.robot_size=1.8 \
  task.skip_existing=false \
  ckpt=010000 \
  exp_name=<EXP_NAME> \
  wandb.mode=disabled
```

Joint export draws one shared 500-sample raw pool and computes
`budget_scores = bincount(type_ids - 1, minlength=5) / 500` before selection.
It groups only by the sampled hard mode, caps each group at 100 candidates,
selects 20 with `prob_pose`, and uses deterministic replacement only when a
non-empty group has fewer than 20 raw samples. A zero-support mode receives a
finite placeholder plus a zero score and is never filled by conditional
generation. BimanBODex is unchanged; its formal synthesis config must keep
`human_prior.min_type_budget=0`.

The compact consumer record remains under `<asset>/<robot>/<scene>.npy` with
the existing `budget_scores`, `index_mcp_pos`, `wrist_quat`, and
`active_hand_mask` fields. The full raw pool is stored separately under
`raw_joint/<scene>.npz`; the compact record saves its relative path, SHA-256,
selected raw indices, replacement mask, zero-support mask, and checkpoint hash.

For a small initial export, append `test_data.test_scene_num=1` to export
one scene. Remove the override to export the full configured set.

To export an exact bounded scene set, set
`test_data.test_scene_list_path=<SCENE_LIST_JSON>` and leave
`test_data.test_scene_num=0`. The JSON may be a plain list of scene ids, or an
object with a `scene_paths` list and optional matching `scene_count`. Relative
entries resolve under `<test_data.object_path>/scene_cfg`; absolute entries
must resolve inside the same scene root. This avoids enumerating the full asset
split when a downstream synthesis experiment references only a known subset.

Reverse export generates one shared 500-pose marginal pool before evaluating
`q(c|T,o)`. It computes the five `budget_scores` before filtering, performs
fixed-seed weighted sampling without replacement to obtain 100 candidates per
mode, and applies the existing full-bimanual `prob_pose` diversity selection to
keep 20. Mode-specific active-hand masks, rescaling, and decentering happen only
after selection. Per-scene files retain the raw pool, posterior probabilities,
raw sampled modes, resampling/selection indices, ESS, checkpoint hashes, and an
explicit `factorization=reverse_T_to_C` tag. Long training, batch export, GPU
synthesis, and benchmark runs should still be launched only after their output
roots and resources are approved.

Visualize an exported object human prior (requires MANO):

```bash
python dexlearn/main.py \
  task=visualize_human_prior \
  data=humanMulti \
  algo=humanMultiHierar \
  task.prior_dir=output/humanMulti_humanMultiHierar_human_prior_full/obj_human_prior/step_007500_000100/DGN_2k/shadow_hand \
  task.visualize_mode=one_scene
```

For this task, set `task.prior_dir` to the full robot-specific export directory, such as
`output/humanMulti_humanMultiHierar_human_prior_full/obj_human_prior/step_007500_000100/DGN_2k/shadow_hand`.
`visualize_human_prior` then reads per-scene files directly from
`<prior_dir>/<object>/...`. Do not pass separate `task.step` or
`task.robot_name` overrides for visualization; the step, asset set, and robot
namespace are already encoded in the full path. The export also writes
`manifest.json`, `scene_index.json`, and `scene_budget_scores.jsonl`.
Visualization does not depend on these summary files, but they are kept for
reproducibility, audit/debug metadata, and downstream score evaluation.
Open `http://localhost:8080`, select a scene and **Grasp Type**, and click
**Apply Selection**. Hand meshes use fixed, flat fingers to illustrate the
exported positions and orientations.
In the web Selection panel, Grasp Type `0_any` shows score-only records for
`random_objects`; selecting a concrete type renders one random wrist pose of
that type per object. In `one_scene`, `0_any` shows all real types and concrete
types show only that row; Next Batch advances the pose sample window within the
same scene, and Next Scene selects another random scene while preserving the
current grasp-type selection.


### Scene Budget

Build human-only scene-budget labels and train the independent geometry budget
head. This task does not train the CE or diffusion model. The default label
source is `task.label_source=hierarchy_count`. The task writes
`scene_budget_label_hierarchy.csv`, a compact per-grasp table grouped as
canonical object, scene id, pose class, grasp type, and grasp record. It then
aggregates those rows in memory into one direct-count row per pose-class scene:
`(component_idx, split, canonical_object_id, pose_class_id)`. The raw label is
`grasp_record_count`; the training target is
`log_count_multiplier = log(clip(grasp_record_count / mean_train_count))`.

By default `task.splits=[train,test]`, so the label build reads the explicit
`train.json` and `test.json` files under the configured object split directory.
The budget head trains on `split=train` rows and validates on `split=test` rows;
`scene_budget` does not create its own random validation split.

The budget head is intentionally small because the direct-count dataset is tiny:
default hidden dimensions are `[16, 16]`, dropout is `0.1`, weight decay is
`0.001`, and validation-MSE early stopping is enabled. These defaults are meant
for a conservative geometry-only budget baseline, not a high-capacity predictor.

The budget head input uses three yaw-invariant bounding-box dimensions:
`bbox_xy_major`, `bbox_xy_minor`, and `bbox_z`. The canonical point cloud is
scaled and transformed by the stored object pose before measuring the bbox. The
XY box uses the minimum-area rectangle over the tabletop plane rather than fixed
world X/Y axes.

Set `task.train.input_type=pointcloud` to train the budget head from object
point clouds instead of bbox features. This path uses the same `WrappedMinkUNet`
backbone family as `task=train`, can initialize from a main training checkpoint
with `task.train.pointcloud.encoder_checkpoint=<CKPT>`, and supports Z-yaw
augmentation through `task.train.pointcloud.z_yaw_aug=true`. The default
`task.train.input_type=bbox` remains the lightweight baseline.

```bash
python dexlearn/main.py task=scene_budget data=humanMulti algo=humanMultiHierar exp_name=<EXP_NAME>
```

Default outputs are written to
`output/humanMulti_humanMultiHierar_<EXP_NAME>/scene_budget/`:

- `scene_budget_label_hierarchy.csv`: compact per-grasp canonical-object / scene / pose-class / type table
- `scene_budget_summary.json`: feature normalization, direct-count statistics, and checks
- `geometry_budget_head.pth`: trained independent budget head checkpoint
- `budget_head_predictions.csv`: train/test target and predicted budget multipliers
- `budget_head_train_summary.json`: train/validation metrics
- `budget_head_train_multiplier_scatter.png`: train-set target-vs-predicted multiplier plot
- `budget_head_test_multiplier_scatter.png`: test-set target-vs-predicted multiplier plot
- `scene_budget_run_summary.json`: resolved task config and output paths

Common overrides:

```bash
# Only build scene-budget labels, without training the budget head.
python dexlearn/main.py \
  task=scene_budget \
  data=humanMulti \
  algo=humanMultiHierar \
  exp_name=<EXP_NAME> \
  task.mode=build_labels

# Train with the point-cloud encoder input instead of bbox features.
python dexlearn/main.py \
  task=scene_budget \
  data=humanMulti \
  algo=humanMultiHierar \
  exp_name=<EXP_NAME> \
  task.train.input_type=pointcloud \
  task.train.pointcloud.encoder_checkpoint=<PATH_TO_TRAIN_CKPT>

# Run budget-head inference on the same test_data interface used by task=sample.
python dexlearn/main.py \
  task=scene_budget \
  data=humanMulti \
  algo=humanMultiHierar \
  test_data=DGNMulti \
  exp_name=<EXP_NAME> \
  task.mode=predict \
  task.inference.checkpoint=<PATH_TO>/geometry_budget_head.pth

# Use the legacy nearest-scene diverse-grasp-class label source for ablation.
python dexlearn/main.py \
  task=scene_budget \
  data=humanMulti \
  algo=humanMultiHierar \
  exp_name=<EXP_NAME> \
  task.label_source=legacy_nearest_n \
  task.legacy_nearest_n.nearest_scene_num=16 \
  task.legacy_nearest_n.orientation_threshold_deg=30.0 \
  task.legacy_nearest_n.direction_threshold_deg=30.0 \
  task.legacy_nearest_n.posed_object_translation_threshold_m=0.1 \
  task.legacy_nearest_n.posed_object_rotation_threshold_deg=45.0 \
  task.label_structure.pose_class_rotation_threshold_deg=45.0 \
  task.label_structure.pose_class_bbox_proportion_threshold=0.2 \
  task.legacy_nearest_n.clip_min=0.5 \
  task.legacy_nearest_n.clip_max=3.0
```
