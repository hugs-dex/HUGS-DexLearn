# 完整工作流参考

下列训练、采样、导出、评估与可视化接口随源码保留，运行前按 [README 安装说明](../README.md#installation) 准备所需输入。
这些命令不是本次真实训练验证证据。预处理会修改输入 grasp，应在明确的数据工作副本上使用。

## Algorithm 与 data 配对

| Algorithm | Data |
| --- | --- |
| nflow、diffusion | bodex_shadow 或相应旧机器人数据配置 |
| humanNflow、humanDiffusion | human |
| humanBiDiffusion | humanbi |
| humanMultiDiffusion、humanMultiHierar、humanMultiJoint、humanMultiReverse | humanMulti |
| robotMultiHierar | shadowMulti、leapspMulti、leapMulti |

旧单手/双手数据配置不能随意与新 humanMulti 模型混搭。

## Multi-GPU Sampling

```bash
python dexlearn/scripts/launch_multi_sample.py --exp-names example --gpus 0 1 --dry-run
```

训练使用下文的 `python -m dexlearn.main task=train` 命令。
采样脚本使用 --common-extra-overrides、--score-extra-overrides、--pose-extra-overrides 表达实验差异，
不再按内部 debug 名自动选择模型参数。GPU 数字指 CUDA_VISIBLE_DEVICES 的物理设备选择；子进程内 device=cuda:0。
dry-run 不创建日志、不运行 GPU；真实采样汇总 worker 失败并返回非零。

## Arguments

- `exp_name`: experiment name used in output paths
- `DATA_NAME`: training dataset config name
- `TEST_DATA_NAME`: test dataset config name
- `ckpt`: checkpoint step or checkpoint path to load

Availabel configs for robot workflow:

- `DATA_NAME`: `shadowMulti`, `leapMulti`
- `TEST_DATA_NAME`: `shadowMulti`, `leapMulti`

## Outputs

- Training checkpoints are saved under `output/<data>_<algo>_<exp_name>/ckpts/`
- Sampled results are saved under `output/<data>_<algo>_<exp_name>/tests/step_<ckpt>/`
- `visualize` reads sampled grasps from the corresponding `tests` directory

## Robot Workflow

### Check Dataloader

Inspect robot dataloader samples before training or debugging. The script buffers samples so visualization follows a fixed grasp-type order such as `1 2 3 4 5`, then `1 2 3 4 5` again when available.

```bash
CUDA_VISIBLE_DEVICES=0 python tests/check_robot_dataloader.py data=<DATA_NAME> exp_name=<EXP_NAME>
```

### Train

Train a robot grasp model.

`robotMultiHierar` uses `RobotHierarchicalModel` with binary contact-mode
availability prediction and `single_stage` training. Existing robot checkpoint
parameter names and shapes are preserved; load them with the current config.

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=train algo=robotMultiHierar data=<DATA_NAME> num_workers=24 prefetch_factor=2 exp_name=<EXP_NAME>
```

### Sample

Generate robot grasps from a trained checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=sample algo=robotMultiHierar data=<DATA_NAME> test_data=<TEST_DATA_NAME> exp_name=<EXP_NAME>

# Override the availability score threshold used when grasp_type_id=0 samples
# all model-predicted available real grasp types. The robotMultiHierar default is 0.5.
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=sample algo=robotMultiHierar data=<DATA_NAME> test_data=<TEST_DATA_NAME> exp_name=<EXP_NAME> algo.model.type_availability.score_threshold=0.35
```

### Visualize

Visualize sampled robot grasps. The current visualization sampler is controlled by `task.visualize_mode`; the old group-balanced grasp-type cycling behavior is deprecated.

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=visualize algo=robotMultiHierar data=<DATA_NAME> test_data=<TEST_DATA_NAME> exp_name=<EXP_NAME> wandb.mode=disabled

# e.g., python dexlearn/main.py task=visualize algo=robotMultiHierar data=leapMulti test_data=leapMulti exp_name=dataset_full_1

# Web visualizer. The browser UI can switch views and apply object or grasp-type selections.
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=visualize task.visualizer=viser algo=robotMultiHierar data=<DATA_NAME> test_data=<TEST_DATA_NAME> exp_name=<EXP_NAME>

# New sample selection modes.
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=visualize task.visualize_mode=random_object task.max_grasps=20 algo=robotMultiHierar data=<DATA_NAME> test_data=<TEST_DATA_NAME> exp_name=<EXP_NAME>
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=visualize task.visualize_mode=one_object task.object_id=<OBJECT_ID> task.max_grasps=20 algo=robotMultiHierar data=<DATA_NAME> test_data=<TEST_DATA_NAME> exp_name=<EXP_NAME>
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=visualize task.visualize_mode=grasp_type task.target_grasp_type_id=<1-5> task.max_grasps=20 algo=robotMultiHierar data=<DATA_NAME> test_data=<TEST_DATA_NAME> exp_name=<EXP_NAME>
```

## Human Workflow

See [Human Prior Architectures](human_prior_architectures.md) for the model
options and architecture-specific training and export commands.

### Preprocess

Compute and save `index_mcp_pos` into the source human grasp files before training with index-MCP positions.

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=human_preprocess data=humanMulti exp_name=<EXP_NAME>

# e.g., CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=human_preprocess data=humanMulti exp_name=example
```

### Check Dataloader

Inspect human dataloader samples after preprocessing and before training. This visualization follows the configured `hand_pos_source` in `dexlearn/config/data/humanMulti.yaml`.

```bash
CUDA_VISIBLE_DEVICES=0 python -m dexlearn.scripts.check_human_dataloader data=humanMulti data.hand_pos_source=<wrist/index_mcp>

# e.g.: CUDA_VISIBLE_DEVICES=0 python -m dexlearn.scripts.check_human_dataloader data=humanMulti data.hand_pos_source=index_mcp
```

### Train

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=train algo=humanMultiHierar data=humanMulti exp_name=<EXP_NAME>

# e.g.: CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=train algo=humanMultiHierar data=humanMulti exp_name=<EXP_NAME>
```

Human Prior training mode can be switched with `algo.training.mode`:

```bash
# 1. two independent from-scratch runs: <EXP_NAME>_diffusion and <EXP_NAME>_type
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py \
  task=train algo=humanMultiHierar data=humanMulti exp_name=<EXP_NAME> \
  algo.training.mode=independent_from_scratch

# 1b. only train the faster type-predictor branch
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py \
  task=train algo=humanMultiHierar data=humanMulti exp_name=<EXP_NAME> \
  algo.training.mode=independent_from_scratch \
  algo.training.independent.run=type

# 1c. only train the diffusion branch
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py \
  task=train algo=humanMultiHierar data=humanMulti exp_name=<EXP_NAME> \
  algo.training.mode=independent_from_scratch \
  algo.training.independent.run=diffusion

# 2. one shared model, joint single-stage training
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py \
  task=train algo=humanMultiHierar data=humanMulti exp_name=<EXP_NAME> \
  algo.training.mode=joint_single_stage
```

### Reverse T-to-C Human Prior

`humanMultiReverse` implements the independent Reverse factorization
`p(T|o) p(c|T,o)`. It launches two from-scratch runs by default:
`<EXP_NAME>_pose_marginal` trains the object-only marginal pose diffusion for
10,000 iterations, then `<EXP_NAME>_type_posterior` trains the independent
hard-label pose-conditioned posterior for 300 iterations. Both branches use
record-uniform sampling without type balancing or pose-group soft labels.

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py \
  task=train algo=humanMultiReverse data=humanMulti \
  data.sampling.train_split=all \
  exp_name=<EXP_NAME>

# Optional: launch only one independent branch.
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py \
  task=train algo=humanMultiReverse data=humanMulti \
  data.sampling.train_split=all \
  algo.training.run=pose_marginal \
  exp_name=<EXP_NAME>

CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py \
  task=train algo=humanMultiReverse data=humanMulti \
  data.sampling.train_split=all \
  algo.training.run=type_posterior \
  exp_name=<EXP_NAME>
```

The posterior owns a separate checkpointed 24D pose normalization. It trains
on clean GT centered canonical `T24`, while inference consumes generated
centered canonical `T24` from the marginal diffusion; this GT-to-generated pose
shift is the intentional initial train-inference gap. The marginal generator
does not accept a contact-mode id or mode-specific sampling path.

### Joint Contact-Mode / Wrist-Pose Human Prior

`humanMultiJoint` implements the coupled factorization `p(c,T|o)` with one
checkpoint. Contact mode is a five-class categorical diffusion state and wrist
pose is the existing Gaussian T24 state. Every reverse step evaluates both
heads from the same old `(c_t, T_t)` state before synchronously updating them;
`0_any` is only a test-loader placeholder and is never generated.

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py \
  task=train algo=humanMultiJoint data=humanMulti \
  data.sampling.train_split=all \
  exp_name=<EXP_NAME>
```

The initial comparison budget is 10,000 iterations, record-uniform data,
`loss_pose_v + loss_categorical`, and the same `WrappedMinkUNet` object encoder
used by the Proposed baseline.

### Sample

For the default `algo.training.mode=independent_from_scratch`, diffusion poses
and grasp-type scores are saved by two different runs. If the base experiment
name is `<EXP_NAME>`, use `<EXP_NAME>_diffusion` for wrist/index-MCP pose
sampling and `<EXP_NAME>_type` for Human Prior grasp-type score sampling.

```bash
# 1. Sample typed wrist/index-MCP poses from the diffusion checkpoint.
# humanMultiHierar defaults to test_grasp_num=100, test_topk=20, and
# algo.sample_selection.mode=pose_diversity.
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py \
  task=sample data=humanMulti algo=humanMultiHierar test_data=humanMulti \
  algo.batch_size=1024 \
  'test_data.grasp_type_lst=["1_right_two","2_right_three","3_right_full","4_both_three","5_both_full"]' \
  ckpt=007500 \
  exp_name=<EXP_NAME>_diffusion

# Optional pose candidate selection overrides:
#   algo.sample_selection.mode=prob            # legacy log-prob top-20
#   algo.sample_selection.mode=random          # random 20 from 100 candidates
#   algo.sample_selection.mode=pose_diversity  # default diverse 20 from 100

# 2. Sample grasp-type scores from the type checkpoint.
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py \
  task=sample data=humanMulti algo=humanMultiHierar test_data=humanMulti \
  algo.model.train_type_only=true \
  algo.batch_size=1024 \
  'test_data.grasp_type_lst=["0_any"]' \
  ckpt=000100 \
  exp_name=<EXP_NAME>_type
```


### Object Human Prior Train and Export

Train:
```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py \
  task=train algo=humanMultiHierar data=humanMulti \
  algo.training.mode=independent_from_scratch \
  data.sampling.train_split=all \
  exp_name=human_prior_<x>
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
  task.robot_name=leap \
  task.robot_size=1.8 \
  task.score_ckpt=000100 \
  task.pose_ckpt=007500 \
  wandb.mode=disabled \
  exp_name=example_prior \
  task.score_exp_name=example_prior_type \
  task.pose_exp_name=example_prior_diffusion
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
`.../step_<POSE_CKPT>_<SCORE_CKPT>/DGN_5k/leap_hand/<object>/<env>/<scene>.npy`.

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

Visualize an exported object human prior:
```bash
python dexlearn/main.py \
  task=visualize_human_prior \
  data=humanMulti \
  algo=humanMultiHierar \
  task.prior_dir=output/humanMulti_humanMultiHierar_example_prior/obj_human_prior/step_007500_000100/DGN_2k/shadow
```

For this task, `dexlearn/config/task/visualize_human_prior.yaml` should set
`task.prior_dir` to the full robot-specific export directory, such as
`output/humanMulti_humanMultiHierar_example_prior/obj_human_prior/step_007500_000100/DGN_2k/shadow`.
`visualize_human_prior` then reads per-scene files directly from
`<prior_dir>/<object>/...`. Do not pass separate `task.step` or
`task.robot_name` overrides for visualization; the step, asset set, and robot
namespace are already encoded in the full path. The export also writes
`manifest.json`, `scene_index.json`, and `scene_budget_scores.jsonl`.
Visualization does not depend on these summary files, but they are kept for
reproducibility, audit/debug metadata, and downstream score evaluation.
In the web Selection panel, Grasp Type `0_any` shows score-only records for
`random_objects`; selecting a concrete type renders one random wrist pose of
that type per object. In `one_scene`, `0_any` shows all real types and concrete
types show only that row; Next Batch advances the pose sample window within the
same scene, and Next Scene selects another random scene while preserving the
current grasp-type selection.


### Visualize

```bash
python dexlearn/main.py task=visualize data=humanMulti algo=humanMultiHierar test_data=<TEST_DATA> exp_name=<EXP_NAME>

# e.g, python dexlearn/main.py task=visualize data=humanMulti algo=humanMultiHierar test_data=humanMulti ckpt=010000 exp_name=<EXP_NAME>

# Web visualizer with multi-scene layout and runtime object or grasp-type selection.
python dexlearn/main.py task=visualize task.visualizer=viser task.viser_port=8080 task.viser_display_mode=single task.viser_scene_id=0 data=humanMulti algo=humanMultiHierar test_data=<TEST_DATA> exp_name=<EXP_NAME>

# New sample selection modes.
python dexlearn/main.py task=visualize task.visualize_mode=random_object task.max_grasps=20 data=humanMulti algo=humanMultiHierar test_data=<TEST_DATA> exp_name=<EXP_NAME>
python dexlearn/main.py task=visualize task.visualize_mode=one_object task.object_id=<OBJECT_ID> task.max_grasps=20 data=humanMulti algo=humanMultiHierar test_data=<TEST_DATA> exp_name=<EXP_NAME>
python dexlearn/main.py task=visualize task.visualizer=viser task.visualize_mode=one_object_multi_seq task.object_id=obj_0_seq_0 task.max_grasps=20 data=humanMulti algo=humanMultiHierar test_data=humanMulti exp_name=<EXP_NAME>
python dexlearn/main.py task=visualize task.visualize_mode=grasp_type task.target_grasp_type_id=<1-5> task.max_grasps=20 data=humanMulti algo=humanMultiHierar test_data=<TEST_DATA> exp_name=<EXP_NAME>
```

### Evaluate

Evaluate an already sampled human model run. Run `task=sample` first; this task
does not generate samples.

```bash
CUDA_VISIBLE_DEVICES=0 python dexlearn/main.py task=type_eval algo=humanMultiHierar data=humanMulti test_data=humanMulti wandb.mode=disabled exp_name=<EXP_NAME> ckpt=<CKPT>

# Example:
python dexlearn/main.py \
    task=type_eval algo=humanMultiHierar data=humanMulti test_data=humanMulti \
    exp_name=debug26 ckpt=010000 wandb.mode=disabled
```

### Diffusion Eval

Evaluate saved human diffusion index-MCP pose samples on `humanMulti` using
record-level recall plus index-MCP-to-object-surface sanity metrics. Run
`task=sample` first; this task does not generate samples or run MANO recovery.
Translation metrics use saved/generated index-MCP positions; rotation metrics
use saved/generated wrist quaternions.

```bash
python dexlearn/main.py \
    task=diffusion_eval algo=humanMultiHierar data=humanMulti test_data=humanMulti \
    exp_name=<EXP_NAME> ckpt=<CKPT> wandb.mode=disabled
```

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
