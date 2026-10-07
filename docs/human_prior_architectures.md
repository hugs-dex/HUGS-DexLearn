# Human Prior Architectures

The [README](../README.md#human-prior) uses `humanMultiHierar` with
`independent_from_scratch` training as the main Human Prior workflow. This
guide describes the architecture options retained for comparisons and how to
select, train, and use them.

## Architecture options

Let `o` denote the object point cloud, `c` one of the five real contact modes,
and `T` the two-hand pose representation (T24). The default position source is
index MCP (`data.hand_pos_source=index_mcp`); rotations describe wrist
orientation. `0_any` is a sampling placeholder, not a sixth contact mode.

| Option | Algorithm config | Modeling and checkpoints |
| --- | --- | --- |
| Main hierarchical prior, Independent training | `humanMultiHierar` | `p(c|o) p(T|c,o)`; separately trained score and conditional-pose branches, saved as `<exp>_type` and `<exp>_diffusion`. |
| Strict Independent baseline | `humanMultiIndependent` | `p(c|o) p(T|o)`; object-only mode and pose marginals, saved as `<exp>_mode_marginal` and `<exp>_pose_marginal`. |
| Joint baseline | `humanMultiJoint` | `p(c,T|o)`; coupled categorical contact-mode and Gaussian pose diffusion in one checkpoint under `<exp>`. |
| Reverse baseline | `humanMultiReverse` | `p(T|o) p(c|T,o)`; pose marginal and pose-conditioned mode posterior, saved as `<exp>_pose_marginal` and `<exp>_type_posterior`. |

In the main workflow, **Independent refers to branch training**: pose generation
still conditions on the requested contact mode. In `humanMultiIndependent`,
the mode and pose are also statistically independent given the object; its
pose generator receives no contact-mode input. These configurations and their
checkpoints are not interchangeable.

The main model is `HierarchicalTypeObjectiveModel`. Strict Independent uses
`ObjectModeMarginalModel` and `MarginalPoseDiffusionModel`; Reverse replaces
the object-only mode predictor with `PoseConditionedTypeModel`. Joint uses
`JointHybridDiffusionModel` with `JointCategoricalPoseDiffusion`. The current
configs use `WrappedMinkUNet` object encoders.

## Training

Run commands from the repository root after following the README installation
and asset setup. Each example uses a distinct experiment name. Commands below
use the default human `train` split; append `data.sampling.train_split=all`
to the `data=humanMulti` commands for full-data synthesis priors.

### Main hierarchical prior: Independent training

```bash
python -m dexlearn.main task=train algo=humanMultiHierar data=humanMulti \
  algo.training.mode=independent_from_scratch exp_name=prior_main
```

The type branch runs first for 300 iterations, followed by the conditional-pose
branch for 10,000 iterations. Each branch starts from scratch with its own
encoder and checkpoint. Use `algo.training.independent.run=type` or
`algo.training.independent.run=diffusion` to train just one branch; the default
is `both`. Branch schedules live under `algo.training.independent.type.*` and
`algo.training.independent.diffusion.*`.

### Strict Independent baseline

```bash
python -m dexlearn.main task=train algo=humanMultiIndependent data=humanMulti \
  exp_name=prior_independent
```

The mode marginal trains for 300 iterations, then the pose marginal for 10,000.
Use `algo.training.run=mode_marginal` or `algo.training.run=pose_marginal` to
select one branch. Branch schedules are under `algo.training.<branch>.*`.
The training route enforces record-uniform sampling without type balancing;
the mode branch uses pose-group soft labels, while the pose branch does not.

### Joint baseline

```bash
python -m dexlearn.main task=train algo=humanMultiJoint data=humanMulti \
  exp_name=prior_joint
```

One model trains for 10,000 iterations using pose and categorical diffusion
losses. Training enforces record-uniform sampling and hard contact-mode labels
without type balancing. The two diffusion states update together at each step.

### Reverse baseline

```bash
python -m dexlearn.main task=train algo=humanMultiReverse data=humanMulti \
  exp_name=prior_reverse
```

The pose marginal trains for 10,000 iterations, then the type posterior for
300. Use `algo.training.run=pose_marginal` or
`algo.training.run=type_posterior` to select a branch, and
`algo.training.<branch>.*` to change its schedule. Both branches use
record-uniform sampling without type balancing or pose-group soft labels.
The posterior learns from clean ground-truth poses but receives generated
poses at export time.

## Other training modes for the main architecture

`humanMultiHierar` also supports these `algo.training.mode` values:

| Training mode | Behavior |
| --- | --- |
| `independent_from_scratch` | Default: separate score and conditional-pose training runs. |
| `joint_single_stage` | Train the score predictor and conditional-pose diffusion together with a shared encoder in one run. |

For example:

```bash
python -m dexlearn.main task=train algo=humanMultiHierar data=humanMulti \
  algo.training.mode=joint_single_stage exp_name=prior_shared
```

`joint_single_stage` keeps the hierarchical model and conditional pose
generator; it does not select the coupled diffusion architecture in
`humanMultiJoint`.

Robot training uses `single_stage`. Algorithms without an explicit
`algo.training.mode` also default to `single_stage`, which trains one model
with its configured losses and checkpoint settings.

## Export for HUGS-BODex

Use the same algorithm and data config as training. The examples use final
default checkpoints; replace them with the saved steps you want to export.

```bash
# Main hierarchical prior: independent score and conditional-pose branches.
python -m dexlearn.main task=obj_human_prior_export \
  algo=humanMultiHierar data=humanMulti test_data=DGNMulti exp_name=prior_main \
  task.score_exp_name=prior_main_type task.score_ckpt=000300 \
  task.pose_exp_name=prior_main_diffusion task.pose_ckpt=010000

# Strict Independent: object-only marginals.
python -m dexlearn.main task=obj_human_prior_export \
  algo=humanMultiIndependent data=humanMulti test_data=DGNMulti exp_name=prior_independent \
  task.score_exp_name=prior_independent_mode_marginal task.score_ckpt=000300 \
  task.pose_exp_name=prior_independent_pose_marginal task.pose_ckpt=010000

# Joint: a single coupled checkpoint.
python -m dexlearn.main task=obj_human_prior_export \
  algo=humanMultiJoint data=humanMulti test_data=DGNMulti \
  exp_name=prior_joint ckpt=010000

# Reverse: pose marginal and pose-conditioned type posterior.
python -m dexlearn.main task=obj_human_prior_export \
  algo=humanMultiReverse data=humanMulti test_data=DGNMulti exp_name=prior_reverse \
  task.score_exp_name=prior_reverse_type_posterior task.score_ckpt=000300 \
  task.pose_exp_name=prior_reverse_pose_marginal task.pose_ckpt=010000
```

Exports default to `task.robot_name=shadow_hand` and `task.robot_size=1.0`.
Override both for the target robot. The robot size rescales the input point
cloud for inference and maps output translations back to physical scene units.

The main prior generates poses conditioned on each mode. Strict Independent
samples separate mode and pose pools; its export adapter selects poses without
using mode scores. Reverse weights a shared pose pool with the posterior to
obtain candidates for each mode. Joint groups a coupled sample pool by its
generated contact modes and estimates budget scores from mode frequencies.
Joint modes with no sampled support receive zero budget scores; downstream
synthesis should keep `human_prior.min_type_budget=0`.

Generic `task=sample` defaults to the pose marginal for Strict Independent and
Reverse. Use `task=obj_human_prior_export` to combine their two checkpoints.
For the main prior's separate score/pose sampling and evaluation commands,
see the [README](../README.md#train-and-evaluate). Export layout and detailed
selection behavior are documented in [workflows](workflows.md#object-human-prior-train-and-export)
and [data contracts](contracts.md).

Append `--cfg job --resolve` to a command to inspect its Hydra configuration
without starting training or export. Configuration validation alone does not
verify a GPU training or synthesis run.
