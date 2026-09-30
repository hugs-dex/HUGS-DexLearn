# Data and model contracts

## Dataset inputs

`HUGS_DATASET_ROOT` points to a versioned bundle containing `object/DGN_2k/`
and `OurHumanGraspFormat/` (`grasp/`, `object/`, and `metadata.csv`). The outer
`archives/`, `extracted/`, and `bundles/` directories are storage organization,
not loader roots. Record the data revision and any missing components.

Human grasps contain left/right `trans`, `rot`, MANO parameters, and optional
`index_mcp_pos`. Translations are metres and rotation vectors are radians.
`index_mcp_pos` uses MANO joint 5: convert MANO output from millimetres to metres,
then add wrist translation. `hand_pos_source=index_mcp` requires this field.
Scene/pose indices and splits retain their producer definitions. Human and DGN
scene frames are not implicitly rotated into each other.

Robot training uses the configured `human_DGN2k_full/{shadow,leap_sp}` paths,
which can be overridden. Joint order comes from the URDF and training metadata;
the dual-hand joint counts are 44 for Shadow and 32 for Leap/Leap-SP. Array
length alone does not define joint order or hand identity. Output `grasp_pose`
retains the model trajectory layout; each hand uses xyz, wxyz, then joints.

## DexLearn to HUGS-BODex

The type order is fixed; hand axes use `[right, left]`:

| ID | Name | Active hands |
| --- | --- | --- |
| 1 | 1_right_two | right |
| 2 | 2_right_three | right |
| 3 | 3_right_full | right |
| 4 | 4_both_three | both |
| 5 | 5_both_full | both |

`0_any` requests scoring/sampling; it is not a sixth class.

| Field | Contract |
| --- | --- |
| budget_scores | `(5,)`; interpretation recorded by score_semantics and factorization |
| index_mcp_pos | `(5, K, 2, 3)`; metres in the physical scene frame |
| wrist_quat | `(5, K, 2, 4)`; WXYZ, normalized for active hands |
| active_hand_mask | `(5, K, 2)` bool; right then left |
| grasp_type_ids/names | Fixed five-type order |
| scene_id / object_id | Stable identifiers; scene_id is relative, without `.npy` |
| robot_name / robot_size | Output namespace and positive size relative to the human hand |
| pc_runtime_scale | `1 / robot_size`; exported positions restore physical scale and point-cloud centre |

A wrist-configured export may contain `wrist_pos`, but the current BODex
consumer requires `index_mcp_pos`. These fields are not interchangeable.
Inactive left-hand poses are placeholders; consumers must respect the mask.

Dataset paths under the configured root are saved relative to `HUGS_DATASET_ROOT`.
NPY records, index rows, score rows, and manifests identify this base with
`path_root: "HUGS_DATASET_ROOT"`. Set the variable when reading relocated exports.
Explicit absolute inputs remain valid. Mesh paths inside scene files remain
scene-relative. Paths outside the dataset root, including checkpoints, retain
their runtime values; review run metadata before distribution.

Each scene has an NPY record. The export also includes `manifest.json`,
`scene_index.json`, and `scene_budget_scores.jsonl`.

## Reproducibility

The manifest records checkpoint hashes, model and data configurations, candidate
counts and selection policy, seed, scene-whitelist hash, code commit/dirty state,
and environment. `coordinate_contract` adds explanatory metadata without changing
required arrays. A dirty worktree is not a validated release commit.

Synthetic exporter/reader tests establish format compatibility. They do not
establish model quality or successful initialization on real human data.
