import sys
import os
import time
import textwrap
import copy
from datetime import datetime

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import trange
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexlearn.utils.logger import Logger
from dexlearn.utils.config import resolve_type_supervision_config
from dexlearn.utils.util import set_seed
from dexlearn.dataset import create_train_dataloader
from dexlearn.network.models import *


def _maybe_cuda_sync(device: str, enabled: bool):
    if enabled and isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _as_list(value) -> list:
    """Convert optional Hydra list-like values to a plain Python list.

    Args:
        value: A scalar, list-like object, or ``None`` from Hydra config.

    Returns:
        Plain list. ``None`` returns an empty list.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _load_model_state(model: torch.nn.Module, ckpt_state: dict, strict: bool, ignore_prefixes: list[str]) -> None:
    """Load checkpoint weights with optional non-strict head replacement.

    Args:
        model: Model instance receiving weights.
        ckpt_state: State dictionary from a checkpoint.
        strict: Whether to require exact key and shape matches.
        ignore_prefixes: Parameter/buffer prefixes to skip intentionally.

    Returns:
        None.
    """
    if strict and not ignore_prefixes:
        model.load_state_dict(ckpt_state)
        return

    model_state = model.state_dict()
    filtered_state = {}
    skipped = []
    for key, value in ckpt_state.items():
        if any(key.startswith(prefix) for prefix in ignore_prefixes):
            skipped.append(key)
            continue
        if key not in model_state:
            skipped.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped.append(key)
            continue
        filtered_state[key] = value

    load_result = model.load_state_dict(filtered_state, strict=False)
    if skipped:
        preview = ", ".join(skipped[:8])
        suffix = "..." if len(skipped) > 8 else ""
        print(f"Skipped {len(skipped)} checkpoint tensor(s): {preview}{suffix}")
    if load_result.missing_keys:
        preview = ", ".join(load_result.missing_keys[:8])
        suffix = "..." if len(load_result.missing_keys) > 8 else ""
        print(f"Missing {len(load_result.missing_keys)} model tensor(s) after load: {preview}{suffix}")
    if load_result.unexpected_keys:
        preview = ", ".join(load_result.unexpected_keys[:8])
        suffix = "..." if len(load_result.unexpected_keys) > 8 else ""
        print(f"Unexpected {len(load_result.unexpected_keys)} checkpoint tensor(s): {preview}{suffix}")


def _freeze_named_module(model: torch.nn.Module, module_name: str) -> bool:
    """Freeze one named top-level module when it exists.

    Args:
        model: Model containing the module.
        module_name: Attribute name such as ``backbone`` or ``type_classifier``.

    Returns:
        ``True`` if the module exists and was frozen, otherwise ``False``.
    """
    module = getattr(model, module_name, None)
    if module is None:
        return False
    module.requires_grad_(False)
    module.eval()
    return True


def _apply_freeze_config(model: torch.nn.Module, config: DictConfig) -> list[str]:
    """Apply configured module freezing.

    Args:
        model: Model whose top-level modules may be frozen.
        config: Full Hydra config.

    Returns:
        Names of modules that were frozen.
    """
    freeze_cfg = OmegaConf.select(config, "algo.freeze")
    if freeze_cfg is None:
        return []

    frozen = []
    for module_name in ("backbone", "type_classifier", "grasp_type_emb", "output_head"):
        if bool(getattr(freeze_cfg, module_name, False)) and _freeze_named_module(model, module_name):
            frozen.append(module_name)
    if frozen:
        print(f"Frozen module(s): {', '.join(frozen)}")
    return frozen


def _keep_frozen_modules_eval(model: torch.nn.Module, frozen_modules: list[str]) -> None:
    """Keep frozen modules in eval mode after calls to ``model.train()``.

    Args:
        model: Model containing frozen modules.
        frozen_modules: Names returned by ``_apply_freeze_config``.

    Returns:
        None.
    """
    for module_name in frozen_modules:
        module = getattr(model, module_name, None)
        if module is not None:
            module.eval()


def _markdown_text(value) -> str:
    """Escape a value for safe use inside the markdown registry.

    Args:
        value: Object to stringify and place into the markdown registry.

    Returns:
        A markdown-safe string.
    """
    if value is None:
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|").strip()


def _summarize_model_features(config: DictConfig) -> str:
    """Build a compact feature summary for the model registry.

    Args:
        config: Full Hydra config for the current training run.

    Returns:
        Human-readable summary of key options that distinguish this run.
    """
    user_note = OmegaConf.select(config, "model_registry.key_features")
    if user_note:
        return str(user_note)

    parts = [
        f"data={config.data_name}",
        f"algo={config.algo_name}",
        f"max_iter={config.algo.max_iter}",
    ]
    pc_random = OmegaConf.select(config, "data.point_cloud.random_pc_across_sequences")
    if pc_random is not None:
        parts.append(f"random_pc_across_sequences={bool(pc_random)}")
    scale_enabled = OmegaConf.select(config, "data.augmentation.scale.enabled")
    if scale_enabled is not None:
        scale_min = OmegaConf.select(config, "data.augmentation.scale.min")
        scale_max = OmegaConf.select(config, "data.augmentation.scale.max")
        parts.append(f"scale_aug={bool(scale_enabled)}[{scale_min},{scale_max}]")
    train_sampling_unit = OmegaConf.select(config, "data.sampling.train_unit")
    if train_sampling_unit is not None:
        parts.append(f"train_sampling_unit={train_sampling_unit}")
    type_objective = OmegaConf.select(config, "algo.model.type_objective")
    if type_objective is not None:
        scope = OmegaConf.select(config, "algo.supervision.scope")
        negative_policy = OmegaConf.select(config, "algo.supervision.negative_policy")
        parts.append(f"type_objective={type_objective}[scope={scope},negative={negative_policy}]")
    train_type_only = OmegaConf.select(config, "algo.model.train_type_only")
    if train_type_only is not None:
        parts.append(f"train_type_only={bool(train_type_only)}")
    training_mode = OmegaConf.select(config, "algo.training.mode")
    if training_mode is not None:
        parts.append(f"training_mode={training_mode}")
    type_balance_enabled = OmegaConf.select(config, "algo.supervision.balancing.enabled")
    if type_balance_enabled is not None:
        sampler_alpha = OmegaConf.select(config, "algo.supervision.balancing.sampler.alpha")
        loss_beta = OmegaConf.select(config, "algo.supervision.balancing.loss_weight.beta")
        parts.append(
            f"type_balancing={bool(type_balance_enabled)}"
            f"[sampler_alpha={sampler_alpha},loss_beta={loss_beta}]"
        )
    return "; ".join(parts)


def _format_feature_lines(summary: str, width: int = 76) -> list[str]:
    """Format a feature summary as short markdown bullet lines."""
    items = [part.strip() for part in str(summary).split(";") if part.strip()]
    if not items:
        return ["- Key Features:", "  -"]

    lines = ["- Key Features:"]
    for item in items:
        wrapped = textwrap.wrap(
            _markdown_text(item),
            width=width,
            initial_indent="  - ",
            subsequent_indent="    ",
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or ["  -"])
    return lines


def _build_model_registry_entry(
    exp_name: str,
    timestamp: str,
    data_name: str,
    algo_name: str,
    max_iter,
    feature_summary: str,
    notes: str = "",
) -> str:
    """Build one markdown registry entry with wrapped lines."""
    lines = [
        f"## {_markdown_text(exp_name)}",
        f"- Timestamp: `{_markdown_text(timestamp)}`",
        f"- Data: `{_markdown_text(data_name)}`",
        f"- Algo: `{_markdown_text(algo_name)}`",
        f"- Max Iter: `{_markdown_text(max_iter)}`",
        *_format_feature_lines(feature_summary),
        f"- Notes: {_markdown_text(notes)}",
        "",
    ]
    return "\n".join(lines)


def _resolve_model_registry_path(config: DictConfig) -> str:
    """Resolve the markdown model registry path from Hydra config.

    Args:
        config: Full Hydra config for the current training run.

    Returns:
        Absolute path to the markdown registry file.
    """
    registry_path = OmegaConf.select(config, "model_registry.path") or "docs/trained_models.md"
    registry_path = str(registry_path)
    if os.path.isabs(registry_path):
        return registry_path
    try:
        root_dir = hydra.utils.get_original_cwd()
    except ValueError:
        root_dir = os.getcwd()
    return os.path.join(root_dir, registry_path)


def _append_model_registry_entry(config: DictConfig) -> None:
    """Append the current training run to the markdown model registry.

    Args:
        config: Full Hydra config for the current training run.

    Returns:
        None.
    """
    if not bool(OmegaConf.select(config, "model_registry.enabled")):
        return

    path = _resolve_model_registry_path(config)
    registry_dir = os.path.dirname(path)
    if registry_dir:
        os.makedirs(registry_dir, exist_ok=True)
    header = (
        "# Trained Models\n\n"
        "This registry is appended automatically by `task=train` when\n"
        "`model_registry.enabled=true`. Use `model_registry.key_features`\n"
        "to describe intentional differences from previous runs.\n\n"
    )
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)

    exp_name = str(config.wandb.id)
    with open(path, "r", encoding="utf-8") as f:
        existing = f.read()
    if f"| {exp_name} |" in existing or f"## {exp_name}\n" in existing:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = _build_model_registry_entry(
        exp_name=exp_name,
        timestamp=timestamp,
        data_name=config.data_name,
        algo_name=config.algo_name,
        max_iter=config.algo.max_iter,
        feature_summary=_summarize_model_features(config),
        notes="",
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)


def _run_id(config: DictConfig, exp_name: str) -> str:
    """Build the canonical local run id for an experiment name.

    Args:
        config: Full Hydra config with data and algo names.
        exp_name: Short experiment name, without data/algo prefixes.

    Returns:
        Run id used for local output directories.
    """
    return f"{config.data_name}_{config.algo_name}_{exp_name}"


def _set_exp_name(
    config: DictConfig,
    exp_name: str,
    output_exp_name: str | None = None,
    ckpt_subdir: str | None = None,
) -> None:
    """Set the Hydra experiment name, wandb id, and optional local output id.

    Args:
        config: Full Hydra config to mutate for one training stage.
        exp_name: Short experiment name, without data/algo prefixes.
        output_exp_name: Optional short experiment name for local checkpoint
            and sample directories. When omitted, ``exp_name`` is used.
        ckpt_subdir: Optional checkpoint subdirectory under ``ckpts``.

    Returns:
        None.
    """
    config.exp_name = exp_name
    config.wandb.id = _run_id(config, exp_name)
    if output_exp_name is not None:
        OmegaConf.update(config, "wandb.output_id", _run_id(config, output_exp_name), force_add=True)
    if ckpt_subdir is not None:
        OmegaConf.update(config, "wandb.ckpt_subdir", str(ckpt_subdir), force_add=True)


def _format_ckpt_step(step_value) -> str:
    """Format a checkpoint step value as the six-digit Logger filename token.

    Args:
        step_value: Integer-like checkpoint step from Hydra config.

    Returns:
        Six-digit string used in checkpoint filenames, such as ``010000``.
    """
    return str(int(step_value)).zfill(6)


def _stage1_ckpt_path(config: DictConfig, output_exp_name: str) -> str:
    """Build the Stage 1 checkpoint path consumed by Stage 2.

    Args:
        config: Full Hydra config containing output folder and two-stage config.
        output_exp_name: Short experiment name for the shared local output dir.

    Returns:
        Relative or absolute checkpoint path for the configured Stage 1 step.
    """
    ckpt_step = _format_ckpt_step(config.algo.two_stage.stage1.ckpt_step)
    return os.path.join(config.output_folder, _run_id(config, output_exp_name), "ckpts", "stage1", f"step_{ckpt_step}.pth")


def _stage_exp_name(base_exp_name: str, suffix) -> str:
    """Compose the concrete experiment name for one two-stage phase.

    Args:
        base_exp_name: User-provided experiment name.
        suffix: Stage suffix from Hydra config. Empty suffix keeps the base
            name unchanged.

    Returns:
        Concrete experiment name used by Logger and wandb.
    """
    suffix_text = "" if suffix is None else str(suffix).strip()
    if not suffix_text:
        return base_exp_name
    return f"{base_exp_name}_{suffix_text}"


def _two_stage_stage1_type_loss(config: DictConfig) -> float:
    """Read the Stage 1 type-prior loss weight.

    Args:
        config: Hydra config containing ``algo.two_stage.stage1``.

    Returns:
        Floating-point Stage 1 type loss weight. A value less than or equal to
        zero means Stage 1 does not train the type predictor.
    """
    return float(OmegaConf.select(config, "algo.two_stage.stage1.loss_type", default=0.0))


def _training_mode(config: DictConfig) -> str:
    """Resolve and validate the high-level training mode.

    Args:
        config: Hydra config containing the optional ``algo.training.mode``.

    Returns:
        Normalized training mode string.
    """
    mode = str(
        OmegaConf.select(
            config,
            "algo.training.mode",
            default="legacy_shared_encoder_two_stage",
        )
    ).strip()
    supported_modes = {
        "joint_coupled_diffusion",
        "legacy_shared_encoder_two_stage",
        "independent_from_scratch",
        "joint_single_stage",
        "independent_marginals_from_scratch",
        "reverse_independent_from_scratch",
        "two_stage_diffusion_then_frozen_type_head",
    }
    if mode not in supported_modes:
        raise ValueError(f"Unsupported algo.training.mode={mode}. Expected one of {sorted(supported_modes)}")
    return mode


def _build_joint_coupled_training_config(config: DictConfig) -> DictConfig:
    """Build the hard-label, record-uniform single-run Joint training config."""
    joint_config = copy.deepcopy(config)
    joint_config.data.sampling.train_unit = "record_uniform"
    joint_config.data.sampling.pose_group_soft_labels = False
    joint_config.algo.supervision.balancing.enabled = False
    joint_config.algo.supervision.balancing.sampler.enabled = False
    joint_config.algo.supervision.balancing.loss_weight.enabled = False
    if not OmegaConf.select(joint_config, "model_registry.key_features"):
        joint_config.model_registry.key_features = (
            "joint_C_T_coupled_categorical_gaussian_diffusion_record_uniform_hard_labels"
        )
    return joint_config


def _build_independent_from_scratch_config(
    config: DictConfig,
    branch_name: str,
    exp_name: str,
) -> DictConfig:
    """Create one branch config for the independent from-scratch route.

    Args:
        config: User-provided base config.
        branch_name: Either ``diffusion`` or ``type``.
        exp_name: Concrete experiment name for this independent branch.

    Returns:
        Deep-copied config mutated for one independent run.
    """
    branch_config = copy.deepcopy(config)
    _set_exp_name(branch_config, exp_name)
    branch_config.algo.two_stage.enabled = False
    branch_config.ckpt = None
    branch_config.resume = False
    branch_config.wandb.resume = False

    branch_schedule = OmegaConf.select(branch_config, f"algo.training.independent.{branch_name}")
    if branch_schedule is None:
        raise ValueError(f"Missing algo.training.independent.{branch_name} config")
    branch_config.algo.max_iter = int(branch_schedule.max_iter)
    branch_config.algo.save_every = int(branch_schedule.save_every)
    branch_config.algo.val_every = int(branch_schedule.val_every)
    branch_config.algo.lr = float(branch_schedule.lr)
    branch_config.algo.lr_min = float(branch_schedule.lr_min)

    if branch_name == "diffusion":
        branch_config.algo.model.train_type_only = False
        branch_config.algo.loss_weight.loss_diffusion = 1.0
        branch_config.algo.loss_weight.loss_type = 0.0
        branch_config.algo.freeze.backbone = False
        branch_config.algo.freeze.type_classifier = True
        branch_config.algo.freeze.grasp_type_emb = False
        branch_config.algo.freeze.output_head = False
        branch_config.model_registry.key_features = (
            "independent_from_scratch_diffusion_only_shared_arch_"
            f"{branch_config.algo.max_iter}iter_record_uniform_soft_labels"
        )
        return branch_config

    if branch_name == "type":
        branch_config.algo.model.train_type_only = True
        branch_config.algo.loss_weight.loss_diffusion = 0.0
        branch_config.algo.loss_weight.loss_type = 1.0
        branch_config.algo.freeze.backbone = False
        branch_config.algo.freeze.type_classifier = False
        branch_config.algo.freeze.grasp_type_emb = True
        branch_config.algo.freeze.output_head = True
        branch_config.model_registry.key_features = (
            "independent_from_scratch_type_only_shared_arch_"
            f"{branch_config.algo.max_iter}iter_record_uniform_soft_labels"
        )
        return branch_config

    raise ValueError(f"Unsupported independent branch_name={branch_name}")


def _independent_from_scratch_branches(config: DictConfig) -> list[str]:
    """Resolve which independent branches should run in this launch.

    Args:
        config: Hydra config containing ``algo.training.independent.run``.

    Returns:
        Ordered branch list. ``both`` runs ``type`` first because it is the
        faster branch to inspect before launching the slower diffusion run.
    """
    run_mode = str(
        OmegaConf.select(config, "algo.training.independent.run", default="both")
    ).strip()
    if run_mode == "both":
        return ["type", "diffusion"]
    if run_mode in {"type", "diffusion"}:
        return [run_mode]
    raise ValueError(
        f"Unsupported algo.training.independent.run={run_mode}. "
        "Expected one of ['both', 'type', 'diffusion']"
    )


def _independent_marginal_training_branches(config: DictConfig) -> list[str]:
    """Resolve which strict Independent marginal branches should run."""
    run_mode = str(OmegaConf.select(config, "algo.training.run", default="both")).strip()
    if run_mode == "both":
        return ["mode_marginal", "pose_marginal"]
    if run_mode in {"mode_marginal", "pose_marginal"}:
        return [run_mode]
    raise ValueError(
        f"Unsupported algo.training.run={run_mode}. "
        "Expected one of ['both', 'mode_marginal', 'pose_marginal']"
    )


def _build_independent_marginal_branch_config(
    config: DictConfig,
    branch_name: str,
    exp_name: str,
) -> DictConfig:
    """Create one isolated strict Independent marginal training config."""
    if branch_name not in {"mode_marginal", "pose_marginal"}:
        raise ValueError(f"Unsupported Independent branch {branch_name!r}")
    branch_config = copy.deepcopy(config)
    _set_exp_name(branch_config, exp_name)
    branch_config.ckpt = None
    branch_config.resume = False
    branch_config.wandb.resume = False

    schedule = OmegaConf.select(branch_config, f"algo.training.{branch_name}")
    model_config = OmegaConf.select(branch_config, f"algo.models.{branch_name}")
    if schedule is None or model_config is None:
        raise ValueError(f"Missing Independent schedule or model config for {branch_name}")
    branch_config.algo.model = copy.deepcopy(model_config)
    branch_config.algo.max_iter = int(schedule.max_iter)
    branch_config.algo.save_every = int(schedule.save_every)
    branch_config.algo.val_every = int(schedule.val_every)
    branch_config.algo.lr = float(schedule.lr)
    branch_config.algo.lr_min = float(schedule.lr_min)

    branch_config.data.sampling.train_unit = "record_uniform"
    branch_config.algo.supervision.balancing.enabled = False
    branch_config.algo.supervision.balancing.sampler.enabled = False
    branch_config.algo.supervision.balancing.loss_weight.enabled = False
    if branch_name == "mode_marginal":
        branch_config.data.sampling.pose_group_soft_labels = True
        branch_config.algo.loss_weight.loss_type = 1.0
        branch_config.algo.loss_weight.loss_diffusion = 0.0
        key_features = "independent_C_T_object_only_mode_marginal_soft_label_ce"
    else:
        branch_config.data.sampling.pose_group_soft_labels = False
        branch_config.algo.loss_weight.loss_type = 0.0
        branch_config.algo.loss_weight.loss_diffusion = 1.0
        key_features = "independent_C_T_object_only_pose_marginal_diffusion"
    branch_config.model_registry.key_features = f"{key_features}_{branch_config.algo.max_iter}iter"
    return branch_config


def _reverse_training_branches(config: DictConfig) -> list[str]:
    """Resolve which Reverse T-to-C branches should run.

    Args:
        config: Full config containing ``algo.training.run``.

    Returns:
        Ordered branch list. ``both`` trains the marginal generator first.
    """
    run_mode = str(OmegaConf.select(config, "algo.training.run", default="both")).strip()
    if run_mode == "both":
        return ["pose_marginal", "type_posterior"]
    if run_mode in {"pose_marginal", "type_posterior"}:
        return [run_mode]
    raise ValueError(
        f"Unsupported algo.training.run={run_mode}. "
        "Expected one of ['both', 'pose_marginal', 'type_posterior']"
    )


def _build_reverse_branch_config(config: DictConfig, branch_name: str, exp_name: str) -> DictConfig:
    """Create one isolated Reverse model training config.

    Args:
        config: Base ``humanMultiReverse`` config.
        branch_name: ``pose_marginal`` or ``type_posterior``.
        exp_name: Concrete experiment name for this branch.

    Returns:
        Deep-copied branch config with its own model and schedule.
    """
    if branch_name not in {"pose_marginal", "type_posterior"}:
        raise ValueError(f"Unsupported Reverse branch {branch_name!r}")
    branch_config = copy.deepcopy(config)
    _set_exp_name(branch_config, exp_name)
    branch_config.ckpt = None
    branch_config.resume = False
    branch_config.wandb.resume = False

    schedule = OmegaConf.select(branch_config, f"algo.training.{branch_name}")
    model_config = OmegaConf.select(branch_config, f"algo.models.{branch_name}")
    if schedule is None or model_config is None:
        raise ValueError(f"Missing Reverse schedule or model config for {branch_name}")
    branch_config.algo.model = copy.deepcopy(model_config)
    branch_config.algo.max_iter = int(schedule.max_iter)
    branch_config.algo.save_every = int(schedule.save_every)
    branch_config.algo.val_every = int(schedule.val_every)
    branch_config.algo.lr = float(schedule.lr)
    branch_config.algo.lr_min = float(schedule.lr_min)

    branch_config.data.sampling.train_unit = "record_uniform"
    branch_config.data.sampling.pose_group_soft_labels = False
    branch_config.algo.supervision.balancing.enabled = False
    branch_config.algo.supervision.balancing.sampler.enabled = False
    branch_config.algo.supervision.balancing.loss_weight.enabled = False

    if branch_name == "pose_marginal":
        branch_config.algo.loss_weight.loss_diffusion = 1.0
        branch_config.algo.loss_weight.loss_type = 0.0
        key_features = "reverse_T_to_C_pose_marginal_object_conditioned_diffusion_only"
    else:
        branch_config.algo.loss_weight.loss_diffusion = 0.0
        branch_config.algo.loss_weight.loss_type = 1.0
        key_features = "reverse_T_to_C_pose_conditioned_hard_label_posterior"
    branch_config.model_registry.key_features = f"{key_features}_{branch_config.algo.max_iter}iter"
    return branch_config


def _build_two_stage_config(
    config: DictConfig,
    stage_name: str,
    exp_name: str,
    output_exp_name: str,
    ckpt_path: str | None = None,
) -> DictConfig:
    """Create a stage-specific config for the integrated two-stage trainer.

    Args:
        config: User-provided base config.
        stage_name: Either ``stage1`` or ``stage2``.
        exp_name: Short experiment name for this stage's wandb/registry id.
        output_exp_name: Short experiment name for the shared local output dir.
        ckpt_path: Stage 1 checkpoint path for Stage 2, otherwise ``None``.

    Returns:
        A deep-copied and mutated config for one concrete training stage.
    """
    stage_config = copy.deepcopy(config)
    _set_exp_name(stage_config, exp_name, output_exp_name=output_exp_name, ckpt_subdir=stage_name)
    stage_config.algo.two_stage.enabled = False

    if stage_name == "stage1":
        stage1_loss_type = _two_stage_stage1_type_loss(stage_config)
        stage1_loss_diffusion = float(
            OmegaConf.select(stage_config, "algo.two_stage.stage1.loss_diffusion", default=1.0)
        )
        stage_config.ckpt = config.ckpt
        stage_config.algo.model.train_type_only = False
        stage_config.algo.loss_weight.loss_diffusion = stage1_loss_diffusion
        stage_config.algo.loss_weight.loss_type = stage1_loss_type
        stage_config.algo.freeze.type_classifier = stage1_loss_type <= 0.0
        stage_config.algo.freeze.backbone = False
        stage_config.algo.freeze.grasp_type_emb = False
        stage_config.algo.freeze.output_head = False
        stage_config.model_registry.key_features = (
            f"integrated_stage1_diffusion_encoder_10000iter_save2500_"
            f"loss_diffusion{stage1_loss_diffusion:g}_loss_type{stage1_loss_type:g}_"
            "record_uniform_soft_labels"
        )
        return stage_config

    if stage_name == "stage2":
        if ckpt_path is None:
            raise ValueError("Stage 2 requires a Stage 1 checkpoint path")
        reset_type_head = _two_stage_stage1_type_loss(stage_config) <= 0.0
        stage_config.ckpt = ckpt_path
        stage_config.resume = False
        stage_config.wandb.resume = False
        stage_config.algo.model.train_type_only = True
        stage_config.algo.loss_weight.loss_diffusion = 0.0
        stage_config.algo.loss_weight.loss_type = 1.0
        stage_config.algo.ckpt_load.load_optimizer = False
        stage_config.algo.ckpt_load.reset_iter = True
        stage_config.algo.ckpt_load.strict_model = not reset_type_head
        stage_config.algo.ckpt_load.ignore_prefixes = ["type_classifier"] if reset_type_head else []
        stage_config.algo.freeze.backbone = True
        stage_config.algo.freeze.type_classifier = False
        stage_config.algo.freeze.grasp_type_emb = True
        stage_config.algo.freeze.output_head = True
        stage_config.algo.max_iter = stage_config.algo.two_stage.stage2.max_iter
        stage_config.algo.save_every = stage_config.algo.two_stage.stage2.save_every
        stage_config.algo.val_every = stage_config.algo.two_stage.stage2.val_every
        stage_config.algo.lr = stage_config.algo.two_stage.stage2.lr
        stage_config.algo.lr_min = stage_config.algo.two_stage.stage2.lr_min
        stage_config.model_registry.key_features = (
            "integrated_stage2_frozen_stage1_encoder_train_yaml_configured_"
            f"{'reset' if reset_type_head else 'continued'}_type_head_record_uniform_soft_labels"
        )
        return stage_config

    raise ValueError(f"Unsupported two-stage stage_name={stage_name}")


def _task_train_two_stage(config: DictConfig, stage1_loss_type_override: float | None = None) -> None:
    """Run Stage 1 and Stage 2 sequentially from one task=train launch.

    Args:
        config: User-provided base Hydra config with ``algo.two_stage.enabled``.
        stage1_loss_type_override: Optional override for the Stage 1 type loss.
            ``0.0`` reproduces the pure-diffusion Stage 1 route.

    Returns:
        None.
    """
    if stage1_loss_type_override is not None:
        config = copy.deepcopy(config)
        config.algo.two_stage.stage1.loss_type = float(stage1_loss_type_override)

    base_exp_name = str(config.exp_name)
    stage1_exp_name = _stage_exp_name(base_exp_name, config.algo.two_stage.stage1.exp_suffix)
    stage2_exp_name = _stage_exp_name(base_exp_name, config.algo.two_stage.stage2.exp_suffix)
    ckpt_path = _stage1_ckpt_path(config, base_exp_name)

    print(f"Two-stage training enabled: Stage 1 exp_name={stage1_exp_name}, output_exp_name={base_exp_name}")
    stage1_config = _build_two_stage_config(config, "stage1", stage1_exp_name, output_exp_name=base_exp_name)
    _task_train_single(stage1_config)
    wandb.finish()

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Stage 1 checkpoint not found for Stage 2: {ckpt_path}")

    print(f"Two-stage training enabled: Stage 2 exp_name={stage2_exp_name}, output_exp_name={base_exp_name}, ckpt={ckpt_path}")
    stage2_config = _build_two_stage_config(
        config,
        "stage2",
        stage2_exp_name,
        output_exp_name=base_exp_name,
        ckpt_path=ckpt_path,
    )
    _task_train_single(stage2_config)
    wandb.finish()


def _task_train_independent_from_scratch(config: DictConfig) -> None:
    """Train diffusion and type models independently from scratch.

    Args:
        config: User-provided base config. This route ignores ``config.ckpt``
            and launches two fresh runs with separate experiment names.

    Returns:
        None.
    """
    base_exp_name = str(config.exp_name)
    diffusion_suffix = OmegaConf.select(config, "algo.training.independent.diffusion_exp_suffix", default="diffusion")
    type_suffix = OmegaConf.select(config, "algo.training.independent.type_exp_suffix", default="type")
    exp_name_by_branch = {
        "diffusion": _stage_exp_name(base_exp_name, diffusion_suffix),
        "type": _stage_exp_name(base_exp_name, type_suffix),
    }

    for branch_name in _independent_from_scratch_branches(config):
        branch_exp_name = exp_name_by_branch[branch_name]
        print(f"Independent-from-scratch mode: {branch_name} exp_name={branch_exp_name}")
        branch_config = _build_independent_from_scratch_config(config, branch_name, branch_exp_name)
        _task_train_single(branch_config)
        wandb.finish()


def _task_train_reverse_independent_from_scratch(config: DictConfig) -> None:
    """Train the Reverse pose marginal and posterior as separate fresh runs."""
    base_exp_name = str(config.exp_name)
    for branch_name in _reverse_training_branches(config):
        suffix = str(OmegaConf.select(config, f"algo.training.{branch_name}.exp_suffix"))
        branch_exp_name = _stage_exp_name(base_exp_name, suffix)
        print(f"Reverse T-to-C branch: {branch_name} exp_name={branch_exp_name}")
        branch_config = _build_reverse_branch_config(config, branch_name, branch_exp_name)
        _task_train_single(branch_config)
        wandb.finish()


def _task_train_independent_marginals_from_scratch(config: DictConfig) -> None:
    """Train strict Independent mode and pose marginals as separate runs."""
    base_exp_name = str(config.exp_name)
    for branch_name in _independent_marginal_training_branches(config):
        suffix = str(OmegaConf.select(config, f"algo.training.{branch_name}.exp_suffix"))
        branch_exp_name = _stage_exp_name(base_exp_name, suffix)
        print(f"Independent C-T branch: {branch_name} exp_name={branch_exp_name}")
        branch_config = _build_independent_marginal_branch_config(
            config,
            branch_name,
            branch_exp_name,
        )
        _task_train_single(branch_config)
        wandb.finish()


def _task_train_single(config: DictConfig):
    resolve_type_supervision_config(config)
    set_seed(config.seed)
    logger = Logger(config)
    _append_model_registry_entry(config)
    train_loader, val_loader = create_train_dataloader(config)
    timing_cfg = config.timing
    timing_enabled = timing_cfg.enabled
    timing_sync_cuda = timing_enabled and timing_cfg.cuda_sync
    timing_buffer = {
        "data_time": 0.0,
        "forward_time": 0.0,
        "backward_time": 0.0,
        "optim_time": 0.0,
        "iter_time": 0.0,
        "count": 0,
    }

    model = eval(config.algo.model.name)(config.algo.model)

    # load ckpt if exists
    optimizer_state = None
    if config.ckpt is not None:
        ckpt = torch.load(config.ckpt, map_location="cpu")
        ckpt_load_cfg = OmegaConf.select(config, "algo.ckpt_load")
        strict_model = bool(getattr(ckpt_load_cfg, "strict_model", True)) if ckpt_load_cfg is not None else True
        ignore_prefixes = _as_list(getattr(ckpt_load_cfg, "ignore_prefixes", [])) if ckpt_load_cfg is not None else []
        load_optimizer = bool(getattr(ckpt_load_cfg, "load_optimizer", True)) if ckpt_load_cfg is not None else True
        reset_iter = bool(getattr(ckpt_load_cfg, "reset_iter", False)) if ckpt_load_cfg is not None else False
        _load_model_state(model, ckpt["model"], strict=strict_model, ignore_prefixes=ignore_prefixes)
        model.to(config.device)
        optimizer_state = ckpt.get("optimizer") if load_optimizer else None
        cur_iter = 0 if reset_iter else ckpt["iter"]
        print(f"loaded ckpt from {config.ckpt}; reset_iter={reset_iter}; load_optimizer={load_optimizer}")
    else:
        cur_iter = 0

    frozen_modules = _apply_freeze_config(model, config)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters remain after applying freeze config")

    optimizer = torch.optim.AdamW(trainable_params, lr=config.algo.lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    scheduler = CosineAnnealingLR(optimizer, config.algo.max_iter, eta_min=config.algo.lr_min)
    for _ in range(cur_iter):
        scheduler.step()

    # training
    model.to(config.device)
    model.train()
    _keep_frozen_modules_eval(model, frozen_modules)

    for it in trange(cur_iter, config.algo.max_iter):
        _keep_frozen_modules_eval(model, frozen_modules)
        if timing_enabled:
            _maybe_cuda_sync(config.device, timing_sync_cuda)
            iter_start = time.perf_counter()
        optimizer.zero_grad()

        if timing_enabled:
            _maybe_cuda_sync(config.device, timing_sync_cuda)
            data_start = time.perf_counter()
        data = train_loader.get()

        if timing_enabled:
            _maybe_cuda_sync(config.device, timing_sync_cuda)
            timing_buffer["data_time"] += time.perf_counter() - data_start
            forward_start = time.perf_counter()
        result_dict = model(data)

        if timing_enabled:
            _maybe_cuda_sync(config.device, timing_sync_cuda)
            timing_buffer["forward_time"] += time.perf_counter() - forward_start
        loss = 0
        for k, v in result_dict.items():
            if k in config.algo.loss_weight:
                loss += config.algo.loss_weight[k] * v
            elif "loss" in k:
                print(f"{k} is not used in loss!")

        if timing_enabled:
            _maybe_cuda_sync(config.device, timing_sync_cuda)
            backward_start = time.perf_counter()
        loss.backward()
        debug_flag = False
        for p in model.parameters():
            if hasattr(p, "grad") and p.grad is not None:
                try:
                    if torch.isnan(p.grad).any():
                        p.grad.zero_()
                        debug_flag = True
                except Exception as e:
                    print("Wrong p", p)
                    print("grad", p.grad)
                    print("shape", p.shape)
                    raise e

        if debug_flag:
            print("grad is nan!")
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.algo.grad_clip)

        if timing_enabled:
            _maybe_cuda_sync(config.device, timing_sync_cuda)
            timing_buffer["backward_time"] += time.perf_counter() - backward_start
        if it == 0:
            if timing_enabled:
                for key in timing_buffer:
                    timing_buffer[key] = 0.0 if key != "count" else 0
            continue

        if timing_enabled:
            _maybe_cuda_sync(config.device, timing_sync_cuda)
            optim_start = time.perf_counter()
        optimizer.step()
        scheduler.step()

        if timing_enabled:
            _maybe_cuda_sync(config.device, timing_sync_cuda)
            timing_buffer["optim_time"] += time.perf_counter() - optim_start
            timing_buffer["iter_time"] += time.perf_counter() - iter_start
            timing_buffer["count"] += 1

        result_dict["lr"] = torch.tensor(scheduler.get_last_lr())
        if timing_enabled and (it + 1) % config.algo.log_every == 0 and timing_buffer["count"] > 0:
            result_dict.update(
                {
                    "time_data": torch.tensor(timing_buffer["data_time"] / timing_buffer["count"]),
                    "time_forward": torch.tensor(timing_buffer["forward_time"] / timing_buffer["count"]),
                    "time_backward": torch.tensor(timing_buffer["backward_time"] / timing_buffer["count"]),
                    "time_optim": torch.tensor(timing_buffer["optim_time"] / timing_buffer["count"]),
                    "time_iter": torch.tensor(timing_buffer["iter_time"] / timing_buffer["count"]),
                }
            )
        if (it + 1) % config.algo.log_every == 0:
            logger.log({k: v.mean().item() for k, v in result_dict.items()}, "train", it)
            if timing_enabled:
                for key in timing_buffer:
                    timing_buffer[key] = 0.0 if key != "count" else 0

        if (it + 1) % config.algo.save_every == 0:
            logger.save(
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    iter=it + 1,
                ),
                it + 1,
            )

        if (it + 1) % config.algo.val_every == 0:
            with torch.no_grad():
                model.eval()
                result_dicts = []
                for _ in range(config.algo.val_num):
                    data = val_loader.get()
                    result_dict = model(data)
                    result_dicts.append(result_dict)
                logger.log(
                    {
                        k: torch.cat([(dic[k] if len(dic[k].shape) else dic[k][None]) for dic in result_dicts]).mean()
                        for k in result_dicts[0].keys()
                    },
                    "eval",
                    it,
                )
                model.train()
                _keep_frozen_modules_eval(model, frozen_modules)

    return


def task_train(config: DictConfig):
    """Run training according to the configured Human Prior training mode.

    Args:
        config: Full Hydra config for ``task=train``.

    Returns:
        None.
    """
    mode = _training_mode(config)

    if mode == "legacy_shared_encoder_two_stage":
        if bool(OmegaConf.select(config, "algo.two_stage.enabled", default=False)):
            _task_train_two_stage(config)
            return
        _task_train_single(config)
        return

    if mode == "independent_from_scratch":
        _task_train_independent_from_scratch(config)
        return

    if mode == "independent_marginals_from_scratch":
        _task_train_independent_marginals_from_scratch(config)
        return

    if mode == "reverse_independent_from_scratch":
        _task_train_reverse_independent_from_scratch(config)
        return

    if mode == "joint_coupled_diffusion":
        _task_train_single(_build_joint_coupled_training_config(config))
        return

    if mode == "joint_single_stage":
        config = copy.deepcopy(config)
        config.algo.two_stage.enabled = False
        config.algo.model.train_type_only = False
        config.algo.freeze.backbone = False
        config.algo.freeze.type_classifier = False
        config.algo.freeze.grasp_type_emb = False
        config.algo.freeze.output_head = False
        if not OmegaConf.select(config, "model_registry.key_features"):
            config.model_registry.key_features = (
                "joint_single_stage_shared_encoder_type_and_diffusion_record_uniform_soft_labels"
            )
        _task_train_single(config)
        return

    if mode == "two_stage_diffusion_then_frozen_type_head":
        config = copy.deepcopy(config)
        config.algo.two_stage.enabled = True
        _task_train_two_stage(config, stage1_loss_type_override=0.0)
        return

    raise ValueError(f"Unhandled training mode: {mode}")
