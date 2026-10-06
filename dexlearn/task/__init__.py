"""Load only the selected task, including legacy task imports."""

from importlib import import_module

TASK_NAMES = (
    "train", "human_preprocess", "sample", "human_prior_format", "visualize",
    "type_eval", "diffusion_eval", "scene_budget", "obj_human_prior_export",
    "visualize_human_prior", "robot_type_eval",
)
__all__ = [f"task_{name}" for name in TASK_NAMES]


def get_task(name):
    """Return a task entry point without importing unrelated workflows."""
    if name not in TASK_NAMES:
        raise ValueError(f"Unknown task {name!r}; expected one of {TASK_NAMES}")
    try:
        module = import_module(f"dexlearn.task.{name}")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Task {name!r} requires missing dependency {exc.name!r}. "
            "See README.md (Installation) for the corresponding runtime profile."
        ) from exc
    return getattr(module, f"task_{name}")


def run_task(name, config):
    """Dispatch a validated task name without evaluating arbitrary code."""
    return get_task(name)(config)


def __getattr__(name):
    if name.startswith("task_") and name[5:] in TASK_NAMES:
        def entry(config):
            return run_task(name[5:], config)
        entry.__name__ = name
        return entry
    raise AttributeError(name)
