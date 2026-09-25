"""Explicit optional dependency boundaries; no replacement algorithms."""

from importlib import import_module


def require_module(name, purpose):
    try:
        return import_module(name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"{purpose} requires {name} (missing {exc.name}). "
            "See docs/installation.md; no fallback algorithm is used."
        ) from exc
