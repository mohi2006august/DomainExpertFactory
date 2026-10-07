"""Locate bundled prompts, configs and templates in both editable and wheel installs."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parents[1]


def data_path(relative: str) -> Path:
    """Return the path of a bundled resource directory such as ``prompts``."""
    for base in (_PKG_DIR / "_data", _REPO_ROOT):
        candidate = base / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"bundled resource {relative!r} not found next to {_PKG_DIR}")


@cache
def load_config(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml`` (cached; treat the result as read-only)."""
    with (data_path("configs") / f"{name}.yaml").open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"configs/{name}.yaml must contain a mapping")
    return data


@cache
def load_prompt_yaml(name: str) -> dict[str, Any]:
    with (data_path("prompts") / f"{name}.yaml").open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"prompts/{name}.yaml must contain a mapping")
    return data


@cache
def _env(directory: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(data_path(directory))),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
        autoescape=directory.startswith("reports"),
    )


def render_prompt(template: str, **context: Any) -> str:
    return _env("prompts").get_template(template).render(**context).strip()


def report_environment() -> Environment:
    return _env("reports/templates")
