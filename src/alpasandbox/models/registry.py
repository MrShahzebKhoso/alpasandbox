"""Model registry: built-ins, `alpasandbox.models` entry points, and `module:Class` specs.

Third-party packages can add a model without touching this repo:

    # pyproject.toml of your package
    [project.entry-points."alpasandbox.models"]
    mymodel = "my_pkg.adapter:MyPolicy"

then `alpasandbox app --model mymodel`. Or skip packaging: `--model my_pkg.adapter:MyPolicy`.
"""
from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .base import Policy


@dataclass(frozen=True)
class ModelEntry:
    name: str
    display_name: str
    target: str                  # "module:Class"
    requires: str = ""           # import name that must be installed for local use
    install_hint: str = ""
    gpu: str = ""
    summary: str = ""


BUILTIN: dict[str, ModelEntry] = {
    "edge": ModelEntry("edge", "Alpamayo 2 Edge", "alpasandbox.models.alpamayo_edge:AlpamayoEdge",
                       "alpamayo2_edge", 'pip install "alpamayo2-edge @ git+https://github.com/NVlabs/alpamayo2-edge"',
                       ">= 16 GB VRAM", "2B reasoning VLA, driving + VQA"),
    "super": ModelEntry("super", "Alpamayo 2 Super", "alpasandbox.models.alpamayo_super:AlpamayoSuper",
                        "alpamayo2_super", 'pip install "alpamayo2-super @ git+https://github.com/NVlabs/alpamayo2"',
                        "~80 GB VRAM", "34B multi-task VLA, driving + VQA (experimental)"),
    "oracle": ModelEntry("oracle", "Oracle (ground truth)", "alpasandbox.models.fixtures:OraclePolicy",
                         summary="Rule-based test fixture; reads ground truth; no GPU"),
    "cruise": ModelEntry("cruise", "Cruise (blind baseline)", "alpasandbox.models.fixtures:CruisePolicy",
                         summary="Constant-speed straight line; no GPU"),
}
ALIASES = {"alpamayo": "edge", "alpamayo-edge": "edge", "alpamayo2-edge": "edge", "alpamayo2_edge": "edge",
           "alpamayo-super": "super", "alpamayo2-super": "super", "alpamayo2_super": "super"}


def _entry_points() -> dict[str, Any]:
    try:
        from importlib.metadata import entry_points
        return {ep.name: ep for ep in entry_points(group="alpasandbox.models")}
    except Exception:
        return {}


def canonical(name: str) -> str:
    return ALIASES.get(name.lower(), name) if ":" not in name else name


def _import(target: str) -> Callable[..., Policy]:
    mod, _, attr = target.partition(":")
    return getattr(importlib.import_module(mod), attr)


def load_local(name: str, settings: dict | None = None) -> Policy:
    """Instantiate a model in this process."""
    name, settings = canonical(name), dict(settings or {})
    if name in BUILTIN:
        entry = BUILTIN[name]
        if entry.requires and importlib.util.find_spec(entry.requires) is None:
            raise ModuleNotFoundError(
                f"{entry.display_name} needs the '{entry.requires}' package in this Python environment.\n"
                f"  Install: {entry.install_hint}\n"
                "  Or run it elsewhere with --runtime subprocess | wsl | remote (see docs/runtimes.md).")
        return _import(entry.target)(**settings)
    eps = _entry_points()
    if name in eps:
        return eps[name].load()(**settings)
    if ":" in name:
        return _import(name)(**settings)
    raise ValueError(f"unknown model '{name}'. Built-in: {sorted(BUILTIN)}; plugins: {sorted(eps)}; "
                     "or pass 'package.module:ClassName'")


def make_policy(model: str = "oracle", runtime: str = "local", settings: dict | None = None,
                runtime_cfg: dict | None = None, log=None) -> Policy:
    """Create a model on the chosen runtime: local | subprocess | wsl | remote."""
    from . import runtimes

    model, rcfg = canonical(model), dict(runtime_cfg or {})
    # Legacy specs from alpasandbox 0.1: "remote:<url>" and "wsl:<distro>".
    if model.startswith("remote:") and "://" in model:
        runtime, rcfg["url"], model = "remote", model[len("remote:"):], "edge"
    elif model.startswith("wsl:"):
        runtime, rcfg["distro"], model = "wsl", model[len("wsl:"):] or rcfg.get("distro", ""), "edge"
    if runtime == "local":
        return load_local(model, settings)
    if runtime == "subprocess":
        return runtimes.start_subprocess_worker(model, settings or {}, rcfg, log)
    if runtime == "wsl":
        return runtimes.start_wsl_worker(model, settings or {}, rcfg, log)
    if runtime == "remote":
        if not rcfg.get("url"):
            raise ValueError("remote runtime needs a URL (--remote-url http://host:8100)")
        return runtimes.RemotePolicy(rcfg["url"], float(rcfg.get("timeout", 300)), rcfg.get("token"))
    raise ValueError(f"unknown runtime '{runtime}' (local | subprocess | wsl | remote)")


def available_models() -> list[dict]:
    rows = []
    for e in BUILTIN.values():
        installed = (not e.requires) or importlib.util.find_spec(e.requires) is not None
        rows.append({"name": e.name, "display_name": e.display_name, "source": "built-in",
                     "installed_here": installed, "gpu": e.gpu, "summary": e.summary, "install": e.install_hint})
    for n, ep in _entry_points().items():
        rows.append({"name": n, "display_name": n, "source": f"plugin ({ep.value})", "installed_here": True,
                     "gpu": "", "summary": "", "install": ""})
    return rows
