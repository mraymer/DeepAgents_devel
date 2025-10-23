import yaml
from jinja2 import Template
from pathlib import Path
from typing import Any, Dict, List, Callable
import importlib
import functools

def _resolve_callable(spec: str) -> Callable:
    """
    Resolve "package.module:attr" to a Python callable.
    """
    if ":" not in spec:
        raise ValueError(f"Tool spec must be 'module.path:callable', got: {spec}")
    mod_path, attr_name = spec.split(":", 1)
    module = importlib.import_module(mod_path)
    fn = getattr(module, attr_name)
    if not callable(fn):
        raise TypeError(f"{spec} resolved to non-callable object")
    return fn

def _resolve_tools(raw_tools: list[Any]) -> list[Callable]:
    """
    Accepts either:
      - ["pkg.mod:func", "pkg.mod:other_func"]
      - [{"name": "pkg.mod:func", "kwargs": {...}}, ...]
    Returns callables; if kwargs provided, returns a functools.partial.
    """
    resolved: list[Callable] = []
    for item in raw_tools:
        if isinstance(item, str):
            resolved.append(_resolve_callable(item))
        elif isinstance(item, dict) and "name" in item:
            fn = _resolve_callable(item["name"])
            kwargs = item.get("kwargs") or {}
            if kwargs:
                fn = functools.partial(fn, **kwargs)
            resolved.append(fn)
        else:
            raise ValueError(f"Unrecognized tool spec: {item!r}")
    return resolved

def make_all_subagents(params: Dict[str, Any], yaml_path: str = "subagents.yaml") -> List[dict]:
    """
    Build dictionary-based subagents:
      - render system prompts via Jinja2 with runtime params
      - resolve tool strings to real callables (optionally partials with kwargs)
    """
    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    subagents: List[dict] = []

    for cfg in data:
        raw_prompt = cfg.get("system_prompt", "")
        rendered_prompt = Template(raw_prompt).render(**params)
        tools_raw = cfg.get("tools", [])
        tools_resolved = _resolve_tools(tools_raw)

        subagents.append({
            "name": cfg["name"],
            "description": cfg.get("description", ""),
            "system_prompt": rendered_prompt,
            "tools": tools_resolved,
            "model": cfg.get("model"),
        })

    return subagents
